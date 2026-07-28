"""FastAPI inference service.

Endpoints:

``POST /generate``          batched text generation
``POST /generate/stream``   server-sent events; watch the text denoise live
``GET  /health``            liveness -- the process is up and the loop responsive
``GET  /ready``             readiness -- a model is loaded and traffic can be served
``GET  /model``             metadata about the served checkpoint
``GET  /metrics``           Prometheus exposition
``POST /admin/swap``        load a different model version
``POST /admin/rollback``    return to the previously served version

**Liveness and readiness are deliberately separate.** They answer different
questions and an orchestrator reacts to them differently: failing liveness gets the
container *restarted*, failing readiness gets it *removed from the load balancer*.
Collapsing them means a pod that is merely still loading a checkpoint gets killed
and restarted forever, never finishing. So ``/health`` returns 200 as soon as the
process can answer, while ``/ready`` returns 503 until weights are resident.

Streaming is exempt from dynamic batching: an SSE connection owns a generator for
its whole lifetime, and interleaving several of those into shared batch steps would
mean the slowest client dictates every other client's frame rate.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from aether.log import configure_logging, get_logger
from aether.serve import metrics as M
from aether.serve.batcher import BatchKey, DynamicBatcher
from aether.serve.registry import LoadedModel, ModelRegistry
from aether.serve.schemas import (
    ErrorResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelInfo,
    ReadyResponse,
    SwapRequest,
)

logger = get_logger("serve")


class ServerSettings:
    """Runtime configuration, supplied by the CLI or defaults."""

    def __init__(
        self,
        model_version: str | None = None,
        tokenizer: str = "gpt2",
        device: str = "auto",
        max_batch_size: int = 32,
        max_wait_ms: float = 20.0,
        schedule: str = "linear",
        cache_dir: str | None = None,
    ) -> None:
        self.model_version = model_version
        self.tokenizer = tokenizer
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.schedule = schedule
        self.cache_dir = cache_dir


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    """Build the ASGI application.

    A factory rather than a module-level singleton so tests can spin up isolated
    instances with different settings in one process.
    """
    cfg = settings or ServerSettings()
    registry = ModelRegistry(
        tokenizer_name=cfg.tokenizer, device=cfg.device, cache_dir=cfg.cache_dir
    )
    started_at = time.time()

    def run_batch(key: BatchKey, total: int) -> Any:
        """Execute one merged batch. Runs in a worker thread, never the event loop."""
        from aether.diffusion.samplers import sample

        loaded = registry.require()
        M.IN_FLIGHT_BATCH_SIZE.set(total)
        try:
            out = sample(
                loaded.model,
                batch=total,
                length=key.length,
                mask_token_id=loaded.mask_token_id,
                sampler=key.sampler,
                steps=key.steps,
                schedule=cfg.schedule,
                device=loaded.device,
                temperature=key.temperature,
            )
            M.BATCHES_RUN.inc()
            M.NFE_TOTAL.inc(out.nfe)
            M.TOKENS_GENERATED.inc(total * key.length)
            return out
        finally:
            M.IN_FLIGHT_BATCH_SIZE.set(0)

    batcher = DynamicBatcher(
        run_batch=run_batch,
        max_batch_size=cfg.max_batch_size,
        max_wait_ms=cfg.max_wait_ms,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging()
        await batcher.start()
        if cfg.model_version:
            try:
                # Loading blocks; keep it off the event loop so /health stays live.
                await asyncio.to_thread(registry.swap, cfg.model_version)
                M.MODEL_LOADED.set(1)
            except Exception as exc:  # start unready, do not crash
                # Serving unready beats crash-looping: /ready reports the problem
                # and an operator can push a working version via /admin/swap.
                M.ERRORS.labels(kind="startup_load").inc()
                logger.warning("startup_load_failed", version=cfg.model_version, error=str(exc))
        yield
        await batcher.stop()
        M.MODEL_LOADED.set(0)

    app = FastAPI(
        title="Aether Inference API",
        description="Masked diffusion language model served with dynamic batching.",
        version="0.1.0",
        lifespan=lifespan,
        responses={503: {"model": ErrorResponse}},
    )
    app.state.registry = registry
    app.state.batcher = batcher
    app.state.settings = cfg

    def require_model() -> LoadedModel:
        loaded = registry.current
        if loaded is None:
            M.ERRORS.labels(kind="not_ready").inc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="no model loaded; POST /admin/swap or check startup logs",
            )
        return loaded

    # -- health ---------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health() -> HealthResponse:
        """Liveness. 200 whenever the process can answer; never gated on the model."""
        return HealthResponse(uptime_s=time.time() - started_at)

    @app.get("/ready", response_model=ReadyResponse, tags=["ops"])
    async def ready(response: Response) -> ReadyResponse:
        """Readiness. 503 until weights are resident, so traffic is held back."""
        loaded = registry.current
        if loaded is None:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
            return ReadyResponse(ready=False, detail="model not loaded")
        return ReadyResponse(ready=True, model_version=loaded.version)

    @app.get("/metrics", tags=["ops"])
    async def prometheus_metrics() -> Response:
        M.QUEUE_DEPTH.set(batcher.queue_depth)
        M.MODEL_LOADED.set(1 if registry.is_ready else 0)
        return Response(generate_latest(M.REGISTRY), media_type=CONTENT_TYPE_LATEST)

    @app.get("/model", response_model=ModelInfo, tags=["ops"])
    async def model_info(loaded: LoadedModel = Depends(require_model)) -> ModelInfo:
        return ModelInfo(
            model_version=loaded.version,
            params=int(loaded.metadata.get("params", 0)),
            d_model=loaded.config.d_model,
            n_layers=loaded.config.n_layers,
            n_heads=loaded.config.n_heads,
            max_seq_len=loaded.config.max_seq_len,
            vocab_size=loaded.config.vocab_size,
            step=loaded.metadata.get("step"),
            device=str(loaded.device),
        )

    # -- generation -----------------------------------------------------------
    def _validate_length(req: GenerateRequest, loaded: LoadedModel) -> None:
        if req.length > loaded.max_length:
            raise HTTPException(
                status_code=422,  # UNPROCESSABLE_CONTENT; numeric avoids starlette rename churn
                detail=(
                    f"length {req.length} exceeds the model's maximum "
                    f"sequence length of {loaded.max_length}"
                ),
            )

    @app.post("/generate", response_model=GenerateResponse, tags=["inference"])
    async def generate(
        req: GenerateRequest, loaded: LoadedModel = Depends(require_model)
    ) -> GenerateResponse:
        """Generate text. Concurrent requests with matching parameters share a batch."""
        _validate_length(req, loaded)
        start = time.perf_counter()
        key = BatchKey(
            sampler=req.sampler,
            steps=req.steps,
            length=req.length,
            temperature=req.temperature,
        )
        try:
            with M.REQUEST_LATENCY.labels(endpoint="generate").time():
                out, offset, count = await batcher.submit(key, req.n_samples)
        except RuntimeError as exc:
            M.REQUESTS.labels(endpoint="generate", status="error").inc()
            M.ERRORS.labels(kind="batch").inc()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
            ) from exc

        rows = out.tokens[offset : offset + count]
        texts = [loaded.tokenizer.decode(row) for row in rows.tolist()]
        M.REQUESTS.labels(endpoint="generate", status="ok").inc()
        M.QUEUE_DEPTH.set(batcher.queue_depth)

        return GenerateResponse(
            texts=texts,
            tokens=rows.tolist() if req.return_tokens else None,
            nfe=out.nfe,
            model_version=loaded.version,
            latency_ms=(time.perf_counter() - start) * 1000,
            batch_size=int(out.tokens.shape[0]),
        )

    @app.post("/generate/stream", tags=["inference"])
    async def generate_stream(
        req: GenerateRequest, loaded: LoadedModel = Depends(require_model)
    ) -> StreamingResponse:
        """Stream the denoising process as server-sent events.

        Each event carries the partially-unmasked text, so a client can watch the
        sequence resolve out of noise. Bypasses the batcher by design -- see the
        module docstring.
        """
        _validate_length(req, loaded)

        async def event_stream() -> AsyncIterator[str]:
            from aether.diffusion.samplers import iter_denoise

            generator = None
            if req.seed is not None:
                generator = torch.Generator(device=loaded.device).manual_seed(req.seed)
            states = iter_denoise(
                loaded.model,
                batch=1,
                length=req.length,
                mask_token_id=loaded.mask_token_id,
                sampler=req.sampler,
                steps=req.steps,
                schedule=cfg.schedule,
                device=loaded.device,
                generator=generator,
                temperature=req.temperature,
            )
            start = time.perf_counter()
            try:
                for state in states:
                    row = state.tokens[0].tolist()
                    # Strip masks so partial text stays renderable client-side.
                    visible = [t for t in row if t != loaded.mask_token_id]
                    payload = {
                        "step": state.step,
                        "total_steps": state.total_steps,
                        "n_masked": state.n_masked,
                        "nfe": state.nfe,
                        "text": loaded.tokenizer.decode(visible),
                    }
                    yield f"event: step\ndata: {json.dumps(payload)}\n\n"
                    # Hand control back so the loop can serve other connections.
                    await asyncio.sleep(0)
                done = {
                    "model_version": loaded.version,
                    "latency_ms": (time.perf_counter() - start) * 1000,
                }
                yield f"event: done\ndata: {json.dumps(done)}\n\n"
                M.REQUESTS.labels(endpoint="generate_stream", status="ok").inc()
            except asyncio.CancelledError:
                # Client hung up mid-stream; not an error worth alerting on.
                logger.info("stream_cancelled")
                raise
            except Exception as exc:  # surface inside the stream
                M.ERRORS.labels(kind="stream").inc()
                M.REQUESTS.labels(endpoint="generate_stream", status="error").inc()
                yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # -- admin ----------------------------------------------------------------
    @app.post("/admin/swap", response_model=ModelInfo, tags=["admin"])
    async def swap(req: SwapRequest) -> ModelInfo:
        """Load a version and make it live, retaining the old one for rollback."""
        try:
            loaded = await asyncio.to_thread(registry.swap, req.version)
        except Exception as exc:  # a bad tag must not kill the server
            M.ERRORS.labels(kind="swap").inc()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"failed to load {req.version!r}: {exc}",
            ) from exc
        M.MODEL_LOADED.set(1)
        return ModelInfo(
            model_version=loaded.version,
            params=int(loaded.metadata.get("params", 0)),
            d_model=loaded.config.d_model,
            n_layers=loaded.config.n_layers,
            n_heads=loaded.config.n_heads,
            max_seq_len=loaded.config.max_seq_len,
            vocab_size=loaded.config.vocab_size,
            step=loaded.metadata.get("step"),
            device=str(loaded.device),
        )

    @app.post("/admin/rollback", response_model=ModelInfo, tags=["admin"])
    async def rollback() -> ModelInfo:
        """Restore the previous version. In memory already, so this cannot fail slowly."""
        try:
            loaded = registry.rollback()
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
        return ModelInfo(
            model_version=loaded.version,
            params=int(loaded.metadata.get("params", 0)),
            d_model=loaded.config.d_model,
            n_layers=loaded.config.n_layers,
            n_heads=loaded.config.n_heads,
            max_seq_len=loaded.config.max_seq_len,
            vocab_size=loaded.config.vocab_size,
            step=loaded.metadata.get("step"),
            device=str(loaded.device),
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> Response:
        M.ERRORS.labels(kind="unhandled").inc()
        logger.warning("unhandled_error", path=request.url.path, error=str(exc))
        return Response(
            content=json.dumps({"detail": "internal server error"}),
            status_code=500,
            media_type="application/json",
        )

    return app
