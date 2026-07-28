"""Request and response contracts for the inference API.

Validation happens at the edge: a request with a 10,000-step budget or a negative
length is rejected by pydantic with a 422 before it reaches the model, rather than
becoming an out-of-memory error or a silent clamp deep in the sampler. The bounds
here are the service's actual operating envelope, and they are part of the public
contract published in the OpenAPI schema.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Sampler = Literal["ancestral", "confidence"]


class GenerateRequest(BaseModel):
    """A text generation request."""

    # ``model_`` is a protected prefix in pydantic v2; the field is part of the
    # public API contract, so relax the namespace rather than rename it.
    model_config = ConfigDict(protected_namespaces=())

    n_samples: int = Field(1, ge=1, le=64, description="How many sequences to generate.")
    length: int = Field(64, ge=1, le=2048, description="Tokens per sequence.")
    steps: int = Field(
        64, ge=1, le=1024, description="Denoising steps (NFE): more steps, better quality."
    )
    sampler: Sampler = Field("ancestral", description="Unmasking strategy.")
    temperature: float = Field(1.0, gt=0.0, le=5.0)
    seed: int | None = Field(None, description="Set for reproducible output.")
    return_tokens: bool = Field(False, description="Include raw token ids in the response.")


class GenerateResponse(BaseModel):
    """Generated text plus the provenance and cost of producing it."""

    model_config = ConfigDict(protected_namespaces=())

    texts: list[str]
    tokens: list[list[int]] | None = None
    nfe: int = Field(description="Model forward passes actually spent.")
    model_version: str
    latency_ms: float
    batch_size: int = Field(description="Sequences in the batch this request shared.")


class HealthResponse(BaseModel):
    """Liveness: the process is up and its event loop is responsive."""

    status: Literal["ok"] = "ok"
    uptime_s: float


class ReadyResponse(BaseModel):
    """Readiness: the process can actually serve traffic."""

    model_config = ConfigDict(protected_namespaces=())

    ready: bool
    model_version: str | None = None
    detail: str | None = None


class ModelInfo(BaseModel):
    """Metadata about the currently served checkpoint."""

    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    params: int
    d_model: int
    n_layers: int
    n_heads: int
    max_seq_len: int
    vocab_size: int
    step: int | None = None
    device: str


class SwapRequest(BaseModel):
    """Load a different model version, keeping the old one for rollback."""

    version: str = Field(description="Version tag, e.g. 'hf:owner/repo@v1' or a local path.")


class ErrorResponse(BaseModel):
    """Uniform error body so clients can branch on ``detail`` reliably."""

    detail: str
