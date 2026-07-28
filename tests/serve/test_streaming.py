"""SSE denoising stream and end-to-end batching under concurrency."""

from __future__ import annotations

import asyncio
import json
from itertools import pairwise
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from aether.serve.app import ServerSettings, create_app


def _parse_sse(raw: str) -> list[tuple[str, dict]]:  # type: ignore[type-arg]
    events = []
    for block in raw.strip().split("\n\n"):
        name, data = None, None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if name and data is not None:
            events.append((name, data))
    return events


def test_stream_emits_steps_then_done(client: TestClient) -> None:
    with client.stream("POST", "/generate/stream", json={"length": 16, "steps": 5}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse("".join(r.iter_text()))

    names = [name for name, _ in events]
    assert names[-1] == "done"
    assert names.count("step") >= 5


def test_stream_progressively_unmasks(client: TestClient) -> None:
    with client.stream("POST", "/generate/stream", json={"length": 24, "steps": 8}) as r:
        events = _parse_sse("".join(r.iter_text()))

    masked = [d["n_masked"] for name, d in events if name == "step"]
    # Monotonically non-increasing, and fully resolved by the end.
    assert all(b <= a for a, b in pairwise(masked))
    assert masked[-1] == 0
    assert masked[0] > 0


def test_stream_reports_nfe_and_progress(client: TestClient) -> None:
    with client.stream("POST", "/generate/stream", json={"length": 16, "steps": 6}) as r:
        events = _parse_sse("".join(r.iter_text()))

    steps = [d for name, d in events if name == "step"]
    assert steps[0]["total_steps"] == 6
    # NFE accumulates as the stream progresses.
    assert steps[-1]["nfe"] >= steps[0]["nfe"] > 0
    assert all(isinstance(d["text"], str) for d in steps)


def test_stream_is_503_without_a_model(unloaded_client: TestClient) -> None:
    r = unloaded_client.post("/generate/stream", json={"length": 8, "steps": 2})
    assert r.status_code == 503


@pytest.mark.asyncio
async def test_concurrent_requests_share_a_batch(checkpoint: Path) -> None:
    """The core batching claim, end to end through HTTP.

    Eight simultaneous requests must be served by one forward pass, not eight.
    """
    app = create_app(
        ServerSettings(
            model_version=f"local:{checkpoint}",
            tokenizer="byte",
            device="cpu",
            max_batch_size=16,
            max_wait_ms=60.0,
        )
    )
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test", timeout=60) as client,
    ):
        body = {"n_samples": 1, "length": 16, "steps": 4}
        responses = await asyncio.gather(*[client.post("/generate", json=body) for _ in range(8)])

    assert all(r.status_code == 200 for r in responses)
    # Every caller sees the merged batch it was served by.
    assert max(r.json()["batch_size"] for r in responses) > 1
    assert app.state.batcher.stats.max_batch_size_seen > 1
    assert app.state.batcher.stats.mean_items_per_batch > 1.0
