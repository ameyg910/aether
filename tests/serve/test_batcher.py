"""Dynamic batching behaviour."""

from __future__ import annotations

import asyncio

import pytest

from aether.serve.batcher import BatchKey, DynamicBatcher

KEY = BatchKey(sampler="ancestral", steps=4, length=8, temperature=1.0)
OTHER = BatchKey(sampler="confidence", steps=4, length=8, temperature=1.0)


def _recording_batcher(**kwargs: object) -> tuple[DynamicBatcher, list[int]]:
    seen: list[int] = []

    def run(key: BatchKey, total: int) -> str:
        seen.append(total)
        return f"result-{total}"

    return DynamicBatcher(run_batch=run, **kwargs), seen  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_concurrent_requests_merge_into_one_batch() -> None:
    batcher, seen = _recording_batcher(max_batch_size=8, max_wait_ms=50)
    await batcher.start()
    try:
        results = await asyncio.gather(*[batcher.submit(KEY, 1) for _ in range(5)])
    finally:
        await batcher.stop()
    # One batch of five, not five batches of one.
    assert seen == [5]
    assert len(results) == 5
    assert all(r[0] == "result-5" for r in results)


@pytest.mark.asyncio
async def test_each_caller_gets_its_own_slice() -> None:
    batcher, _ = _recording_batcher(max_batch_size=8, max_wait_ms=50)
    await batcher.start()
    try:
        results = await asyncio.gather(batcher.submit(KEY, 2), batcher.submit(KEY, 3))
    finally:
        await batcher.stop()
    offsets = [(offset, count) for _, offset, count in results]
    # Disjoint, contiguous, and sized as requested.
    assert offsets == [(0, 2), (2, 3)]


@pytest.mark.asyncio
async def test_incompatible_requests_are_not_merged() -> None:
    # Different samplers change the compute graph and cannot share a forward pass.
    batcher, seen = _recording_batcher(max_batch_size=8, max_wait_ms=30)
    await batcher.start()
    try:
        await asyncio.gather(batcher.submit(KEY, 1), batcher.submit(OTHER, 1))
    finally:
        await batcher.stop()
    assert sorted(seen) == [1, 1]


@pytest.mark.asyncio
async def test_batch_never_exceeds_max_size() -> None:
    batcher, seen = _recording_batcher(max_batch_size=4, max_wait_ms=30)
    await batcher.start()
    try:
        await asyncio.gather(*[batcher.submit(KEY, 1) for _ in range(10)])
    finally:
        await batcher.stop()
    assert seen
    assert max(seen) <= 4
    assert sum(seen) == 10


@pytest.mark.asyncio
async def test_model_errors_propagate_to_every_caller() -> None:
    def boom(key: BatchKey, total: int) -> str:
        raise ValueError("model exploded")

    batcher = DynamicBatcher(run_batch=boom, max_batch_size=4, max_wait_ms=20)
    await batcher.start()
    try:
        with pytest.raises(ValueError, match="model exploded"):
            await asyncio.gather(*[batcher.submit(KEY, 1) for _ in range(3)])
    finally:
        await batcher.stop()


@pytest.mark.asyncio
async def test_shutdown_fails_queued_requests_rather_than_hanging() -> None:
    started = asyncio.Event()

    def slow(key: BatchKey, total: int) -> str:
        started.set()
        import time

        time.sleep(0.3)
        return "done"

    batcher = DynamicBatcher(run_batch=slow, max_batch_size=1, max_wait_ms=1)
    await batcher.start()
    first = asyncio.create_task(batcher.submit(KEY, 1))
    await started.wait()
    queued = asyncio.create_task(batcher.submit(KEY, 1))
    await asyncio.sleep(0.02)
    await batcher.stop()
    # A never-resolved future would hang the client until its own timeout.
    with pytest.raises(RuntimeError, match="shutting down"):
        await queued
    first.cancel()


@pytest.mark.asyncio
async def test_submit_before_start_is_rejected() -> None:
    batcher, _ = _recording_batcher(max_batch_size=2, max_wait_ms=5)
    with pytest.raises(RuntimeError, match="not running"):
        await batcher.submit(KEY, 1)


@pytest.mark.asyncio
async def test_stats_track_batching_effectiveness() -> None:
    batcher, _ = _recording_batcher(max_batch_size=8, max_wait_ms=50)
    await batcher.start()
    try:
        await asyncio.gather(*[batcher.submit(KEY, 1) for _ in range(6)])
    finally:
        await batcher.stop()
    assert batcher.stats.batches_run == 1
    assert batcher.stats.items_batched == 6
    assert batcher.stats.max_batch_size_seen == 6
    assert batcher.stats.mean_items_per_batch == 6.0


class _NonBuiltinTimeoutError(Exception):
    """Stands in for Python 3.10's ``asyncio.TimeoutError``.

    Before 3.11 that class is ``concurrent.futures.TimeoutError``, which is *not*
    a subclass of the builtin ``TimeoutError``. An ``except TimeoutError`` there
    misses it, the error escapes the gather phase, and any request already popped
    off the queue loses its future -- an unbounded client hang.
    """


@pytest.mark.asyncio
async def test_uncaught_gather_error_still_serves_the_request() -> None:
    batcher, seen = _recording_batcher(max_batch_size=8, max_wait_ms=20)

    async def boom(*args: object, **kwargs: object) -> list[object]:
        raise _NonBuiltinTimeoutError

    batcher._gather = boom  # type: ignore[method-assign]
    await batcher.start()
    try:
        result = await asyncio.wait_for(batcher.submit(KEY, 1), timeout=5)
    finally:
        await batcher.stop()
    # Degraded to a batch of one, but served -- never left hanging.
    assert result[0] == "result-1"
    assert seen == [1]


@pytest.mark.asyncio
async def test_batcher_survives_repeated_gather_failures() -> None:
    batcher, seen = _recording_batcher(max_batch_size=8, max_wait_ms=20)

    async def boom(*args: object, **kwargs: object) -> list[object]:
        raise _NonBuiltinTimeoutError

    batcher._gather = boom  # type: ignore[method-assign]
    await batcher.start()
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*[batcher.submit(KEY, 1) for _ in range(4)]), timeout=5
        )
    finally:
        await batcher.stop()
    assert len(results) == 4
    assert len(seen) == 4


def test_timeout_errors_constant_covers_both_classes() -> None:
    """The constant exists so a linter cannot collapse the tuple back to one name."""
    from aether.serve.batcher import _TIMEOUT_ERRORS

    assert TimeoutError in _TIMEOUT_ERRORS
    assert asyncio.TimeoutError in _TIMEOUT_ERRORS
