"""Async dynamic batching.

A GPU is enormously more efficient on a batch of 32 than on 32 batches of 1: the
same weight matrices are read from HBM either way, so single-request inference is
memory-bandwidth bound and leaves most of the arithmetic units idle. Dynamic
batching exploits that by holding arriving requests for a few milliseconds and
running whatever accumulated as one batch.

The tradeoff is explicit and worth being able to state: **every request pays up to
``max_wait_ms`` of added latency so that all of them share one forward pass.**
Under low load that wait is pure cost -- a lone request waits for companions that
never arrive. Under concurrency it is heavily net-positive, because the batch runs
in roughly the time one request would have taken alone. Hence two knobs:
``max_batch_size`` caps memory, ``max_wait_ms`` caps the latency penalty.

**Batching requires compatible work.** Generation parameters that change the
compute graph -- sampler, step count, sequence length, temperature -- cannot be
merged, so requests are grouped by those into a :class:`BatchKey`. Mismatched
requests are deferred to the next round rather than dropped or forced together.

The model call itself is synchronous CPU/GPU work, so it runs in a worker thread
via ``asyncio.to_thread``; running it inline would block the event loop and stall
every other connection, which defeats the point of an async server.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, NamedTuple

from aether.log import get_logger

logger = get_logger("batcher")

# ``asyncio.TimeoutError`` only became an alias of the builtin in Python 3.11.
# On 3.10 it is ``concurrent.futures.TimeoutError``, which is NOT a subclass of
# the builtin, so an `except TimeoutError` there silently misses it. Catch both
# via a constant -- writing them inline lets a linter collapse them again.
_TIMEOUT_ERRORS: tuple[type[BaseException], ...] = (TimeoutError, asyncio.TimeoutError)


class BatchKey(NamedTuple):
    """Generation parameters that must match for requests to share a batch."""

    sampler: str
    steps: int
    length: int
    temperature: float


@dataclass
class BatchItem:
    """One queued request awaiting execution."""

    key: BatchKey
    n_samples: int
    future: asyncio.Future[Any]
    enqueued_at: float = field(default_factory=time.perf_counter)


@dataclass
class BatchStats:
    """Observability counters, surfaced through Prometheus."""

    batches_run: int = 0
    items_batched: int = 0
    samples_generated: int = 0
    last_batch_size: int = 0
    max_batch_size_seen: int = 0

    @property
    def mean_items_per_batch(self) -> float:
        return self.items_batched / self.batches_run if self.batches_run else 0.0


class DynamicBatcher:
    """Accumulates concurrent requests and executes them as single batches."""

    def __init__(
        self,
        run_batch: Callable[[BatchKey, int], Any],
        max_batch_size: int = 32,
        max_wait_ms: float = 20.0,
    ) -> None:
        self.run_batch = run_batch
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_ms / 1000.0
        self.stats = BatchStats()
        self._queue: asyncio.Queue[BatchItem] = asyncio.Queue()
        self._deferred: deque[BatchItem] = deque()
        self._task: asyncio.Task[None] | None = None
        self._running = False

    # -- lifecycle -----------------------------------------------------------
    async def start(self) -> None:
        if self._task is None:
            self._running = True
            self._task = asyncio.create_task(self._loop())
            logger.info(
                "batcher_started",
                max_batch_size=self.max_batch_size,
                max_wait_ms=self.max_wait_s * 1000,
            )

    async def stop(self) -> None:
        """Stop accepting work and fail anything still queued.

        Called on shutdown: a pending future that is never resolved would hang the
        client until it times out, so queued items are failed explicitly.
        """
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        for item in list(self._deferred) + self._drain_queue():
            if not item.future.done():
                item.future.set_exception(RuntimeError("server shutting down"))
        self._deferred.clear()
        logger.info("batcher_stopped", batches_run=self.stats.batches_run)

    def _drain_queue(self) -> list[BatchItem]:
        drained = []
        while True:
            try:
                drained.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                return drained

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize() + len(self._deferred)

    # -- submission ----------------------------------------------------------
    async def submit(self, key: BatchKey, n_samples: int) -> Any:
        """Enqueue a request and await its result."""
        if not self._running:
            raise RuntimeError("batcher is not running")
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        await self._queue.put(BatchItem(key=key, n_samples=n_samples, future=future))
        return await future

    # -- execution -----------------------------------------------------------
    async def _next_item(self) -> BatchItem:
        if self._deferred:
            return self._deferred.popleft()
        return await self._queue.get()

    async def _collect(self) -> list[BatchItem]:
        """Gather a compatible batch, waiting at most ``max_wait_ms`` for company."""
        first = await self._next_item()
        batch = [first]
        total = first.n_samples
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.max_wait_s

        try:
            yield_batch = await self._gather(first, batch, total, deadline, loop)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Whatever failed while looking for companions, ``batch`` already holds
            # real requests whose futures nobody else will resolve. Run them rather
            # than dropping them -- a lost future is an infinite client hang.
            logger.warning("collect_degraded", error=repr(exc), held=len(batch))
            return batch
        return yield_batch

    async def _gather(
        self,
        first: BatchItem,
        batch: list[BatchItem],
        total: int,
        deadline: float,
        loop: asyncio.AbstractEventLoop,
    ) -> list[BatchItem]:
        while total < self.max_batch_size:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            if self._deferred and self._deferred[0].key == first.key:
                item = self._deferred.popleft()
            else:
                try:
                    item = await asyncio.wait_for(self._queue.get(), remaining)
                except _TIMEOUT_ERRORS:
                    break
            if item.key != first.key or total + item.n_samples > self.max_batch_size:
                # Incompatible or would overflow: hold it for the next round.
                self._deferred.append(item)
                continue
            batch.append(item)
            total += item.n_samples
        return batch

    async def _loop(self) -> None:
        while self._running:
            try:
                batch = await self._collect()
            except asyncio.CancelledError:
                raise
            if not batch:
                continue

            key = batch[0].key
            total = sum(item.n_samples for item in batch)
            self.stats.batches_run += 1
            self.stats.items_batched += len(batch)
            self.stats.samples_generated += total
            self.stats.last_batch_size = total
            self.stats.max_batch_size_seen = max(self.stats.max_batch_size_seen, total)

            try:
                # Offload the blocking model call so the event loop stays responsive.
                result = await asyncio.to_thread(self.run_batch, key, total)
            except asyncio.CancelledError:
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(RuntimeError("server shutting down"))
                raise
            except Exception as exc:  # propagate to every caller
                logger.warning("batch_failed", error=str(exc), batch_size=total)
                for item in batch:
                    if not item.future.done():
                        item.future.set_exception(exc)
                continue

            # Split the batch result back out to the individual callers.
            offset = 0
            for item in batch:
                if not item.future.done():
                    item.future.set_result((result, offset, item.n_samples))
                offset += item.n_samples
