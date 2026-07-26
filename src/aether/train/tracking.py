"""Pluggable experiment tracking.

A tiny ``Tracker`` protocol with three backends so the training loop is decoupled
from any one vendor and stays testable offline:

- ``jsonl``  -- append metrics to ``run_dir/metrics.jsonl`` (always available, no
  network); good enough for CI and local runs.
- ``wandb``  -- Weights & Biases; imported lazily so it is an optional dependency.
- ``none``   -- discard everything.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Tracker(Protocol):
    """Minimal metrics/artifact sink."""

    def log(self, metrics: dict[str, float], step: int) -> None: ...
    def log_text(self, key: str, text: str, step: int) -> None: ...
    def finish(self) -> None: ...


class NoOpTracker:
    """Discards all logs."""

    def log(self, metrics: dict[str, float], step: int) -> None:
        return None

    def log_text(self, key: str, text: str, step: int) -> None:
        return None

    def finish(self) -> None:
        return None


class JSONLTracker:
    """Appends newline-delimited JSON records to ``run_dir/metrics.jsonl``."""

    def __init__(self, run_dir: Path) -> None:
        self.path = run_dir / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, record: dict[str, Any]) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._write({"step": step, **metrics})

    def log_text(self, key: str, text: str, step: int) -> None:
        self._write({"step": step, key: text})

    def finish(self) -> None:
        return None


class WandbTracker:
    """Weights & Biases backend (lazy import so wandb stays optional)."""

    def __init__(
        self,
        project: str,
        run_dir: Path,
        entity: str | None = None,
        tags: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        import wandb

        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            entity=entity,
            tags=tags or [],
            dir=str(run_dir),
            config=config or {},
        )

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._wandb.log(metrics, step=step)

    def log_text(self, key: str, text: str, step: int) -> None:
        self._wandb.log({key: self._wandb.Html(f"<pre>{text}</pre>")}, step=step)

    def finish(self) -> None:
        self._run.finish()


def build_tracker(
    backend: str,
    run_dir: Path,
    project: str = "aether",
    entity: str | None = None,
    tags: list[str] | None = None,
    config: dict[str, Any] | None = None,
) -> Tracker:
    if backend == "none":
        return NoOpTracker()
    if backend == "jsonl":
        return JSONLTracker(run_dir)
    if backend == "wandb":
        return WandbTracker(project, run_dir, entity, tags, config)
    raise ValueError(f"Unknown tracking backend {backend!r}")
