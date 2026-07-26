"""Tracking backends."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aether.train.tracking import JSONLTracker, NoOpTracker, build_tracker


def test_jsonl_tracker_writes_records(tmp_path: Path) -> None:
    tr = JSONLTracker(tmp_path)
    tr.log({"loss": 1.5, "lr": 0.001}, step=1)
    tr.log_text("sample", "hello world", step=1)
    lines = [json.loads(x) for x in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert lines[0]["loss"] == 1.5
    assert lines[1]["sample"] == "hello world"


def test_build_tracker_selects_backend(tmp_path: Path) -> None:
    assert isinstance(build_tracker("none", tmp_path), NoOpTracker)
    assert isinstance(build_tracker("jsonl", tmp_path), JSONLTracker)
    with pytest.raises(ValueError, match="Unknown tracking backend"):
        build_tracker("bogus", tmp_path)
