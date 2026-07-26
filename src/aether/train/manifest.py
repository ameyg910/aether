"""Run manifest: the provenance record written to every run directory.

Captures exactly what is needed to reproduce or audit a run -- the resolved
config, the git SHA, the dataset hash, the seed, and the environment. Written
once at run start alongside a snapshot of the resolved config.
"""

from __future__ import annotations

import json
import platform
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return out.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None


def dataset_hash(data_dir: str | Path) -> str | None:
    manifest = Path(data_dir) / "manifest.json"
    if not manifest.exists():
        return None
    try:
        obj = json.loads(manifest.read_text())
        value = obj.get("dataset_hash")
        return str(value) if value is not None else None
    except (OSError, json.JSONDecodeError):
        return None


def write_manifest(
    run_dir: Path,
    resolved_config_yaml: str,
    seed: int,
    data_dir: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``manifest.json`` and ``config.yaml`` into ``run_dir``; return the manifest."""
    import torch

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(resolved_config_yaml)
    manifest: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "git_sha": git_sha(),
        "dataset_hash": dataset_hash(data_dir) if data_dir else None,
        "seed": seed,
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        **(extra or {}),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
