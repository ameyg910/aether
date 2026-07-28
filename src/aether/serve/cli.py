"""Launch the inference server: ``aether-serve [hydra overrides]``.

    aether-serve serve.model_version=runs/my-run
    aether-serve serve.model_version=hf:owner/repo@v1 serve.port=8080
    aether-serve serve.max_batch_size=64 serve.max_wait_ms=10

The server starts even when the model fails to load: ``/health`` answers, ``/ready``
reports 503, and an operator can push a working checkpoint with ``/admin/swap``.
Crash-looping on a bad version tag would be strictly worse.
"""

from __future__ import annotations

import sys

import uvicorn

from aether.config import load_config
from aether.log import configure_logging, get_logger
from aether.serve.app import ServerSettings, create_app

logger = get_logger("serve.cli")


def main() -> None:
    cfg = load_config(sys.argv[1:])
    configure_logging()

    settings = ServerSettings(
        model_version=cfg.serve.model_version,
        tokenizer=cfg.data.tokenizer,
        device=cfg.serve.device,
        max_batch_size=cfg.serve.max_batch_size,
        max_wait_ms=cfg.serve.max_wait_ms,
        schedule=cfg.diffusion.schedule.kind,
        cache_dir=cfg.serve.cache_dir,
    )
    logger.info(
        "server_starting",
        host=cfg.serve.host,
        port=cfg.serve.port,
        model_version=cfg.serve.model_version,
        max_batch_size=cfg.serve.max_batch_size,
        max_wait_ms=cfg.serve.max_wait_ms,
    )
    uvicorn.run(
        create_app(settings),
        host=cfg.serve.host,
        port=cfg.serve.port,
        log_level=cfg.serve.log_level,
        # Graceful shutdown: let in-flight requests finish before the loop closes.
        timeout_graceful_shutdown=30,
    )


if __name__ == "__main__":
    main()
