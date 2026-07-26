"""Structured logging setup (structlog).

Call :func:`configure_logging` once at process start. Human-readable console
output by default; set ``json_logs=True`` for machine-parseable JSON lines, which
is what you want when shipping logs to a collector in later weeks.
"""

from __future__ import annotations

import logging

import structlog


def configure_logging(json_logs: bool = False, level: str = "INFO") -> None:
    logging.basicConfig(format="%(message)s", level=getattr(logging, level.upper()))
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    logger: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    return logger
