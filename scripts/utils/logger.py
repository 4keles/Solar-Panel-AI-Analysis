"""Structured logging utility."""

from __future__ import annotations

import logging
from typing import Any

try:
    import structlog
except Exception:  # pragma: no cover
    structlog = None  # type: ignore[assignment]


def get_logger(name: str) -> Any:
    """Return a structured logger compatible with contract API."""
    if structlog is not None:
        structlog.configure_once(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
            cache_logger_on_first_use=True,
        )
        return structlog.get_logger(name)

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt='{"timestamp":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
