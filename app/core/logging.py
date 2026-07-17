"""Minimal structured-ish logging setup shared across the app."""
from __future__ import annotations

import logging
import os

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=os.environ.get("LOG_LEVEL", "INFO"),
            format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        )
        _CONFIGURED = True
    return logging.getLogger(name)
