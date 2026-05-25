"""Lightweight logging wrapper. Uses loguru if available, falls back to stdlib."""
from __future__ import annotations

import logging
import os
import sys

try:
    from loguru import logger as _loguru_logger

    _USE_LOGURU = True
except ImportError:
    _USE_LOGURU = False

_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()


def get_logger(name: str = "beverage_ai"):
    if _USE_LOGURU:
        return _loguru_logger.bind(scope=name)
    lg = logging.getLogger(name)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        lg.addHandler(h)
        lg.setLevel(getattr(logging, _LEVEL, logging.INFO))
    return lg
