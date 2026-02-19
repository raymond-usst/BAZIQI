"""Structured logging for AI modules. Use get_logger(__name__) in each module."""

import logging
import sys


def get_logger(name: str, level: int = None) -> logging.Logger:
    """Return a module-level logger with a single handler and consistent format."""
    log = logging.getLogger(name)
    if level is not None:
        log.setLevel(level)
    if not log.handlers:
        h = logging.StreamHandler(sys.stderr)
        h.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        log.addHandler(h)
    return log
