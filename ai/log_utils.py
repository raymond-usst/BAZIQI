"""Structured logging for AI modules. Use get_logger(__name__) in each module."""

import logging
import logging.handlers
import os
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


def setup_file_logging(log_dir: str, filename: str = "training.log",
                       max_bytes: int = 50 * 1024 * 1024, backup_count: int = 5):
    """Add a RotatingFileHandler to the root 'ai' logger.
    
    All child loggers (ai.train_async, ai.self_play, etc.) inherit this handler.
    Logs are written to log_dir/filename with rotation at max_bytes.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)
    
    root_ai = logging.getLogger("ai")
    root_ai.setLevel(logging.DEBUG)
    
    # Avoid duplicate file handlers on re-init
    for h in root_ai.handlers[:]:
        if isinstance(h, logging.handlers.RotatingFileHandler):
            root_ai.removeHandler(h)
    
    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_ai.addHandler(fh)
    root_ai.info("=== File logging initialized: %s ===", log_path)
    return log_path

