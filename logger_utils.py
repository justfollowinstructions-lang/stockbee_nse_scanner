"""
NSE Darvas Box Scanner - Logging Utilities
Provides named loggers writing to both console and dedicated log files.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from config import LOG_FORMAT, LOG_DATE, LOG_FILES

_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    """Return a cached, fully configured logger for *name*."""
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    fmt = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Named file handler
    log_key = name if name in LOG_FILES else "scanner"
    fh = RotatingFileHandler(
        LOG_FILES[log_key],
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Always also write to error.log for WARNING+
    eh = RotatingFileHandler(
        LOG_FILES["error"],
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    eh.setLevel(logging.WARNING)
    eh.setFormatter(fmt)
    logger.addHandler(eh)

    _loggers[name] = logger
    return logger
