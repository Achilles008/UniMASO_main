"""
Structured logging for UniMASO runs.
Each run produces one log file under log/ named with run details.
"""
import logging
import os
import sys
from datetime import datetime

_logger = None
_log_file = None


def setup_logger(dataset: str, graph: str, mode: str, log_dir: str = "log") -> logging.Logger:
    """Initialize logger for a single run. Call once at startup.

    File name: log/{dataset}_{graph}_{mode}_{YYYYMMDD_HHMMSS}.log
    """
    global _logger, _log_file

    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_file = os.path.join(log_dir, f"{dataset}_{graph}_{mode}_{ts}.log")

    _logger = logging.getLogger("UniMASO")
    _logger.setLevel(logging.DEBUG)
    _logger.handlers.clear()

    # File handler — full detail
    fh = logging.FileHandler(_log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    ))

    # Console handler — info and above only
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(levelname)-7s | %(message)s"))

    _logger.addHandler(fh)
    _logger.addHandler(ch)

    _logger.info(f"Log file: {_log_file}")
    return _logger


def get_logger() -> logging.Logger:
    """Get the current run's logger."""
    if _logger is None:
        return logging.getLogger("UniMASO")  # fallback
    return _logger
