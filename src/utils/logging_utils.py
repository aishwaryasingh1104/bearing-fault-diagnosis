"""
src/utils/logging_utils.py
Structured logging configuration for the entire project.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "bfd",
    level: str = "INFO",
    log_dir: Optional[str | Path] = None,
    to_stdout: bool = True,
) -> logging.Logger:
    """Configure and return a named logger.

    Parameters
    ----------
    name:
        Logger name (usually the module name).
    level:
        Logging level string: ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``.
    log_dir:
        Optional directory where a dated log file will be written.
    to_stdout:
        When ``True`` a StreamHandler is added to stdout.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    if to_stdout and not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(log_dir / f"{name}_{timestamp}.log", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# Root project logger – importable directly
logger = setup_logger("bfd")
