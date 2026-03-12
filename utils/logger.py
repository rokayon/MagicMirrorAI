"""
utils/logger.py
===============
Centralized logger for the Magic Mirror project.
All modules must use get_logger() — never use print() in production.
"""

import logging
import sys
from pathlib import Path

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
DATE_FORMAT = "%H:%M:%S"
LOG_DIR = Path("logs")


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger with console + file handlers.

    Args:
        name:  Module name (e.g. 'vision.face_detection')
        level: Logging level (default INFO)

    Returns:
        Configured Logger instance
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(level)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(console)

    # File handler
    LOG_DIR.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(LOG_DIR / "mirror.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(file_handler)

    return logger
