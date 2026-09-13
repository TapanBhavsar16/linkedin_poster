import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler

LOGGER_NAME = "linkedin_poster"


class LevelFormatter(logging.Formatter):
    """Different format for each log level."""

    FORMATS = {
        logging.DEBUG: (
            "%(asctime)s | DEBUG | %(filename)s:%(lineno)d | "
            "%(funcName)s() | %(message)s"
        ),
        logging.INFO: (
            "%(asctime)s | INFO | %(message)s"
        ),
        logging.WARNING: (
            "%(asctime)s | WARNING | %(filename)s | %(message)s"
        ),
        logging.ERROR: (
            "%(asctime)s | ERROR | %(filename)s:%(lineno)d | "
            "%(funcName)s() | %(message)s"
        ),
        logging.CRITICAL: (
            "%(asctime)s | CRITICAL | %(filename)s:%(lineno)d | "
            "%(funcName)s() | %(message)s"
        ),
    }

    def format(self, record: logging.LogRecord) -> str:
        formatter = logging.Formatter(
            self.FORMATS.get(record.levelno),
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        return formatter.format(record)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure console and persistent application logging."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    formatter = LevelFormatter()
    if not any(getattr(handler, "_linkedin_console", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler._linkedin_console = True
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "linkedin_poster.log"
    if not any(getattr(handler, "_linkedin_file", False) for handler in logger.handlers):
        handler = RotatingFileHandler(
            log_path,
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        handler._linkedin_file = True
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
