import logging

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
    """Configure and return the application logger without touching root logging."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(LevelFormatter())
        logger.addHandler(handler)

    return logger
