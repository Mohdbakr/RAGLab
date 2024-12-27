import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import Settings

settings = Settings()


def setup_logging():
    """Configure logging with file and console handlers."""

    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger("rag_app")
    logger.setLevel(settings.LOG_LEVEL)

    # Create formatters
    formatter = logging.Formatter(settings.LOG_FORMAT)

    # Create file handler
    file_handler = RotatingFileHandler(
        settings.LOG_FILE,
        maxBytes=10485760,  # 10MB
        backupCount=5,
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(settings.LOG_LEVEL)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(settings.LOG_LEVEL)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
