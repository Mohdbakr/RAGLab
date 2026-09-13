import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.core.config import Settings

settings = Settings()


# core/logging.py
class SingletonLogger:
    _instance = None
    _initialized = False

    @classmethod
    def get_logger(cls):
        if not cls._instance:
            cls._instance = logging.getLogger("rag_app")
            if not cls._initialized:
                cls._setup_logger()
                cls._initialized = True
        return cls._instance

    @classmethod
    def _setup_logger(cls):
        logger = cls._instance
        if logger.handlers:
            return  # Prevent duplicate handlers

        logger.setLevel(settings.LOG_LEVEL)
        formatter = logging.Formatter(settings.LOG_FORMAT)

        # File handler
        Path("logs").mkdir(exist_ok=True)
        file_handler = RotatingFileHandler(
            settings.LOG_FILE, maxBytes=10485760, backupCount=5
        )
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
