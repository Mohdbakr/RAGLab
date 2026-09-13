"""A tiny, self-contained loguru setup — no shared package dependency.

Deliberately not importing a shared `raglab_common`: this project doesn't
depend on anything else in the repo.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from loguru import logger as _logger

if TYPE_CHECKING:
    from loguru import Logger

_configured = False


def get_logger() -> Logger:
    """Configure (once) and return the loguru logger for this CLI.

    Returns:
        The process-wide loguru logger, writing to stderr.
    """
    global _configured
    if not _configured:
        _logger.remove()
        _logger.add(sys.stderr, level="INFO", format="<level>{level: <8}</level> | {message}")
        _configured = True
    return _logger
