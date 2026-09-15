"""Console + optional file logging, and a one-time startup banner.

Deliberately not importing a shared `raglab_common`: this project doesn't
depend on anything else in the repo.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger as _logger
from rich.console import Console
from rich.panel import Panel

if TYPE_CHECKING:
    from loguru import Logger

_CONSOLE_FORMAT = "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"
_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"


def get_logger(*, level: str = "INFO", log_file: Path | None = None) -> Logger:
    """Configure console (and optional file) sinks and return the logger.

    Reconfigures on every call rather than guarding with a "first call
    wins" flag — cheap, and it keeps this testable without global state
    leaking between tests. Safe in practice since it's normally called
    once per process, at CLI startup.

    Args:
        level: Minimum level for both sinks.
        log_file: If given, also log to this file (rotated at 10 MB,
            kept for 7 days). If omitted, console-only.

    Returns:
        The process-wide loguru logger.
    """
    _logger.remove()
    _logger.add(sys.stderr, level=level, format=_CONSOLE_FORMAT)
    if log_file is not None:
        _logger.add(
            log_file, level=level, format=_FILE_FORMAT, rotation="10 MB", retention="7 days"
        )
    return _logger


def print_banner(version: str) -> None:
    """Print a one-time startup banner to the console (not logged).

    Args:
        version: The application version to display.
    """
    console = Console(stderr=True)
    console.print(
        Panel.fit(
            f"[bold cyan]plainrag[/] [dim]v{version}[/]\n"
            "[dim]RAG from scratch — no framework, no vector-DB service[/]",
            border_style="cyan",
        )
    )
