"""One-line loguru configuration shared by every RAGLab project.

Replaces the ad-hoc ``SingletonLogger``/stdlib-``logging`` pattern from the
original scaffold with a single call: ``log = configure_logging(project_id)``
gives back a logger pre-bound with ``project_id`` that writes to both the
console and a per-project rotating log file.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger as _logger

if TYPE_CHECKING:
    from loguru import Logger, Record

_console_sink_added = False
_file_sinks_by_project: dict[str, int] = {}

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{extra[project_id]}</cyan> | {message}"
)
_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {extra[project_id]} | {message}"


def _has_project_id(record: Record) -> bool:
    """Filter predicate: only route records bound with a project_id."""
    return "project_id" in record["extra"]


def configure_logging(project_id: str, log_dir: Path | str = Path("logs")) -> Logger:
    """Configure loguru for one project and return a bound logger.

    Idempotent per ``project_id``: calling this more than once for the same
    project does not add duplicate file sinks. The shared console sink is
    added once per process and shows records from every project that has
    called this function.

    Args:
        project_id: Catalog id of the calling project, used to tag every
            record and to name its log file.
        log_dir: Directory the rotating log file is written into. Created
            if missing.

    Returns:
        A loguru logger pre-bound with ``project_id``, ready for
        ``log.info(...)``-style calls.
    """
    global _console_sink_added

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not _console_sink_added:
        _logger.remove()
        _logger.add(
            sys.stderr,
            level="INFO",
            format=_CONSOLE_FORMAT,
            filter=_has_project_id,
        )
        _console_sink_added = True

    if project_id not in _file_sinks_by_project:
        log_file = log_dir / f"{project_id}.log"

        def _matches_this_project(record: Record, pid: str = project_id) -> bool:
            return record["extra"].get("project_id") == pid

        sink_id = _logger.add(
            log_file,
            level="DEBUG",
            rotation="10 MB",
            retention=5,
            format=_FILE_FORMAT,
            filter=_matches_this_project,
        )
        _file_sinks_by_project[project_id] = sink_id

    return _logger.bind(project_id=project_id)
