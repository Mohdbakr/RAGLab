"""Tests for raglab_common.logging_setup, written before the implementation."""

from __future__ import annotations

from pathlib import Path

from loguru import logger as loguru_logger

from raglab_common.logging_setup import configure_logging


def test_configure_logging_writes_to_a_project_scoped_log_file(
    tmp_path: Path,
) -> None:
    log_dir = tmp_path / "logs"

    log = configure_logging(project_id="01-rag-from-scratch", log_dir=log_dir)
    log.info("hello from the test suite")
    loguru_logger.complete()  # flush async sinks before reading the file back

    log_file = log_dir / "01-rag-from-scratch.log"
    assert log_file.exists()
    assert "hello from the test suite" in log_file.read_text()


def test_configure_logging_binds_project_id_into_records(tmp_path: Path) -> None:
    log_dir = tmp_path / "logs"

    log = configure_logging(project_id="03-production-rag-reference", log_dir=log_dir)
    log.info("bound context check")
    loguru_logger.complete()

    log_file = log_dir / "03-production-rag-reference.log"
    assert "03-production-rag-reference" in log_file.read_text()
