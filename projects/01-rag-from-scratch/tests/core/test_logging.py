"""Tests for plainrag.core.logging."""

from __future__ import annotations

from pathlib import Path

import pytest
from loguru import logger as loguru_logger

from plainrag.core.logging import get_logger, print_banner


class TestGetLogger:
    def test_returns_the_loguru_logger(self) -> None:
        assert get_logger() is loguru_logger

    def test_writes_to_the_given_log_file_when_one_is_configured(self, tmp_path: Path) -> None:
        log_file = tmp_path / "plainrag.log"
        logger = get_logger(log_file=log_file)

        logger.info("hello")
        logger.remove()  # flush/close the file sink before reading it back

        assert "hello" in log_file.read_text()


class TestPrintBanner:
    def test_prints_the_app_name_and_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        print_banner("0.1.0")

        captured = capsys.readouterr()
        assert "plainrag" in captured.err
        assert "0.1.0" in captured.err
