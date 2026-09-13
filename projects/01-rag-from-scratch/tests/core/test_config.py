"""Tests for plainrag.core.config."""

from __future__ import annotations

from pathlib import Path

import pytest

from plainrag.core.config import Settings, get_settings


class TestSettings:
    def test_defaults(self) -> None:
        settings = Settings(_env_file=None)

        assert settings.embedding_model == "text-embedding-3-small"
        assert settings.chat_model == "gpt-4o-mini"
        assert settings.chunk_size == 200
        assert settings.chunk_overlap == 20
        assert settings.retrieval_k == 4
        assert settings.index_path == Path(".plainrag_index.json")
        assert settings.log_level == "INFO"
        assert settings.log_file is None

    def test_reads_overrides_from_prefixed_environment_variables(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PLAINRAG_CHAT_MODEL", "gpt-4o")
        monkeypatch.setenv("PLAINRAG_CHUNK_SIZE", "500")

        settings = Settings(_env_file=None)

        assert settings.chat_model == "gpt-4o"
        assert settings.chunk_size == 500


class TestGetSettings:
    def test_returns_the_same_cached_instance_on_repeated_calls(self) -> None:
        get_settings.cache_clear()
        try:
            assert get_settings() is get_settings()
        finally:
            get_settings.cache_clear()
