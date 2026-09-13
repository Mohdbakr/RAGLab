"""Tests for plainrag.cli, written before the implementation.

The underlying rag functions are monkeypatched at the plainrag.cli
module namespace (where cli.py imports them), so no litellm call happens
and these stay fast/offline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from plainrag.domain.models import AnswerResult, DocumentChunk, ScoredChunk
from plainrag.services.index import CosineSimilarityIndex

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run every test from an empty temp directory (default index path is relative)."""
    monkeypatch.chdir(tmp_path)


class TestIngestCommand:
    def test_reads_the_file_and_reports_how_many_chunks_were_added(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from plainrag import cli

        source_file = tmp_path / "doc.txt"
        source_file.write_text("some content")
        captured: dict[str, Any] = {}

        async def fake_ingest_text(
            index: CosineSimilarityIndex, text: str, source: str, **kwargs: Any
        ) -> int:
            captured["text"] = text
            captured["source"] = source
            captured["kwargs"] = kwargs
            return 3

        monkeypatch.setattr(cli, "ingest_text", fake_ingest_text)

        result = runner.invoke(cli.app, ["ingest", str(source_file)])

        assert result.exit_code == 0
        assert captured["text"] == "some content"
        assert captured["source"] == str(source_file)
        assert "3" in result.output

    def test_saves_the_index_after_ingesting(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from plainrag import cli

        source_file = tmp_path / "doc.txt"
        source_file.write_text("some content")
        index_path = tmp_path / "custom_index.json"

        async def fake_ingest_text(
            index: CosineSimilarityIndex, text: str, source: str, **kwargs: Any
        ) -> int:
            index.add([DocumentChunk(text=text, source=source, embedding=[1.0, 0.0])])
            return 1

        monkeypatch.setattr(cli, "ingest_text", fake_ingest_text)

        result = runner.invoke(
            cli.app, ["ingest", str(source_file), "--index-path", str(index_path)]
        )

        assert result.exit_code == 0
        assert index_path.exists()
        assert len(CosineSimilarityIndex.load(index_path)) == 1


class TestAskCommand:
    def test_exits_with_an_error_when_the_index_is_empty(self, tmp_path: Path) -> None:
        from plainrag import cli

        result = runner.invoke(
            cli.app, ["ask", "anything?", "--index-path", str(tmp_path / "missing.json")]
        )

        assert result.exit_code == 1

    def test_prints_the_answer_and_its_sources(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from plainrag import cli

        index_path = tmp_path / "index.json"
        index = CosineSimilarityIndex()
        index.add([DocumentChunk(text="seed", source="doc.txt", embedding=[1.0, 0.0])])
        index.save(index_path)

        async def fake_ask(
            index: CosineSimilarityIndex, question: str, **kwargs: Any
        ) -> AnswerResult:
            return AnswerResult(
                answer="42",
                sources=[
                    ScoredChunk(chunk=DocumentChunk("seed", "doc.txt", [1.0, 0.0]), score=0.9)
                ],
            )

        monkeypatch.setattr(cli, "ask_question", fake_ask)

        result = runner.invoke(cli.app, ["ask", "what?", "--index-path", str(index_path)])

        assert result.exit_code == 0
        assert "42" in result.output
        assert "doc.txt" in result.output


class TestBanner:
    def test_prints_on_a_real_command(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from plainrag import cli

        source_file = tmp_path / "doc.txt"
        source_file.write_text("content")

        async def fake_ingest_text(
            index: CosineSimilarityIndex, text: str, source: str, **kwargs: Any
        ) -> int:
            return 0

        monkeypatch.setattr(cli, "ingest_text", fake_ingest_text)

        result = runner.invoke(cli.app, ["ingest", str(source_file)])

        assert "no vector-DB service" in result.output

    def test_does_not_print_on_help(self) -> None:
        from plainrag import cli

        result = runner.invoke(cli.app, ["--help"])

        assert "no vector-DB service" not in result.output
