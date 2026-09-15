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

from plainrag.core.exceptions import EmbeddingError, EmptyIndexError
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

    def test_prints_progress_from_on_step(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from plainrag import cli

        source_file = tmp_path / "doc.txt"
        source_file.write_text("some content")

        async def fake_ingest_text(
            index: CosineSimilarityIndex, text: str, source: str, **kwargs: Any
        ) -> int:
            on_step = kwargs["on_step"]
            on_step("chunked into 1 piece(s)")
            on_step("embedded 1 chunk(s) via fake-model")
            return 1

        monkeypatch.setattr(cli, "ingest_text", fake_ingest_text)

        result = runner.invoke(cli.app, ["ingest", str(source_file)])

        assert result.exit_code == 0
        assert "chunked into 1 piece(s)" in result.output
        assert "embedded 1 chunk(s) via fake-model" in result.output

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

    def test_exits_with_an_error_when_ingest_text_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from plainrag import cli

        source_file = tmp_path / "doc.txt"
        source_file.write_text("some content")

        async def failing_ingest_text(
            index: CosineSimilarityIndex, text: str, source: str, **kwargs: Any
        ) -> int:
            raise EmbeddingError("embedding provider unavailable")

        monkeypatch.setattr(cli, "ingest_text", failing_ingest_text)

        result = runner.invoke(cli.app, ["ingest", str(source_file)])

        assert result.exit_code == 1

    def test_exits_with_an_error_when_the_input_file_is_missing(self, tmp_path: Path) -> None:
        from plainrag import cli

        result = runner.invoke(cli.app, ["ingest", str(tmp_path / "does-not-exist.txt")])

        assert result.exit_code == 1
        # On this Click/Typer version, a clean `typer.Exit(code=1)` bubbles up as a
        # real SystemExit (Click's own sys.exit(e.exit_code)), while an *uncaught*
        # exception (the old bug) reaches Click's runner as the raw exception itself
        # (e.g. FileNotFoundError) -- `result.exception is None` never holds for a
        # nonzero exit code in this Click version, so we assert on the type instead.
        assert isinstance(result.exception, SystemExit)


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

    def test_prints_progress_from_on_step(
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
            kwargs["on_step"]("retrieved 1 of 1 chunk(s) (scores: 0.90)")
            return AnswerResult(answer="42", sources=[])

        monkeypatch.setattr(cli, "ask_question", fake_ask)

        result = runner.invoke(cli.app, ["ask", "what?", "--index-path", str(index_path)])

        assert result.exit_code == 0
        assert "retrieved 1 of 1 chunk(s) (scores: 0.90)" in result.output


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


class TestDemoCommand:
    def test_ingest_then_search_round_trip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from plainrag import cli

        source_file = tmp_path / "doc.txt"
        source_file.write_text("some content")
        index_path = tmp_path / "index.json"

        async def fake_ingest_text(
            index: CosineSimilarityIndex, text: str, source: str, **kwargs: Any
        ) -> int:
            index.add([DocumentChunk(text=text, source=source, embedding=[1.0, 0.0])])
            return 1

        async def fake_retrieve(
            index: CosineSimilarityIndex, question: str, **kwargs: Any
        ) -> list[ScoredChunk]:
            (chunk,) = index._chunks  # noqa: SLF001
            return [ScoredChunk(chunk=chunk, score=0.87)]

        monkeypatch.setattr(cli, "ingest_text", fake_ingest_text)
        monkeypatch.setattr(cli, "retrieve_chunks", fake_retrieve)

        result = runner.invoke(
            cli.app,
            ["demo", "--index-path", str(index_path)],
            input=f"ingest {source_file}\nsearch what's in here?\nquit\n",
        )

        assert result.exit_code == 0
        assert "Added 1 chunk(s)" in result.output
        assert "(0.870)" in result.output
        assert str(source_file) in result.output
        assert index_path.exists()
        assert len(CosineSimilarityIndex.load(index_path)) == 1

    def test_reports_loaded_chunk_count_at_startup(self, tmp_path: Path) -> None:
        from plainrag import cli

        index_path = tmp_path / "index.json"
        index = CosineSimilarityIndex()
        index.add([DocumentChunk(text="a", source="doc.txt", embedding=[1.0, 0.0])])
        index.save(index_path)

        result = runner.invoke(cli.app, ["demo", "--index-path", str(index_path)], input="quit\n")

        assert "1 chunk(s) loaded" in result.output

    def test_ingest_without_a_path_shows_usage(self, tmp_path: Path) -> None:
        from plainrag import cli

        result = runner.invoke(
            cli.app,
            ["demo", "--index-path", str(tmp_path / "index.json")],
            input="ingest\nquit\n",
        )

        assert "Usage: ingest <path>" in result.output

    def test_search_without_a_query_shows_usage(self, tmp_path: Path) -> None:
        from plainrag import cli

        result = runner.invoke(
            cli.app,
            ["demo", "--index-path", str(tmp_path / "index.json")],
            input="search\nquit\n",
        )

        assert "Usage: search <query>" in result.output

    def test_unknown_command_reports_an_error_without_crashing(self, tmp_path: Path) -> None:
        from plainrag import cli

        result = runner.invoke(
            cli.app,
            ["demo", "--index-path", str(tmp_path / "index.json")],
            input="bogus\nquit\n",
        )

        assert result.exit_code == 0
        assert "Unknown command" in result.output

    def test_ingesting_a_missing_file_reports_an_error_and_keeps_looping(
        self, tmp_path: Path
    ) -> None:
        from plainrag import cli

        result = runner.invoke(
            cli.app,
            ["demo", "--index-path", str(tmp_path / "index.json")],
            input=f"ingest {tmp_path / 'does-not-exist.txt'}\nquit\n",
        )

        assert result.exit_code == 0
        assert "Error:" in result.output

    def test_searching_an_empty_index_reports_an_error_and_keeps_looping(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from plainrag import cli

        async def failing_retrieve(
            index: CosineSimilarityIndex, question: str, **kwargs: Any
        ) -> list[ScoredChunk]:
            raise EmptyIndexError("The index is empty — ingest something before searching.")

        monkeypatch.setattr(cli, "retrieve_chunks", failing_retrieve)

        result = runner.invoke(
            cli.app,
            ["demo", "--index-path", str(tmp_path / "index.json")],
            input="search anything\nquit\n",
        )

        assert result.exit_code == 0
        assert "Error:" in result.output

    def test_exits_cleanly_when_input_runs_out_without_quit(self, tmp_path: Path) -> None:
        from plainrag import cli

        result = runner.invoke(
            cli.app, ["demo", "--index-path", str(tmp_path / "index.json")], input=""
        )

        assert result.exit_code == 0
