"""Tests for plainrag.services.rag, written before the implementation.

embed_texts and answer_question are monkeypatched at the plainrag.services.rag
module namespace (where rag.py imports them), so no litellm call happens.
"""

from __future__ import annotations

from typing import Any

import pytest

from plainrag.core.exceptions import EmptyIndexError
from plainrag.domain.models import AnswerResult, DocumentChunk, ScoredChunk
from plainrag.services.index import CosineSimilarityIndex
from plainrag.services.rag import ask, ingest_text, retrieve


class TestIngestText:
    @pytest.mark.asyncio
    async def test_empty_text_adds_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_embed_texts(texts: list[str], **kwargs: Any) -> list[list[float]]:
            raise AssertionError("should not embed when there's nothing to chunk")

        monkeypatch.setattr("plainrag.services.rag.embed_texts", fake_embed_texts)
        index = CosineSimilarityIndex()

        added = await ingest_text(index, "", source="empty.txt")

        assert added == 0
        assert len(index) == 0

    @pytest.mark.asyncio
    async def test_chunks_are_embedded_and_added_with_their_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        async def fake_embed_texts(texts: list[str], **kwargs: Any) -> list[list[float]]:
            captured["texts"] = texts
            captured["kwargs"] = kwargs
            return [[float(i), 0.0] for i in range(len(texts))]

        monkeypatch.setattr("plainrag.services.rag.embed_texts", fake_embed_texts)
        index = CosineSimilarityIndex()

        added = await ingest_text(
            index, "one two three four five six", source="doc.txt", chunk_size=3, overlap=0
        )

        assert added == len(index) == 2
        assert captured["texts"] == ["one two three", "four five six"]
        assert all(c.source == "doc.txt" for c in index._chunks)  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_on_step_is_called_once_per_pipeline_stage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_embed_texts(texts: list[str], **kwargs: Any) -> list[list[float]]:
            return [[float(i), 0.0] for i in range(len(texts))]

        monkeypatch.setattr("plainrag.services.rag.embed_texts", fake_embed_texts)
        index = CosineSimilarityIndex()
        steps: list[str] = []

        await ingest_text(
            index,
            "one two three four",
            source="doc.txt",
            chunk_size=2,
            overlap=0,
            on_step=steps.append,
        )

        assert len(steps) == 3
        assert "chunk" in steps[0]
        assert "embed" in steps[1]
        assert "index" in steps[2]

    @pytest.mark.asyncio
    async def test_on_step_defaults_to_a_no_op(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_embed_texts(texts: list[str], **kwargs: Any) -> list[list[float]]:
            return [[0.0, 0.0] for _ in texts]

        monkeypatch.setattr("plainrag.services.rag.embed_texts", fake_embed_texts)
        index = CosineSimilarityIndex()

        added = await ingest_text(index, "one two", source="doc.txt", chunk_size=2, overlap=0)

        assert added == 1


class TestRetrieve:
    @pytest.mark.asyncio
    async def test_embeds_the_query_and_searches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        embed_calls: list[list[str]] = []

        async def fake_embed_texts(texts: list[str], **kwargs: Any) -> list[list[float]]:
            embed_calls.append(texts)
            return [[1.0, 0.0]]

        monkeypatch.setattr("plainrag.services.rag.embed_texts", fake_embed_texts)
        index = CosineSimilarityIndex()
        index.add([DocumentChunk(text="seed fact", source="doc.txt", embedding=[1.0, 0.0])])

        results = await retrieve(index, "What is the seed fact?", k=2)

        assert [s.chunk.text for s in results] == ["seed fact"]
        assert embed_calls == [["What is the seed fact?"]]

    @pytest.mark.asyncio
    async def test_raises_empty_index_error_without_calling_any_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def unexpected_embed_texts(texts: list[str], **kwargs: Any) -> list[list[float]]:
            raise AssertionError("should not embed against an empty index")

        monkeypatch.setattr("plainrag.services.rag.embed_texts", unexpected_embed_texts)

        with pytest.raises(EmptyIndexError):
            await retrieve(CosineSimilarityIndex(), "Anything?")

    @pytest.mark.asyncio
    async def test_on_step_reports_embed_and_retrieve_stages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_embed_texts(texts: list[str], **kwargs: Any) -> list[list[float]]:
            return [[1.0, 0.0]]

        monkeypatch.setattr("plainrag.services.rag.embed_texts", fake_embed_texts)
        index = CosineSimilarityIndex()
        index.add([DocumentChunk(text="seed", source="doc.txt", embedding=[1.0, 0.0])])
        steps: list[str] = []

        await retrieve(index, "anything?", on_step=steps.append)

        assert len(steps) == 2
        assert "embed" in steps[0]
        assert "retriev" in steps[1]


class TestAsk:
    @pytest.mark.asyncio
    async def test_embeds_the_question_searches_and_generates_an_answer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        embed_calls: list[list[str]] = []

        async def fake_embed_texts(texts: list[str], **kwargs: Any) -> list[list[float]]:
            embed_calls.append(texts)
            return [[1.0, 0.0]]

        answer_calls: dict[str, Any] = {}

        async def fake_answer_question(
            question: str, chunks: list[ScoredChunk], **kwargs: Any
        ) -> str:
            answer_calls["question"] = question
            answer_calls["chunks"] = chunks
            return "the answer"

        monkeypatch.setattr("plainrag.services.rag.embed_texts", fake_embed_texts)
        monkeypatch.setattr("plainrag.services.rag.answer_question", fake_answer_question)

        index = CosineSimilarityIndex()
        # Seed the index directly with a known vector, bypassing ingest.
        index.add([DocumentChunk(text="seed fact", source="doc.txt", embedding=[1.0, 0.0])])

        result = await ask(index, "What is the seed fact?", k=2)

        assert isinstance(result, AnswerResult)
        assert result.answer == "the answer"
        assert [s.chunk.text for s in result.sources] == ["seed fact"]
        assert embed_calls == [["What is the seed fact?"]]
        assert answer_calls["question"] == "What is the seed fact?"

    @pytest.mark.asyncio
    async def test_raises_empty_index_error_without_calling_any_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def unexpected_embed_texts(texts: list[str], **kwargs: Any) -> list[list[float]]:
            raise AssertionError("should not embed against an empty index")

        async def unexpected_answer_question(
            question: str, chunks: list[ScoredChunk], **kwargs: Any
        ) -> str:
            raise AssertionError("should not generate against an empty index")

        monkeypatch.setattr("plainrag.services.rag.embed_texts", unexpected_embed_texts)
        monkeypatch.setattr("plainrag.services.rag.answer_question", unexpected_answer_question)

        with pytest.raises(EmptyIndexError):
            await ask(CosineSimilarityIndex(), "Anything?")

    @pytest.mark.asyncio
    async def test_on_step_reports_all_three_stages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_embed_texts(texts: list[str], **kwargs: Any) -> list[list[float]]:
            return [[1.0, 0.0]]

        async def fake_answer_question(
            question: str, chunks: list[ScoredChunk], **kwargs: Any
        ) -> str:
            return "the answer"

        monkeypatch.setattr("plainrag.services.rag.embed_texts", fake_embed_texts)
        monkeypatch.setattr("plainrag.services.rag.answer_question", fake_answer_question)
        index = CosineSimilarityIndex()
        index.add([DocumentChunk(text="seed", source="doc.txt", embedding=[1.0, 0.0])])
        steps: list[str] = []

        await ask(index, "anything?", on_step=steps.append)

        assert len(steps) == 3
        assert "embed" in steps[0]
        assert "retriev" in steps[1]
        assert "generat" in steps[2]
