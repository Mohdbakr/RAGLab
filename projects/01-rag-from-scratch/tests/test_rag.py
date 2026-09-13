"""Tests for ragscratch.rag, written before the implementation.

embed_texts and answer_question are monkeypatched at the ragscratch.rag
module namespace (where rag.py imports them), so no litellm call happens.
"""

from __future__ import annotations

from typing import Any

import pytest

from ragscratch.index import CosineSimilarityIndex, DocumentChunk, ScoredChunk
from ragscratch.rag import AnswerResult, ask, ingest_text


class TestIngestText:
    @pytest.mark.asyncio
    async def test_empty_text_adds_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_embed_texts(texts: list[str], **kwargs: Any) -> list[list[float]]:
            raise AssertionError("should not embed when there's nothing to chunk")

        monkeypatch.setattr("ragscratch.rag.embed_texts", fake_embed_texts)
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

        monkeypatch.setattr("ragscratch.rag.embed_texts", fake_embed_texts)
        index = CosineSimilarityIndex()

        added = await ingest_text(
            index, "one two three four five six", source="doc.txt", chunk_size=3, overlap=0
        )

        assert added == len(index) == 2
        assert captured["texts"] == ["one two three", "four five six"]
        assert all(c.source == "doc.txt" for c in index._chunks)  # noqa: SLF001


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

        monkeypatch.setattr("ragscratch.rag.embed_texts", fake_embed_texts)
        monkeypatch.setattr("ragscratch.rag.answer_question", fake_answer_question)

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
    async def test_returns_no_sources_when_the_index_is_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_embed_texts(texts: list[str], **kwargs: Any) -> list[list[float]]:
            return [[1.0, 0.0]]

        async def fake_answer_question(
            question: str, chunks: list[ScoredChunk], **kwargs: Any
        ) -> str:
            return "I don't know."

        monkeypatch.setattr("ragscratch.rag.embed_texts", fake_embed_texts)
        monkeypatch.setattr("ragscratch.rag.answer_question", fake_answer_question)

        result = await ask(CosineSimilarityIndex(), "Anything?")

        assert result.sources == []
        assert result.answer == "I don't know."
