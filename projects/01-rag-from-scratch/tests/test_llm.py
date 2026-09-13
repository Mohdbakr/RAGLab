"""Tests for ragscratch.llm, written before the implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from ragscratch.index import DocumentChunk, ScoredChunk
from ragscratch.llm import answer_question, build_context_block, build_messages


def scored(text: str, source: str, score: float) -> ScoredChunk:
    return ScoredChunk(chunk=DocumentChunk(text=text, source=source, embedding=[0.0]), score=score)


class TestBuildContextBlock:
    def test_empty_chunks_yields_empty_string(self) -> None:
        assert build_context_block([]) == ""

    def test_numbers_chunks_and_includes_their_source(self) -> None:
        block = build_context_block(
            [scored("first fact", "a.txt", 0.9), scored("second fact", "b.txt", 0.8)]
        )

        assert "[1]" in block
        assert "a.txt" in block
        assert "first fact" in block
        assert "[2]" in block
        assert "b.txt" in block
        assert "second fact" in block


class TestBuildMessages:
    def test_includes_a_system_prompt_and_the_question(self) -> None:
        messages = build_messages("What is X?", [])

        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert "What is X?" in messages[-1]["content"]

    def test_notes_when_no_context_was_retrieved(self) -> None:
        messages = build_messages("What is X?", [])
        assert "No context" in messages[-1]["content"]

    def test_includes_the_context_block_when_chunks_are_given(self) -> None:
        messages = build_messages("What is X?", [scored("X is a thing.", "a.txt", 0.9)])
        assert "X is a thing." in messages[-1]["content"]
        assert "[1]" in messages[-1]["content"]


@dataclass
class FakeMessage:
    content: str | None


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeCompletionResponse:
    choices: list[FakeChoice]


class TestAnswerQuestion:
    @pytest.mark.asyncio
    async def test_forwards_model_and_returns_the_completion_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        async def fake_acompletion(**kwargs: Any) -> FakeCompletionResponse:
            captured.update(kwargs)
            return FakeCompletionResponse(choices=[FakeChoice(message=FakeMessage(content="42"))])

        monkeypatch.setattr("ragscratch.llm.litellm.acompletion", fake_acompletion)

        answer = await answer_question(
            "What is the answer?",
            [scored("The answer is 42.", "a.txt", 0.9)],
            model="gpt-4o-mini",
        )

        assert answer == "42"
        assert captured["model"] == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_returns_empty_string_rather_than_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_acompletion(**kwargs: Any) -> FakeCompletionResponse:
            return FakeCompletionResponse(choices=[FakeChoice(message=FakeMessage(content=None))])

        monkeypatch.setattr("ragscratch.llm.litellm.acompletion", fake_acompletion)

        answer = await answer_question("Q?", [])

        assert answer == ""
