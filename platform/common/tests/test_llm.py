"""Tests for raglab_common.llm, written before the implementation.

litellm itself is never called — `litellm.acompletion` is monkeypatched so
these stay fast, free, and offline, and only our adapter's own logic is
under test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from raglab_common.llm import ChatMessage, LiteLLMClient, LLMClient, LLMResponse


@dataclass
class FakeUsage:
    prompt_tokens: int
    completion_tokens: int


@dataclass
class FakeMessage:
    content: str | None


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeCompletionResponse:
    choices: list[FakeChoice]
    usage: FakeUsage


class TestLLMClientProtocol:
    def test_lite_llm_client_satisfies_the_protocol(self) -> None:
        assert isinstance(LiteLLMClient(model="openai/gpt-4o-mini"), LLMClient)


class TestLiteLLMClient:
    @pytest.mark.asyncio
    async def test_complete_returns_content_and_token_usage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        async def fake_acompletion(**kwargs: Any) -> FakeCompletionResponse:
            captured.update(kwargs)
            return FakeCompletionResponse(
                choices=[FakeChoice(message=FakeMessage(content="hello there"))],
                usage=FakeUsage(prompt_tokens=12, completion_tokens=3),
            )

        monkeypatch.setattr("raglab_common.llm.litellm.acompletion", fake_acompletion)
        client = LiteLLMClient(model="openai/gpt-4o-mini")

        response = await client.complete([ChatMessage(role="user", content="hi")])

        assert isinstance(response, LLMResponse)
        assert response.content == "hello there"
        assert response.prompt_tokens == 12
        assert response.completion_tokens == 3
        assert response.model == "openai/gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_complete_forwards_model_messages_and_temperature(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        async def fake_acompletion(**kwargs: Any) -> FakeCompletionResponse:
            captured.update(kwargs)
            return FakeCompletionResponse(
                choices=[FakeChoice(message=FakeMessage(content="ok"))],
                usage=FakeUsage(prompt_tokens=1, completion_tokens=1),
            )

        monkeypatch.setattr("raglab_common.llm.litellm.acompletion", fake_acompletion)
        client = LiteLLMClient(model="anthropic/claude-3-5-sonnet-latest")

        await client.complete(
            [
                ChatMessage(role="system", content="be terse"),
                ChatMessage(role="user", content="hi"),
            ],
            temperature=0.2,
        )

        assert captured["model"] == "anthropic/claude-3-5-sonnet-latest"
        assert captured["temperature"] == 0.2
        assert captured["messages"] == [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "hi"},
        ]

    @pytest.mark.asyncio
    async def test_default_params_are_merged_under_call_time_kwargs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        async def fake_acompletion(**kwargs: Any) -> FakeCompletionResponse:
            captured.update(kwargs)
            return FakeCompletionResponse(
                choices=[FakeChoice(message=FakeMessage(content="ok"))],
                usage=FakeUsage(prompt_tokens=1, completion_tokens=1),
            )

        monkeypatch.setattr("raglab_common.llm.litellm.acompletion", fake_acompletion)
        client = LiteLLMClient(model="ollama/llama3", max_tokens=256)

        await client.complete([ChatMessage(role="user", content="hi")])

        assert captured["max_tokens"] == 256

    @pytest.mark.asyncio
    async def test_missing_usage_yields_none_token_counts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def fake_acompletion(**kwargs: Any) -> FakeCompletionResponse:
            return FakeCompletionResponse(
                choices=[FakeChoice(message=FakeMessage(content="ok"))],
                usage=None,  # type: ignore[arg-type]
            )

        monkeypatch.setattr("raglab_common.llm.litellm.acompletion", fake_acompletion)
        client = LiteLLMClient(model="openai/gpt-4o-mini")

        response = await client.complete([ChatMessage(role="user", content="hi")])

        assert response.prompt_tokens is None
        assert response.completion_tokens is None

    @pytest.mark.asyncio
    async def test_tool_call_only_response_has_no_content_and_does_not_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OpenAI/litellm set content=None when the model's turn is a tool
        call rather than text — documented as supported via `tools` in
        **kwargs, so this must return a response, not raise."""

        async def fake_acompletion(**kwargs: Any) -> FakeCompletionResponse:
            return FakeCompletionResponse(
                choices=[FakeChoice(message=FakeMessage(content=None))],
                usage=FakeUsage(prompt_tokens=5, completion_tokens=2),
            )

        monkeypatch.setattr("raglab_common.llm.litellm.acompletion", fake_acompletion)
        client = LiteLLMClient(model="openai/gpt-4o-mini")

        response = await client.complete(
            [ChatMessage(role="user", content="hi")],
            tools=[{"type": "function", "function": {"name": "lookup"}}],
        )

        assert response.content is None
        assert response.prompt_tokens == 5


class TestChatMessageRoleValidation:
    def test_rejects_unknown_role(self) -> None:
        with pytest.raises(ValueError):
            ChatMessage(role="not-a-real-role", content="x")  # type: ignore[arg-type]
