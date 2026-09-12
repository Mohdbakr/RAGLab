"""Provider-agnostic LLM access.

Every project depends on the :class:`LLMClient` Protocol, never on a
specific vendor SDK, so switching providers — or comparing them, which is
the whole point of this repo — is a one-line config change: the ``model``
string follows litellm's ``"<provider>/<model>"`` convention, e.g.
``"openai/gpt-4o-mini"``, ``"anthropic/claude-3-5-sonnet-latest"``,
``"ollama/llama3"``, ``"azure/my-deployment"``. litellm is the engine
behind the default implementation so this package doesn't have to
hand-roll and maintain an adapter per vendor; provider credentials are
picked up the same way litellm always reads them (standard env vars like
``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ...).
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

import litellm
from pydantic import BaseModel

Role = Literal["system", "user", "assistant", "tool"]


class ChatMessage(BaseModel):
    """One turn in a chat completion request."""

    role: Role
    content: str


class LLMResponse(BaseModel):
    """A completion, normalized across providers.

    Attributes:
        content: The model's reply text.
        model: The provider/model string that produced this response.
        prompt_tokens: Prompt tokens consumed, if the provider reported it.
        completion_tokens: Completion tokens produced, if the provider
            reported it.
    """

    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@runtime_checkable
class LLMClient(Protocol):
    """The only interface RAGLab projects should depend on for chat completions."""

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """Request a chat completion.

        Args:
            messages: The conversation so far, oldest first.
            temperature: Sampling temperature.
            **kwargs: Provider-specific extras (e.g. ``max_tokens``,
                ``tools``), passed through as-is.

        Returns:
            The model's response, with token usage when available.
        """
        ...


class LiteLLMClient:
    """Default :class:`LLMClient`, backed by litellm.

    Any provider litellm supports works by passing the right ``model``
    string — this class adds no vendor-specific logic of its own.
    """

    def __init__(self, model: str, **default_params: Any) -> None:
        """Create a client for one provider/model.

        Args:
            model: A litellm model string, e.g. ``"openai/gpt-4o-mini"``.
            **default_params: Extra parameters applied to every call
                (e.g. ``max_tokens=512``), overridable per call.
        """
        self._model = model
        self._default_params = default_params

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> LLMResponse:
        """See :meth:`LLMClient.complete`."""
        params = {**self._default_params, **kwargs}
        response = await litellm.acompletion(
            model=self._model,
            messages=[m.model_dump() for m in messages],
            temperature=temperature,
            **params,
        )
        choice = response.choices[0]
        usage = getattr(response, "usage", None)
        return LLMResponse(
            content=choice.message.content,
            model=self._model,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )
