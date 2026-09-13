"""Request/response schemas for the /embed endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    """A batch of texts to embed.

    Attributes:
        texts: The texts to embed, in order. Must be non-empty.
        model: Model spec to use, e.g. ``"local/all-mpnet-base-v2"`` or
            ``"openai/text-embedding-3-small"``. Omit to use the service's
            configured default.
    """

    texts: list[str] = Field(min_length=1)
    model: str | None = None


class EmbedResponse(BaseModel):
    """The resulting vectors.

    Attributes:
        model: The model spec actually used.
        embeddings: One vector per input text, same order as the request.
    """

    model: str
    embeddings: list[list[float]]
