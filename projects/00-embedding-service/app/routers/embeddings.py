"""The /embed endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from raglab_common import BenchmarkLogger

from app.core.client_cache import EmbeddingClientCache
from app.core.config import Settings
from app.dependencies import get_benchmark_logger, get_client_cache, get_settings
from app.models.schemas import EmbedRequest, EmbedResponse

router = APIRouter()


@router.post("/embed", response_model=EmbedResponse)
async def embed(
    body: EmbedRequest,
    settings: Settings = Depends(get_settings),
    cache: EmbeddingClientCache = Depends(get_client_cache),
    benchmark: BenchmarkLogger = Depends(get_benchmark_logger),
) -> EmbedResponse:
    """Embed a batch of texts with the requested (or default) model."""
    model = body.model or settings.default_embedding_model
    client = cache.get(model)

    with benchmark.measure("embed") as fields:
        vectors = await client.embed(body.texts)
        fields["retrieved_k"] = len(body.texts)

    return EmbedResponse(model=model, embeddings=vectors)


@router.get("/healthz", tags=["Health"])
async def healthz() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}
