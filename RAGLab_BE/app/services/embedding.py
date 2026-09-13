import numpy as np
from typing import List

from sentence_transformers import SentenceTransformer
from fastapi import Depends, HTTPException, status, Request

from ..core.logging import SingletonLogger

logger = SingletonLogger.get_logger()


class EmbeddingService:
    """Service for handling text embeddings using SentenceTransformer."""

    def __init__(self, model: SentenceTransformer):
        self.model = model

    async def get_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text input.

        Args:
            text: Input text to embed

        Returns:
            numpy.ndarray: Text embedding vector
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    async def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List[numpy.ndarray]: List of embedding vectors
        """
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings batch: {str(e)}")
            raise

    def __str__(self):
        return f"Embedding Service with model: {self.model}"

    def __repr__(self):
        return f"Embedding Service(model={self.model})"


async def get_embedding_model(request: Request) -> SentenceTransformer:
    """Dependency to get the embedding model from app state."""
    if not hasattr(request.app.state, "embedding_model"):
        raise HTTPException(
            status_code=503, detail="Embedding model not initialized"
        )
    return request.app.state.embedding_model


def get_embedding_service(
    model: SentenceTransformer = Depends(get_embedding_model),
) -> EmbeddingService:
    """
    Dependency for getting the embedding service.

    Args:
        model: SentenceTransformer model instance

    Returns:
        EmbeddingService: Initialized embedding service

    Raises:
        HTTPException: If model initialization fails
    """
    try:
        return EmbeddingService(model)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to initialize embedding service: {str(e)}",
        )
