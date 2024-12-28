from typing import List, Dict, Any
import os
from datetime import datetime

from ..embedding import EmbeddingService
from .interfaces import ProcessedDocument, DocumentChunk, TextChunker
from .processors import ProcessorFactory
from .chunkers import OverlappingChunker
from ...core.logging import setup_logging

logger = setup_logging()


class DocumentProcessingPipeline:
    """Main pipeline for processing documents."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        # vector_store: MilvusRepository,
        chunker: TextChunker = None,
    ):
        self.embedding_service = embedding_service
        # self.vector_store = vector_store
        self.chunker = chunker or OverlappingChunker()

    async def process_file(
        self, file_path: str, filename: str, metadata: Dict[str, Any] = None
    ) -> ProcessedDocument:
        """
        Process a file through the complete pipeline.

        Args:
            file: File-like object
            filename: Original filename
            metadata: Additional metadata

        Returns:
            ProcessedDocument: Processed document with chunks
        """
        logger.info(f"getting file: {filename} extension")
        # Extract file extension
        file_extension = os.path.splitext(filename)[1][1:]

        logger.info(
            f"getting processor for file: {filename} with extension: {file_extension}"
        )
        # Get appropriate processor
        processor = await ProcessorFactory.get_processor(file_extension)

        # Create base metadata
        logger.info(f"creating base metadata for file: {filename}")
        base_metadata = {
            "filename": filename,
            "file_type": file_extension,
            "processed_at": datetime.utcnow().isoformat(),
            **(metadata or {}),
        }

        # Extract text
        logger.info(f"extracting text from file: {filename}")
        text = await processor.extract_text(file_path)

        # Create chunks
        logger.info(f"creating chunks from text of file: {filename}")
        chunks = await self.chunker.chunk_text(text, base_metadata)

        # Process chunks and store in vector database
        logger.info(f"processing and storing chunks for file: {filename}")
        await self._process_and_store_chunks(chunks)

        return ProcessedDocument(chunks=chunks, metadata=base_metadata)

    async def _process_and_store_chunks(self, chunks: List[DocumentChunk]):
        """Process chunks and store in vector database."""
        # Generate embeddings for all chunks
        logger.info("getting embeddings for chunks")
        texts = [chunk.text for chunk in chunks]
        embeddings = await self.embedding_service.get_embeddings(texts)

        logger.info("storing embeddings in vector database")
        return {"embeddings": embeddings, "texts": texts}
        # Store in vector database
        # await self.vector_store.insert_many(
        #     embeddings=embeddings,
        #     metadatas=[chunk.metadata for chunk in chunks],
        #     texts=texts,
        # )
