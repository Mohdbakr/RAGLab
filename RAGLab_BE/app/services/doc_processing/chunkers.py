from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter

from .interfaces import TextChunker, DocumentChunk
from ...core.logging import setup_logging

logger = setup_logging()


class OverlappingChunker(TextChunker):
    """Chunks text with overlapping windows."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 32,
        separator: str = " ",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    async def chunk_text(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Split text into overlapping chunks."""
        logger.debug(
            f"Chunking text with size {self.chunk_size} and overlap {self.chunk_overlap}"
        )
        chunks = []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        chunks = splitter.split_text(text)

        start_idx = 0
        chunk_index = 0
        # while start_idx < len(chunks):
        for chunk in chunks:
            # Create chunk with metadata
            chunk_metadata = metadata.copy()
            logger.debug(f"Chunk index: {chunk_index}, chunk: {chunk}")
            chunk_metadata.update(
                {
                    "chunk_text": chunk,
                    "chunk_index": chunk_index,
                    "chunk_start": start_idx,
                    "chunk_size": chunk,
                }
            )

            chunks.append(DocumentChunk(text=chunk, metadata=chunk_metadata))

            # Move start index by chunk size minus overlap
            start_idx += self.chunk_size - self.chunk_overlap

        return chunks
