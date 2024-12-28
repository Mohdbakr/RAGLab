from abc import ABC, abstractmethod
from typing import List, Dict, Any, BinaryIO
from dataclasses import dataclass
from enum import Enum

from ...core.logging import setup_logging

logger = setup_logging()


class DocumentType(Enum):
    """Supported document types."""

    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    MD = "md"


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document."""

    text: str
    metadata: Dict[str, Any]


@dataclass
class ProcessedDocument:
    """Represents a fully processed document with chunks."""

    chunks: List[DocumentChunk]
    metadata: Dict[str, Any]


class DocumentProcessor(ABC):
    """Base class for document processors."""

    @abstractmethod
    async def can_process(self, file_extension: str) -> bool:
        """Check if this processor can handle the file type."""
        pass

    @abstractmethod
    async def extract_text(self, file: BinaryIO) -> str:
        """Extract text from the document."""
        pass


class TextChunker(ABC):
    """Base class for text chunking strategies."""

    @abstractmethod
    async def chunk_text(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Split text into chunks with metadata."""
        logger.debug(
            f"Chunking text with size {self.chunk_size} and overlap {self.chunk_overlap}"
        )
        pass
