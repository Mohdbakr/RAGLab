from abc import ABC, abstractmethod
from typing import List, Dict, Any, BinaryIO
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ...core.logging import SingletonLogger

logger = SingletonLogger.get_logger()


class DocumentType(Enum):
    """Supported document types."""

    PDF = "pdf"
    TXT = "txt"
    DOCX = "docx"
    MD = "md"


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document."""

    embedding: np.ndarray
    metadata: Dict[str, Any]


@dataclass
class ProcessedDocument:
    """Represents a fully processed document with chunks."""

    chunks: List[DocumentChunk]
    # metadata: Dict[str, Any]


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

    def __str__(self):
        return f"{self.__class__.__name__} Processor"

    def __repr__(self):
        return f"{self.__class__.__name__}Processor"
