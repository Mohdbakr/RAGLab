from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter

from ...core.logging import SingletonLogger

logger = SingletonLogger.get_logger()


class TextChunker:
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

    async def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        logger.info(
            f"Chunking text with size {self.chunk_size}\
                and overlap {self.chunk_overlap}"
        )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )

        text_chunks = splitter.split_text(text)

        return text_chunks

    def __str__(self):
        return (
            f"Text Chunker with size {self.chunk_size} "
            + f"and overlap {self.chunk_overlap}"
        )

    def __repr__(self):
        return (
            f"Text Chunker(chunk_size={self.chunk_size}, "
            + f"chunk_overlap={self.chunk_overlap})"
        )
