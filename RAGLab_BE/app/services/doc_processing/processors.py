import PyPDF2
import docx
import markdown

from .interfaces import DocumentProcessor, DocumentType
from ...core.logging import setup_logging

logger = setup_logging()


class PDFProcessor(DocumentProcessor):
    """Processor for PDF files."""

    async def can_process(self, file_extension: str) -> bool:
        logger.debug(f"Checking if PDF processor can handle {file_extension}")
        return file_extension.lower() == DocumentType.PDF.value

    async def extract_text(self, file: str) -> str:
        with open(file, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()


class TXTProcessor(DocumentProcessor):
    """Processor for text files."""

    async def can_process(self, file_extension: str) -> bool:
        return file_extension.lower() == DocumentType.TXT.value

    async def extract_text(self, file: str) -> str:
        with open(file, "r", encoding="utf-8") as f:
            return f.read().strip()


class DOCXProcessor(DocumentProcessor):
    """Processor for DOCX files."""

    async def can_process(self, file_extension: str) -> bool:
        return file_extension.lower() == DocumentType.DOCX.value

    async def extract_text(self, file: str) -> str:
        doc = docx.Document(file)
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])


class MarkdownProcessor(DocumentProcessor):
    """Processor for Markdown files."""

    async def can_process(self, file_extension: str) -> bool:
        return file_extension.lower() == DocumentType.MD.value

    async def extract_text(self, file: str) -> str:
        md_text = file.read().decode("utf-8")
        html = markdown.markdown(md_text)
        # TODO: Need more robust HTML tag removal
        text = html.replace("<p>", "\n").replace("</p>", "\n")
        return " ".join(text.split())


class ProcessorFactory:
    """Factory for creating document processors."""

    _processors = {
        DocumentType.PDF: PDFProcessor(),
        DocumentType.TXT: TXTProcessor(),
        DocumentType.DOCX: DOCXProcessor(),
        DocumentType.MD: MarkdownProcessor(),
    }

    @classmethod
    async def get_processor(cls, file_extension: str) -> DocumentProcessor:
        """Get appropriate processor for file type."""
        logger.debug(f"Getting processor for {file_extension}")
        for processor in cls._processors.values():
            if await processor.can_process(file_extension):
                logger.debug(f"Found {processor} for {file_extension}")
                return processor
        raise ValueError(f"Unsupported file type: {file_extension}")
