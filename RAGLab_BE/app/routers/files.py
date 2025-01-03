import os

from fastapi import (
    APIRouter,
    File,
    UploadFile,
    HTTPException,
    BackgroundTasks,
    Depends,
)

from ..models.documents import validate_document
from ..services.embedding import EmbeddingService, get_embedding_service
from ..core.logging import SingletonLogger
from ..services.doc_processing.pipeline import DocumentProcessingPipeline
from ..services.doc_processing.chunkers import TextChunker

router = APIRouter(prefix="/documents")
UPLOAD_DIR = "./api_data/file_locker/"
os.makedirs(UPLOAD_DIR, exist_ok=True)
logger = SingletonLogger.get_logger()


@router.post("/upload/")
async def file_upload(
    background_task: BackgroundTasks,
    file: UploadFile = File(...),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
):
    if not validate_document(file):
        raise HTTPException(status_code=400, detail="Invalid document type")
    try:
        fname = os.path.join(UPLOAD_DIR, file.filename)

        # Create pipeline with sentence-based chunking
        pipeline = DocumentProcessingPipeline(
            embedding_service=embedding_service,
            # vector_store=vector_store,
            chunker=TextChunker(chunk_size=512, chunk_overlap=32),
        )
        logger.info(
            f"Processing file: {file.filename}\n"
            f"{pipeline.embedding_service}\n"
            f"{pipeline.chunker}"
        )
        with open(fname, "wb") as f:
            while content := file.file.read(1024 * 1024):
                f.write(content)
        logger.info(f"File saved: {file.filename}")
        background_task.add_task(pipeline.process_file, fname, file.filename)
        logger.info(f"Processing file: {file.filename}")
        return {"message": f"Started processing file: {file.filename}."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
