import os

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks

from ..models.documents import validate_document

router = APIRouter()
UPLOAD_DIR = "./api_data/file_locker/"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload/")
async def file_upload(file: UploadFile = File(...)):
    if not validate_document(file):
        raise HTTPException(status_code=400, detail="Invalid document type")
    try:
        fname = os.path.join(UPLOAD_DIR, file.filename)
        with open(fname, "wb") as f:
            while content := file.file.read(1024 * 1024):
                f.write(content)
        BackgroundTasks.add_task(fname)
        return {
            "message": f"Successfully uploaded {file.filename}. Saving data now..."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
