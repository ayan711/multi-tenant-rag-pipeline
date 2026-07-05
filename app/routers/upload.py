import hashlib

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.storage.base import StorageBackend
from app.storage.local import get_local_storage

router = APIRouter(prefix="/docs", tags=["documents"])

# PDF files always start with the header %PDF (Magic bytes)
_PDF_MAGIC = b"%PDF"


@router.post("/upload")
async def upload_document(
    tenant_id: str = Form(...),
    file: UploadFile = File(...),
    storage: StorageBackend = Depends(get_local_storage),
) -> dict:
    data = await file.read()

    # Two-layer check: MIME type (client claim) + magic bytes (actual content).
    if file.content_type != "application/pdf" or not data.startswith(_PDF_MAGIC):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    file_hash = hashlib.sha256(data).hexdigest()

    # Delegate persistence to the injected backend.
    # LocalStorage → absolute path on disk.
    # Future S3Storage → s3://bucket/key URI.
    locator = storage.save(file_hash, data)

    # Placeholder return — Task 5.4 replaces this with 202 + task_id,
    # passing locator and file_hash straight to Celery.
    return {
        "filename": file.filename,
        "tenant_id": tenant_id,
        "size_bytes": len(data),
        "file_hash": file_hash,
        "locator": locator,
    }
