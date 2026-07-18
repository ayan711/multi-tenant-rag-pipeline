import hashlib

from celery import Celery
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from app.dependencies import get_celery_app, get_vector_store
from app.storage.base import StorageBackend
from app.storage.local import get_local_storage
from app.vector_db import VectorStoreManager

router = APIRouter(prefix="/docs", tags=["documents"])

# PDF files always start with the header %PDF (Magic bytes)
_PDF_MAGIC = b"%PDF"

# Dispatched by name (not imported from app.worker) so the FastAPI process
# never triggers worker.py's module-level LocalEmbedder/VectorStoreManager
# singletons — those are only meant to load inside the Celery worker process.
_INGEST_TASK_NAME = "enterprise_rag.ingest_document_pipeline"


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    tenant_id: str = Form(...),
    file: UploadFile = File(...),
    storage: StorageBackend = Depends(get_local_storage),
    vector_store: VectorStoreManager = Depends(get_vector_store),
    celery_app: Celery = Depends(get_celery_app),
) -> JSONResponse:
    data = await file.read()

    # Two-layer check: MIME type (client claim) + magic bytes (actual content).
    if file.content_type != "application/pdf" or not data.startswith(_PDF_MAGIC):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    file_hash = hashlib.sha256(data).hexdigest()

    # Task 4.5's guard also runs inside the worker, but checking here first
    # skips the disk write and the task dispatch entirely for a re-upload.
    if vector_store.document_exists(file_hash, tenant_id):
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "already_exists", "file_hash": file_hash},
        )

    # Delegate persistence to the injected backend.
    # LocalStorage → absolute path on disk.
    # Future S3Storage → s3://bucket/key URI.
    locator = storage.save(file_hash, data)

    async_result = celery_app.send_task(
        _INGEST_TASK_NAME, args=[locator, tenant_id, file_hash]
    )

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"task_id": async_result.id, "file_hash": file_hash},
    )
