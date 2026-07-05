# ─────────────────────────────────────────────────────────────────────────────
# worker.py — Celery Background Task Worker
# ─────────────────────────────────────────────────────────────────────────────
#
# PDF ingestion is slow (parse → chunk → embed → store can take several seconds
# per document). Doing this inside an HTTP request handler would block the
# server and time out for large files.
#
# Solution: offload the work to a Celery worker running as a separate process.
#
# How it works:
#   1. main.py receives the uploaded file and drops a job onto a Redis queue.
#   2. This worker process is always listening to that queue.
#   3. When a job arrives, the worker picks it up and runs the full ingestion
#      pipeline (parse → chunk → embed → store in ChromaDB).
#   4. The HTTP response was already sent (202 Accepted) while this runs in the background.
#
# Important pattern — lazy loading:
#   The LocalEmbedder and VectorStoreManager instances are created OUTSIDE the
#   task function, at module load time. This means the 80MB model weights are
#   loaded into memory exactly once when the worker starts, not on every task.
#   Loading them inside the task function would add ~2s to every ingestion job.
#
# Implemented in: Task 4.1 through Task 4.5.

import os

from celery import Celery

from app.config import settings
from app.embedder import LocalEmbedder
from app.parser import DocumentParser
from app.vector_db import VectorStoreManager

# ---------------------------------------------------------------------------
# Celery application instance
#
# broker_url   — Redis queue where FastAPI pushes task messages.
#                The worker polls this constantly, waiting for work.
#
# result_backend — same Redis instance, different logical DB (db=0 here).
#                  After a task finishes, the worker writes status + return
#                  value here so callers can check progress via
#                  AsyncResult(task_id).state / .result.
#
# Why the same Redis for both?  For a project of this scale it keeps ops
# simple — one container, one URL.  In production you might separate them
# to avoid broker messages and result data competing for memory.
# ---------------------------------------------------------------------------
celery_app = Celery(
    "enterprise_rag",                 # logical name shown in Celery logs
    broker=settings.redis_url,
    backend=settings.redis_url,
)

# ---------------------------------------------------------------------------
# Runtime configuration
#
# task_serializer / result_serializer / accept_content:
#   JSON is explicit, human-readable, and safe against pickle deserialization
#   attacks that can execute arbitrary code.  Celery's default is pickle;
#   we override it here.
#
# task_track_started:
#   By default a task's state jumps from PENDING → SUCCESS (or FAILURE).
#   Setting this to True adds a STARTED state so Task 5.9's status-polling
#   endpoint can distinguish "queued" from "actively running".
#
# timezone / enable_utc:
#   Store all timestamps in UTC so logs are unambiguous regardless of where
#   the worker process is running.
# ---------------------------------------------------------------------------
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    timezone="UTC",
    enable_utc=True,
    result_expires=3600,  # drop completed task results from Redis after 1 hour
)

# ---------------------------------------------------------------------------
# Module-level singletons — Task 4.2
#
# Both objects are expensive to create:
#   LocalEmbedder      — loads ~80 MB of model weights from disk on first call.
#   VectorStoreManager — opens an HTTP connection to ChromaDB.
#
# Placing them here (not inside a task function) means the cost is paid once
# when the worker process starts, not on every task invocation.
# A document with 200 chunks ingested 10 times would otherwise reload the
# model 10×, adding ~1 s per task for no reason.
# ---------------------------------------------------------------------------
embedder = LocalEmbedder()
vector_store = VectorStoreManager()


# ---------------------------------------------------------------------------
# Task definitions — Task 4.3
# ---------------------------------------------------------------------------

@celery_app.task(name="enterprise_rag.ingest_document_pipeline")
def ingest_document_pipeline(path: str, tenant_id: str, file_hash: str) -> dict:
    """Full PDF ingestion pipeline — runs asynchronously in the Celery worker.

    Steps: extract pages → chunk → embed batch → insert to ChromaDB → delete temp file.

    Args:
        path:      Absolute path to the temporary PDF on disk (written by FastAPI).
        tenant_id: Scopes all inserted vectors to this tenant's partition.
        file_hash: SHA-256 hex digest of the original file bytes.
                   Used as the vector ID prefix and idempotency key.

    Returns:
        {"status": "ingested"|"already_exists", "file_hash": str, "chunks_inserted": int, "tenant_id": str}
        Stored in Redis; readable via AsyncResult(task_id).result.
        FastAPI (Task 5.4) checks document_exists() *before* calling .delay(), so the
        "already_exists" path here is a belt-and-suspenders guard for edge cases
        (e.g. two concurrent uploads of the same file before either finishes).
    """
    # Idempotency guard — skip re-processing if this file/tenant pair is already indexed.
    if vector_store.document_exists(file_hash, tenant_id):
        if os.path.exists(path):
            os.unlink(path)
        return {"status": "already_exists", "file_hash": file_hash, "chunks_inserted": 0, "tenant_id": tenant_id}

    try:
        parser = DocumentParser()
        # 1 - Extract pages from the PDF and chunk them into smaller pieces
        pages = parser.extract_text_per_page(path)
        chunks = parser.chunk_pages(pages)

        if chunks:
            texts = [c["text"] for c in chunks]
            # 2 - Embed all chunks in one forward pass
            embeddings = embedder.embed_batch(texts)
            count = vector_store.insert_chunks(chunks, embeddings, tenant_id, file_hash)
        else:
            # Blank PDF — nothing to embed; avoid calling ChromaDB with empty lists
            # (it raises ValueError: "You must provide at least one embedding to add.")
            count = 0

    finally:
        # Always delete the temp file, even when an earlier step raises.
        # FastAPI saves the upload here before dispatching; orphaned files would
        # silently exhaust disk storage over time.
        if os.path.exists(path):
            os.unlink(path)

    return {"status": "ingested", "file_hash": file_hash, "chunks_inserted": count, "tenant_id": tenant_id}
