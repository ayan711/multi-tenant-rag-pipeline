from typing import Any

from pydantic import BaseModel


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    # Any, not a typed model: the shape is whatever the underlying Celery task
    # returns (ingest_document_pipeline's summary dict today) — this schema
    # doesn't know or care which task ran, only that it's a status envelope.
    result: Any | None = None
    # Populated only on FAILURE — the router stringifies the caught exception
    # before it gets here, since a raw exception instance isn't JSON-serialisable.
    error: str | None = None
