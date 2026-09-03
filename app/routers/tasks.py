from celery import Celery
from celery.result import AsyncResult
from fastapi import APIRouter, Depends

from app.dependencies import get_celery_app
from app.schemas.tasks import TaskStatusResponse

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    celery_app: Celery = Depends(get_celery_app),
) -> TaskStatusResponse:
    # AsyncResult never validates that task_id was ever issued — Celery just
    # looks up a key in Redis. An unknown or made-up id reads back as PENDING,
    # identical to a real task that hasn't started yet.
    async_result = AsyncResult(task_id, app=celery_app)
    status = async_result.state

    if status == "SUCCESS":
        return TaskStatusResponse(task_id=task_id, status=status, result=async_result.result)

    if status == "FAILURE":
        # On failure, .result holds the raised exception instance, not a dict —
        # stringify it so the response stays JSON-serialisable.
        return TaskStatusResponse(task_id=task_id, status=status, error=str(async_result.result))

    # PENDING / STARTED / RETRY / REVOKED — nothing to report yet besides state.
    return TaskStatusResponse(task_id=task_id, status=status)
