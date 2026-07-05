import redis.asyncio as aioredis
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.dependencies import get_redis_client, get_vector_store
from app.schemas.health import HealthResponse
from app.vector_db import VectorStoreManager

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    redis_client: aioredis.Redis = Depends(get_redis_client),
    vector_store: VectorStoreManager = Depends(get_vector_store),
) -> JSONResponse:
    redis_status = "ok"
    chroma_status = "ok"

    try:
        await redis_client.ping()
    except Exception as exc:
        redis_status = f"error: {exc}"

    try:
        vector_store.heartbeat()
    except Exception as exc:
        chroma_status = f"error: {exc}"

    healthy = redis_status == "ok" and chroma_status == "ok"
    # 503 when any dependency is down — Kubernetes readiness probes check the status code.
    status_code = 200 if healthy else 503

    return JSONResponse(
        status_code=status_code,
        content=HealthResponse(redis=redis_status, chroma=chroma_status).model_dump(),
    )
