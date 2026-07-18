# ─────────────────────────────────────────────────────────────────────────────
# main.py — FastAPI Application Entry Point
# ─────────────────────────────────────────────────────────────────────────────
#
# This file is intentionally thin: app instantiation, error handlers, and
# router mounting only. All route handlers live in app/routers/<feature>.py.
# Adding a new feature = create app/routers/<feature>.py, import it here.
# ─────────────────────────────────────────────────────────────────────────────

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRouter

from app.routers import health, query, tasks, upload

logger = logging.getLogger(__name__)

# ── Application instance ──────────────────────────────────────────────────────

app = FastAPI(
    title="Enterprise RAG",
    description="Multi-tenant document Q&A engine — upload PDFs, get grounded answers.",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Versioned base router ─────────────────────────────────────────────────────
#
# Feature routers (upload, query, health, tasks) are included here.
# The /api/v1 prefix is applied once here, not repeated in each feature router.

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(upload.router)
v1_router.include_router(query.router)
v1_router.include_router(tasks.router)

# ── Error handlers ────────────────────────────────────────────────────────────


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    # Pydantic v2 @field_validator puts the raised exception object into
    # error['ctx']['error'], which is not JSON-serializable. Stringify it.
    def _safe_errors(errors: list) -> list:
        safe = []
        for e in errors:
            entry = dict(e)
            if isinstance(entry.get("ctx"), dict):
                entry["ctx"] = {k: str(v) for k, v in entry["ctx"].items()}
            safe.append(entry)
        return safe

    return JSONResponse(
        status_code=422,
        content={"error": "Invalid request", "detail": _safe_errors(exc.errors())},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, _exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"error": "Internal server error"})


# ── Mount ─────────────────────────────────────────────────────────────────────

app.include_router(v1_router)
# /health is top-level (not versioned) — standard for k8s readiness/liveness probes.
app.include_router(health.router)
