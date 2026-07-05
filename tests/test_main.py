"""
Tests for Task 5.1 — FastAPI Application Bootstrap.

Covers:
  - App metadata (title, version)
  - Router prefix registration
  - HTTPException handler → {"error": ...}
  - RequestValidationError handler → {"error": "Invalid request", "detail": [...]}
  - Catch-all 500 handler → {"error": "Internal server error"}
"""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.main import app, v1_router


@pytest.fixture
def client():
    return TestClient(app, raise_server_exceptions=False)


# ── App metadata ──────────────────────────────────────────────────────────────


def test_app_title():
    assert app.title == "Enterprise RAG"


def test_app_version():
    assert app.version == "0.1.0"


# ── Router registration ───────────────────────────────────────────────────────


def test_v1_router_prefix():
    assert v1_router.prefix == "/api/v1"


def test_v1_router_is_exported():
    # The router is a public symbol — later tasks import and add routes to it.
    from app.main import v1_router as imported
    assert imported is v1_router


# ── Error handler: HTTPException ──────────────────────────────────────────────


def test_http_exception_returns_error_shape(client):
    # Add a temporary route that raises an HTTPException, then hit it.
    @app.get("/test-http-exc")
    async def _raise():
        raise HTTPException(status_code=404, detail="not found")

    resp = client.get("/test-http-exc")
    assert resp.status_code == 404
    assert resp.json() == {"error": "not found"}


def test_http_exception_preserves_status_code(client):
    @app.get("/test-403")
    async def _raise():
        raise HTTPException(status_code=403, detail="forbidden")

    resp = client.get("/test-403")
    assert resp.status_code == 403


# ── Error handler: RequestValidationError ────────────────────────────────────


def test_validation_error_returns_error_shape(client):
    from pydantic import BaseModel

    class Body(BaseModel):
        count: int

    @app.post("/test-validation")
    async def _validate(body: Body):
        return {"count": body.count}

    # Send a string where an int is expected.
    resp = client.post("/test-validation", json={"count": "not-an-int"})
    assert resp.status_code == 422
    data = resp.json()
    assert data["error"] == "Invalid request"
    assert "detail" in data


# ── Error handler: unhandled Exception ───────────────────────────────────────


def test_unhandled_exception_returns_500(client):
    @app.get("/test-500")
    async def _crash():
        raise RuntimeError("something broke")

    resp = client.get("/test-500")
    assert resp.status_code == 500
    assert resp.json() == {"error": "Internal server error"}
