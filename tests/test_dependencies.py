"""
Tests for dependency-provider error handling.

Covers:
  - get_vector_store() converts a construction failure (ChromaDB unreachable)
    into a clean HTTPException(503) instead of letting the raw exception
    propagate as an unhandled 500.
  - A failed construction is not cached — lru_cache only memoizes successes,
    so the next call retries construction from scratch.
  - The route surfaces that 503 to the client (via main.py's HTTPException
    handler) instead of the generic "Internal server error" 500 body.
"""

from unittest.mock import patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import dependencies
from app.main import app


@pytest.fixture(autouse=True)
def _clear_vector_store_cache():
    """Every test gets a fresh lru_cache slot — a prior test's cached instance
    (real or mocked) must not leak into the next test's assertions."""
    dependencies.get_vector_store.cache_clear()
    yield
    dependencies.get_vector_store.cache_clear()


def test_get_vector_store_raises_503_on_construction_failure():
    with patch.object(
        dependencies, "VectorStoreManager", side_effect=ConnectionError("chroma down")
    ):
        with pytest.raises(HTTPException) as exc_info:
            dependencies.get_vector_store()

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Vector store is unavailable"


def test_get_vector_store_failure_is_not_cached():
    with patch.object(
        dependencies, "VectorStoreManager", side_effect=ConnectionError("chroma down")
    ):
        with pytest.raises(HTTPException):
            dependencies.get_vector_store()

    # Second call, ChromaDB "back up" — must retry construction, not replay the error.
    with patch.object(dependencies, "VectorStoreManager", return_value="fake-store"):
        assert dependencies.get_vector_store() == "fake-store"


def test_upload_route_returns_503_when_vector_store_unavailable():
    def _raise_unavailable():
        raise HTTPException(status_code=503, detail="Vector store is unavailable")

    app.dependency_overrides[dependencies.get_vector_store] = _raise_unavailable
    try:
        client = TestClient(app)
        files = {"file": ("doc.pdf", b"%PDF-1.4 fake pdf content", "application/pdf")}
        resp = client.post(
            "/api/v1/docs/upload", files=files, data={"tenant_id": "acme"}
        )
    finally:
        app.dependency_overrides.clear()

    assert resp.status_code == 503
    assert resp.json() == {"error": "Vector store is unavailable"}
