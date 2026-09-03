"""
Tests for Task 5.7 — Health Check Endpoint.

Covers:
  - Both healthy → 200 with redis=ok, chroma=ok
  - Redis down → 503 with redis=error:..., chroma=ok
  - Chroma down → 503 with redis=ok, chroma=error:...
  - Both down → 503 with both error fields set
  - Error messages include exception detail
  - Response shape matches HealthResponse schema
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_redis_client, get_vector_store
from app.main import app


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_redis():
    m = AsyncMock()
    m.ping.return_value = True
    return m


@pytest.fixture
def mock_vector_store():
    m = MagicMock()
    m.heartbeat.return_value = 1234567890
    return m


@pytest.fixture
def client(mock_redis, mock_vector_store):
    app.dependency_overrides[get_redis_client] = lambda: mock_redis
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    yield TestClient(app)
    app.dependency_overrides.clear()


# ── Happy path ────────────────────────────────────────────────────────────────


def test_both_healthy_returns_200(client):
    assert client.get("/health").status_code == 200


def test_both_healthy_response_body(client):
    body = client.get("/health").json()
    assert body["redis"] == "ok"
    assert body["chroma"] == "ok"


# ── Redis failure ─────────────────────────────────────────────────────────────


def test_redis_down_returns_503(mock_vector_store):
    broken_redis = AsyncMock()
    broken_redis.ping.side_effect = ConnectionError("Connection refused")

    app.dependency_overrides[get_redis_client] = lambda: broken_redis
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    try:
        resp = TestClient(app).get("/health")
        assert resp.status_code == 503
    finally:
        app.dependency_overrides.clear()


def test_redis_down_body_shows_error(mock_vector_store):
    broken_redis = AsyncMock()
    broken_redis.ping.side_effect = ConnectionError("Connection refused")

    app.dependency_overrides[get_redis_client] = lambda: broken_redis
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    try:
        body = TestClient(app).get("/health").json()
        assert body["redis"].startswith("error:")
        assert body["chroma"] == "ok"
    finally:
        app.dependency_overrides.clear()


# ── ChromaDB failure ──────────────────────────────────────────────────────────


def test_chroma_down_returns_503(mock_redis):
    broken_store = MagicMock()
    broken_store.heartbeat.side_effect = Exception("ChromaDB unreachable")

    app.dependency_overrides[get_redis_client] = lambda: mock_redis
    app.dependency_overrides[get_vector_store] = lambda: broken_store
    try:
        resp = TestClient(app).get("/health")
        assert resp.status_code == 503
    finally:
        app.dependency_overrides.clear()


def test_chroma_down_body_shows_error(mock_redis):
    broken_store = MagicMock()
    broken_store.heartbeat.side_effect = Exception("ChromaDB unreachable")

    app.dependency_overrides[get_redis_client] = lambda: mock_redis
    app.dependency_overrides[get_vector_store] = lambda: broken_store
    try:
        body = TestClient(app).get("/health").json()
        assert body["chroma"].startswith("error:")
        assert body["redis"] == "ok"
    finally:
        app.dependency_overrides.clear()


# ── Both down ─────────────────────────────────────────────────────────────────


def test_both_down_returns_503():
    broken_redis = AsyncMock()
    broken_redis.ping.side_effect = ConnectionError("Redis unreachable")
    broken_store = MagicMock()
    broken_store.heartbeat.side_effect = Exception("ChromaDB unreachable")

    app.dependency_overrides[get_redis_client] = lambda: broken_redis
    app.dependency_overrides[get_vector_store] = lambda: broken_store
    try:
        resp = TestClient(app).get("/health")
        assert resp.status_code == 503
        body = resp.json()
        assert body["redis"].startswith("error:")
        assert body["chroma"].startswith("error:")
    finally:
        app.dependency_overrides.clear()


# ── Error detail propagation ──────────────────────────────────────────────────


def test_redis_error_message_included_in_body(mock_vector_store):
    broken_redis = AsyncMock()
    broken_redis.ping.side_effect = ConnectionError("ECONNREFUSED 127.0.0.1:6379")

    app.dependency_overrides[get_redis_client] = lambda: broken_redis
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    try:
        body = TestClient(app).get("/health").json()
        assert "ECONNREFUSED" in body["redis"]
    finally:
        app.dependency_overrides.clear()


def test_chroma_error_message_included_in_body(mock_redis):
    broken_store = MagicMock()
    broken_store.heartbeat.side_effect = Exception("Connection to localhost:8000 failed")

    app.dependency_overrides[get_redis_client] = lambda: mock_redis
    app.dependency_overrides[get_vector_store] = lambda: broken_store
    try:
        body = TestClient(app).get("/health").json()
        assert "localhost:8000" in body["chroma"]
    finally:
        app.dependency_overrides.clear()
