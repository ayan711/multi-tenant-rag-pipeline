"""
Tests for Task 5.2 — File Upload Endpoint.
Tests for Task 5.3 — SHA-256 Fingerprinting.
Tests for Task 5.4 — Async Task Dispatch.
Tests for storage abstraction layer.

Covers:
  - Happy path: valid PDF → 202 with task_id + file_hash; task dispatched
    via celery_app.send_task with (locator, tenant_id, file_hash)
  - Already-ingested file → 200 with status=already_exists, no dispatch,
    no disk write
  - Wrong content_type → 400
  - Non-PDF content (bad magic bytes) → 400
  - Missing tenant_id / file → 422
  - file_hash is a valid 64-char hex SHA-256 digest
  - Same file → same hash; different files → different hashes
  - Storage backend can be swapped via dependency_overrides (abstraction test)
"""

import hashlib
import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_celery_app, get_vector_store
from app.main import app
from app.storage.base import StorageBackend
from app.storage.local import LocalStorage, get_local_storage

# Minimal valid-looking PDF: starts with the %PDF magic bytes.
_FAKE_PDF = b"%PDF-1.4 fake pdf content"
_NOT_PDF = b"this is plain text, not a pdf"


def _mock_vector_store(exists: bool = False):
    mock = MagicMock()
    mock.document_exists.return_value = exists
    return mock


def _mock_celery_app(task_id: str = "fake-task-id"):
    mock = MagicMock()
    mock.send_task.return_value.id = task_id
    return mock


@pytest.fixture
def client():
    """Default: document not yet indexed, Celery dispatch mocked (no broker needed)."""
    app.dependency_overrides[get_vector_store] = lambda: _mock_vector_store(exists=False)
    app.dependency_overrides[get_celery_app] = _mock_celery_app
    yield TestClient(app)
    app.dependency_overrides.clear()


def _upload(client, *, content=_FAKE_PDF, content_type="application/pdf", tenant_id="acme"):
    """Helper: POST to /api/v1/docs/upload with sensible defaults."""
    files = {"file": ("doc.pdf", io.BytesIO(content), content_type)}
    data = {"tenant_id": tenant_id} if tenant_id is not None else {}
    return client.post("/api/v1/docs/upload", files=files, data=data)


# ── Happy path ────────────────────────────────────────────────────────────────


def test_upload_valid_pdf_returns_202(client):
    resp = _upload(client)
    assert resp.status_code == 202


def test_upload_response_contains_expected_fields(client):
    body = _upload(client, tenant_id="acme").json()
    assert body["task_id"] == "fake-task-id"
    assert "file_hash" in body


def test_upload_dispatches_task_with_locator_tenant_and_hash(tmp_path):
    """celery_app.send_task must receive the storage locator, tenant_id, and file_hash."""
    storage = LocalStorage(base_dir=str(tmp_path))
    mock_celery = _mock_celery_app()

    app.dependency_overrides[get_local_storage] = lambda: storage
    app.dependency_overrides[get_vector_store] = lambda: _mock_vector_store(exists=False)
    app.dependency_overrides[get_celery_app] = lambda: mock_celery
    try:
        client = TestClient(app)
        body = _upload(client, tenant_id="acme").json()
        expected_locator = str(tmp_path / f"{body['file_hash']}.pdf")
        mock_celery.send_task.assert_called_once_with(
            "enterprise_rag.ingest_document_pipeline",
            args=[expected_locator, "acme", body["file_hash"]],
        )
    finally:
        app.dependency_overrides.clear()


# ── Idempotency guard (Task 4.5, checked before dispatch) ──────────────────────


def test_already_ingested_file_returns_200(client):
    app.dependency_overrides[get_vector_store] = lambda: _mock_vector_store(exists=True)
    resp = _upload(client)
    assert resp.status_code == 200
    assert resp.json() == {"status": "already_exists", "file_hash": hashlib.sha256(_FAKE_PDF).hexdigest()}


def test_already_ingested_file_skips_dispatch(client):
    mock_celery = _mock_celery_app()
    app.dependency_overrides[get_vector_store] = lambda: _mock_vector_store(exists=True)
    app.dependency_overrides[get_celery_app] = lambda: mock_celery
    _upload(client)
    mock_celery.send_task.assert_not_called()


# ── PDF validation ────────────────────────────────────────────────────────────


def test_wrong_content_type_returns_400(client):
    resp = _upload(client, content_type="text/plain")
    assert resp.status_code == 400
    assert resp.json()["error"] == "Only PDF files are accepted"


def test_non_pdf_content_with_pdf_content_type_returns_400(client):
    resp = _upload(client, content=_NOT_PDF, content_type="application/pdf")
    assert resp.status_code == 400


def test_non_pdf_content_and_wrong_content_type_returns_400(client):
    resp = _upload(client, content=_NOT_PDF, content_type="text/plain")
    assert resp.status_code == 400


# ── Missing required fields ───────────────────────────────────────────────────


def test_missing_tenant_id_returns_422(client):
    assert _upload(client, tenant_id=None).status_code == 422


def test_missing_file_returns_422(client):
    resp = client.post("/api/v1/docs/upload", data={"tenant_id": "acme"})
    assert resp.status_code == 422


# ── SHA-256 fingerprinting ────────────────────────────────────────────────────


def test_file_hash_is_valid_sha256_hex(client):
    file_hash = _upload(client).json()["file_hash"]
    assert len(file_hash) == 64
    assert all(c in "0123456789abcdef" for c in file_hash)


def test_file_hash_matches_sha256_of_content(client):
    expected = hashlib.sha256(_FAKE_PDF).hexdigest()
    assert _upload(client).json()["file_hash"] == expected


def test_duplicate_upload_same_hash(client):
    a = _upload(client, content=_FAKE_PDF).json()["file_hash"]
    b = _upload(client, content=_FAKE_PDF).json()["file_hash"]
    assert a == b


def test_different_files_different_hashes(client):
    a = _upload(client, content=_FAKE_PDF).json()["file_hash"]
    b = _upload(client, content=b"%PDF-1.4 different content").json()["file_hash"]
    assert a != b


# ── LocalStorage unit tests ───────────────────────────────────────────────────


def test_local_storage_save_writes_file(tmp_path):
    storage = LocalStorage(base_dir=str(tmp_path))
    file_hash = hashlib.sha256(_FAKE_PDF).hexdigest()
    locator = storage.save(file_hash, _FAKE_PDF)
    assert Path(locator).read_bytes() == _FAKE_PDF


def test_local_storage_save_is_idempotent(tmp_path):
    storage = LocalStorage(base_dir=str(tmp_path))
    file_hash = hashlib.sha256(_FAKE_PDF).hexdigest()
    loc1 = storage.save(file_hash, _FAKE_PDF)
    loc2 = storage.save(file_hash, _FAKE_PDF)  # second call must not raise
    assert loc1 == loc2


def test_local_storage_exists(tmp_path):
    storage = LocalStorage(base_dir=str(tmp_path))
    file_hash = hashlib.sha256(_FAKE_PDF).hexdigest()
    assert not storage.exists(file_hash)
    storage.save(file_hash, _FAKE_PDF)
    assert storage.exists(file_hash)


# ── Storage abstraction: swap backend via dependency_overrides ────────────────


def test_storage_backend_can_be_overridden():
    """The route must use whatever backend is injected — not hardcode LocalStorage."""
    mock_storage = MagicMock(spec=StorageBackend)
    mock_storage.save.return_value = "mock://fake-locator"
    mock_celery = _mock_celery_app()

    app.dependency_overrides[get_local_storage] = lambda: mock_storage
    app.dependency_overrides[get_vector_store] = lambda: _mock_vector_store(exists=False)
    app.dependency_overrides[get_celery_app] = lambda: mock_celery
    try:
        client = TestClient(app)
        resp = _upload(client)
        assert resp.status_code == 202
        mock_storage.save.assert_called_once()
        mock_celery.send_task.assert_called_once_with(
            "enterprise_rag.ingest_document_pipeline",
            args=["mock://fake-locator", "acme", resp.json()["file_hash"]],
        )
    finally:
        app.dependency_overrides.clear()
