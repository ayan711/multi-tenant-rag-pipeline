"""
Tests for Task 5.2 — File Upload Endpoint.
Tests for Task 5.3 — SHA-256 Fingerprinting.
Tests for storage abstraction layer.

Covers:
  - Happy path: valid PDF → 200 with filename, tenant_id, size_bytes, file_hash, locator
  - Wrong content_type → 400
  - Non-PDF content (bad magic bytes) → 400
  - Missing tenant_id / file → 422
  - file_hash is a valid 64-char hex SHA-256 digest
  - Same file → same hash; different files → different hashes
  - Storage backend can be swapped via dependency_overrides (abstraction test)
"""

import hashlib
import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.storage.base import StorageBackend
from app.storage.local import LocalStorage, get_local_storage

# Minimal valid-looking PDF: starts with the %PDF magic bytes.
_FAKE_PDF = b"%PDF-1.4 fake pdf content"
_NOT_PDF = b"this is plain text, not a pdf"


@pytest.fixture
def client():
    return TestClient(app)


def _upload(client, *, content=_FAKE_PDF, content_type="application/pdf", tenant_id="acme"):
    """Helper: POST to /api/v1/docs/upload with sensible defaults."""
    files = {"file": ("doc.pdf", io.BytesIO(content), content_type)}
    data = {"tenant_id": tenant_id} if tenant_id is not None else {}
    return client.post("/api/v1/docs/upload", files=files, data=data)


# ── Happy path ────────────────────────────────────────────────────────────────


def test_upload_valid_pdf_returns_200(client):
    resp = _upload(client)
    assert resp.status_code == 200


def test_upload_response_contains_expected_fields(client):
    body = _upload(client, tenant_id="acme").json()
    assert body["filename"] == "doc.pdf"
    assert body["tenant_id"] == "acme"
    assert body["size_bytes"] == len(_FAKE_PDF)
    assert "file_hash" in body
    assert "locator" in body


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

    app.dependency_overrides[get_local_storage] = lambda: mock_storage
    try:
        client = TestClient(app)
        resp = _upload(client)
        assert resp.status_code == 200
        assert resp.json()["locator"] == "mock://fake-locator"
        mock_storage.save.assert_called_once()
    finally:
        app.dependency_overrides.clear()


def test_locator_in_response_matches_storage_output(tmp_path):
    """locator in the response is exactly what storage.save() returned."""
    storage = LocalStorage(base_dir=str(tmp_path))
    app.dependency_overrides[get_local_storage] = lambda: storage
    try:
        client = TestClient(app)
        body = _upload(client).json()
        expected_path = str(tmp_path / f"{body['file_hash']}.pdf")
        assert body["locator"] == expected_path
    finally:
        app.dependency_overrides.clear()
