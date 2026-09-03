"""
Tests for Task 5.9 — Task Status Endpoint.

Two layers of coverage:
  - Unit tests: AsyncResult is patched, so each Celery state (PENDING, STARTED,
    SUCCESS, FAILURE) can be exercised without a real task ever running.
  - Integration tests: a real task_always_eager dispatch through the actual
    ingest_document_pipeline task, then a real AsyncResult lookup through the
    endpoint — proves the whole path (.delay() -> Redis -> AsyncResult -> API)
    actually wires together, not just each half in isolation.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.celery_app import celery_app
from app.dependencies import get_celery_app
from app.main import app
from app.worker import ingest_document_pipeline

# ── Unit tests: AsyncResult mocked ───────────────────────────────────────────


@pytest.fixture
def client():
    """TestClient with get_celery_app swapped for a bare mock.

    The mock Celery app is never actually queried — AsyncResult itself is
    patched in each test — but overriding the dependency proves the route
    reads the app via Depends() rather than importing the module-level
    singleton directly.
    """
    app.dependency_overrides[get_celery_app] = lambda: MagicMock()
    yield TestClient(app)
    app.dependency_overrides.clear()


def _mock_async_result(state, result=None):
    m = MagicMock()
    m.state = state
    m.result = result
    return m


def test_pending_status(client):
    with patch("app.routers.tasks.AsyncResult", return_value=_mock_async_result("PENDING")):
        resp = client.get("/api/v1/tasks/some-task-id")
    assert resp.status_code == 200
    body = resp.json()
    assert body["task_id"] == "some-task-id"
    assert body["status"] == "PENDING"
    assert body["result"] is None
    assert body["error"] is None


def test_started_status(client):
    with patch("app.routers.tasks.AsyncResult", return_value=_mock_async_result("STARTED")):
        resp = client.get("/api/v1/tasks/some-task-id")
    assert resp.json()["status"] == "STARTED"


def test_success_status_includes_result(client):
    fake_result = {"status": "ingested", "file_hash": "abc123", "chunks_inserted": 5}
    with patch(
        "app.routers.tasks.AsyncResult",
        return_value=_mock_async_result("SUCCESS", result=fake_result),
    ):
        resp = client.get("/api/v1/tasks/some-task-id")
    body = resp.json()
    assert body["status"] == "SUCCESS"
    assert body["result"] == fake_result
    assert body["error"] is None


def test_failure_status_includes_error_as_string(client):
    with patch(
        "app.routers.tasks.AsyncResult",
        return_value=_mock_async_result("FAILURE", result=RuntimeError("boom")),
    ):
        resp = client.get("/api/v1/tasks/some-task-id")
    body = resp.json()
    assert body["status"] == "FAILURE"
    assert body["result"] is None
    assert body["error"] == "boom"


def test_async_result_is_constructed_with_task_id_and_injected_app():
    mock_app = MagicMock()
    app.dependency_overrides[get_celery_app] = lambda: mock_app
    try:
        with patch(
            "app.routers.tasks.AsyncResult", return_value=_mock_async_result("PENDING")
        ) as mock_async_result:
            TestClient(app).get("/api/v1/tasks/task-xyz")
        mock_async_result.assert_called_once_with("task-xyz", app=mock_app)
    finally:
        app.dependency_overrides.clear()


# ── Integration tests: real Celery app, eager dispatch ───────────────────────


class TestRealDispatchAndLookup:
    """Runs the actual task and reads its status back through the real endpoint."""

    @pytest.fixture(autouse=True)
    def eager_mode(self):
        # task_eager_propagates stays False (the default): failures land in
        # the AsyncResult as FAILURE instead of raising at .delay() time.
        # (task_store_eager_result is on permanently in app/celery_app.py —
        # see the comment there for why it can't just be toggled here.)
        celery_app.conf.update(task_always_eager=True)
        yield
        celery_app.conf.update(task_always_eager=False)

    @pytest.fixture
    def real_client(self):
        app.dependency_overrides[get_celery_app] = lambda: celery_app
        yield TestClient(app)
        app.dependency_overrides.clear()

    def test_unknown_task_id_reads_back_as_pending(self, real_client):
        # Never dispatched — Celery has no distinction between "unknown" and
        # "queued but not started yet", so both read back as PENDING.
        resp = real_client.get("/api/v1/tasks/never-dispatched-id")
        assert resp.status_code == 200
        assert resp.json()["status"] == "PENDING"

    def test_successful_task_reads_back_as_success_with_result(self, real_client, tmp_path):
        dummy_pdf = tmp_path / "task_status_smoke.pdf"
        dummy_pdf.write_bytes(b"fake pdf content")

        mock_chunks = [{"text": "smoke test", "page_number": 1, "chunk_index": 0}]
        mock_parser = MagicMock()
        mock_parser.extract_text_per_page.return_value = [{"page_number": 1, "text": "smoke test"}]
        mock_parser.chunk_pages.return_value = mock_chunks

        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = [[0.1] * 384]

        mock_vector_store = MagicMock()
        mock_vector_store.document_exists.return_value = False
        mock_vector_store.insert_chunks.return_value = 1

        with patch("app.worker.DocumentParser", return_value=mock_parser), \
             patch("app.worker.embedder", mock_embedder), \
             patch("app.worker.vector_store", mock_vector_store):
            async_result = ingest_document_pipeline.delay(
                str(dummy_pdf), "tenant_status_smoke", "hash_status_smoke"
            )

        resp = real_client.get(f"/api/v1/tasks/{async_result.id}")
        body = resp.json()
        assert resp.status_code == 200
        assert body["status"] == "SUCCESS"
        assert body["result"] == {
            "status": "ingested",
            "file_hash": "hash_status_smoke",
            "chunks_inserted": 1,
            "tenant_id": "tenant_status_smoke",
        }
        assert body["error"] is None

    def test_failed_task_reads_back_as_failure_with_error(self, real_client, tmp_path):
        dummy_pdf = tmp_path / "task_status_bad.pdf"
        dummy_pdf.write_bytes(b"fake")

        mock_parser = MagicMock()
        mock_parser.extract_text_per_page.side_effect = RuntimeError("boom")

        mock_vector_store = MagicMock()
        mock_vector_store.document_exists.return_value = False

        with patch("app.worker.DocumentParser", return_value=mock_parser), \
             patch("app.worker.vector_store", mock_vector_store):
            async_result = ingest_document_pipeline.delay(
                str(dummy_pdf), "tenant_status_smoke", "hash_status_bad"
            )

        resp = real_client.get(f"/api/v1/tasks/{async_result.id}")
        body = resp.json()
        assert resp.status_code == 200
        assert body["status"] == "FAILURE"
        assert body["result"] is None
        assert "boom" in body["error"]
