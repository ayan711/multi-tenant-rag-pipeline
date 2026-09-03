"""
Tests for Tasks 4.1–4.4 — Celery Application Setup, Singleton Worker Resources,
Ingestion Pipeline Task, and Worker Smoke Test.

Unit tests (Tasks 4.1–4.3): no Redis, ChromaDB, or disk-based PDFs required.
Smoke tests (Task 4.4): use task_always_eager for dispatch; Redis ping is an
integration test that skips automatically when Docker is not running.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
import app.worker as worker_module
from celery import Celery

from app.config import settings
from app.embedder import LocalEmbedder
from app.vector_db import VectorStoreManager
from app.worker import celery_app, ingest_document_pipeline


class TestCeleryAppObject:
    def test_is_celery_instance(self):
        assert isinstance(celery_app, Celery)

    def test_app_name(self):
        assert celery_app.main == "enterprise_rag"

    def test_broker_is_redis(self):
        # conf.broker_url is set from settings.redis_url
        assert celery_app.conf.broker_url.startswith("redis://")

    def test_backend_is_redis(self):
        assert celery_app.conf.result_backend.startswith("redis://")

    def test_broker_and_backend_match(self):
        # Both should point to the same Redis URL
        assert celery_app.conf.broker_url == celery_app.conf.result_backend


class TestCeleryConfig:
    def test_json_serializer(self):
        assert celery_app.conf.task_serializer == "json"

    def test_json_result_serializer(self):
        assert celery_app.conf.result_serializer == "json"

    def test_accepts_only_json(self):
        assert "json" in celery_app.conf.accept_content

    def test_track_started_enabled(self):
        # Needed so the status-polling endpoint can see STARTED state
        assert celery_app.conf.task_track_started is True

    def test_utc_enabled(self):
        assert celery_app.conf.enable_utc is True

    def test_timezone_utc(self):
        assert celery_app.conf.timezone == "UTC"

    def test_result_expires_one_hour(self):
        assert celery_app.conf.result_expires == 3600


class TestWorkerSingletons:
    """Task 4.2 — Singleton Worker Resources.

    LocalEmbedder and VectorStoreManager are instantiated at module level so
    model weights and the ChromaDB connection are initialised once per worker
    process, not once per task invocation.
    """

    def test_embedder_is_local_embedder(self):
        assert isinstance(worker_module.embedder, LocalEmbedder)

    def test_vector_store_is_vector_store_manager(self):
        assert isinstance(worker_module.vector_store, VectorStoreManager)

    def test_singletons_share_identity_across_imports(self):
        # Python's module cache guarantees repeated imports return the same object.
        import app.worker as second_import
        assert second_import.embedder is worker_module.embedder
        assert second_import.vector_store is worker_module.vector_store


class TestIngestPipeline:
    """Task 4.3 — Ingestion Pipeline Task.

    The task is called directly (not via .delay()) so no broker is needed.
    All I/O — parser, embedder, vector_store, and os.unlink — is mocked.
    """

    @pytest.fixture
    def mocks(self, tmp_path):
        """Create a real temp file and wire up all mocked collaborators."""
        dummy_pdf = tmp_path / "abc123.pdf"
        dummy_pdf.write_bytes(b"fake pdf content")

        mock_chunks = [{"text": "hello world", "page_number": 1, "chunk_index": 0}]
        mock_embeddings = [[0.1] * 384]

        mock_parser = MagicMock()
        mock_parser.extract_text_per_page.return_value = [{"page_number": 1, "text": "hello world"}]
        mock_parser.chunk_pages.return_value = mock_chunks

        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = mock_embeddings

        mock_vector_store = MagicMock()
        mock_vector_store.document_exists.return_value = False  # not yet indexed
        mock_vector_store.insert_chunks.return_value = 1

        with patch("app.worker.DocumentParser", return_value=mock_parser), \
             patch("app.worker.embedder", mock_embedder), \
             patch("app.worker.vector_store", mock_vector_store):
            yield {
                "path": str(dummy_pdf),
                "parser": mock_parser,
                "embedder": mock_embedder,
                "vector_store": mock_vector_store,
                "chunks": mock_chunks,
                "embeddings": mock_embeddings,
            }

    def test_task_is_registered(self):
        assert "enterprise_rag.ingest_document_pipeline" in celery_app.tasks

    def test_calls_extract_text_per_page(self, mocks):
        ingest_document_pipeline(mocks["path"], "tenant_a", "abc123")
        mocks["parser"].extract_text_per_page.assert_called_once_with(mocks["path"])

    def test_calls_chunk_pages_with_extracted_pages(self, mocks):
        ingest_document_pipeline(mocks["path"], "tenant_a", "abc123")
        mocks["parser"].chunk_pages.assert_called_once()

    def test_calls_embed_batch_with_chunk_texts(self, mocks):
        ingest_document_pipeline(mocks["path"], "tenant_a", "abc123")
        mocks["embedder"].embed_batch.assert_called_once_with(["hello world"])

    def test_calls_insert_chunks_with_correct_args(self, mocks):
        ingest_document_pipeline(mocks["path"], "tenant_a", "abc123")
        mocks["vector_store"].insert_chunks.assert_called_once_with(
            mocks["chunks"], mocks["embeddings"], "tenant_a", "abc123"
        )

    def test_returns_summary_dict(self, mocks):
        result = ingest_document_pipeline(mocks["path"], "tenant_a", "abc123")
        assert result == {"status": "ingested", "file_hash": "abc123", "chunks_inserted": 1, "tenant_id": "tenant_a"}

    def test_deletes_temp_file_on_success(self, mocks):
        assert os.path.exists(mocks["path"])
        ingest_document_pipeline(mocks["path"], "tenant_a", "abc123")
        assert not os.path.exists(mocks["path"])

    def test_deletes_temp_file_on_failure(self, tmp_path):
        """Temp file must be cleaned up even if the pipeline raises."""
        dummy_pdf = tmp_path / "fail.pdf"
        dummy_pdf.write_bytes(b"fake")

        mock_parser = MagicMock()
        mock_parser.extract_text_per_page.side_effect = RuntimeError("parse failed")

        mock_vs = MagicMock()
        mock_vs.document_exists.return_value = False  # let the pipeline proceed so it can raise

        with patch("app.worker.DocumentParser", return_value=mock_parser), \
             patch("app.worker.vector_store", mock_vs):
            with pytest.raises(RuntimeError, match="parse failed"):
                ingest_document_pipeline(str(dummy_pdf), "tenant_a", "abc123")

        assert not dummy_pdf.exists()

    def test_blank_pdf_returns_zero_chunks(self, tmp_path):
        """A PDF with no extractable text inserts nothing and returns count=0."""
        dummy_pdf = tmp_path / "blank.pdf"
        dummy_pdf.write_bytes(b"fake")

        mock_parser = MagicMock()
        mock_parser.extract_text_per_page.return_value = []
        mock_parser.chunk_pages.return_value = []

        mock_vector_store = MagicMock()
        mock_vector_store.document_exists.return_value = False  # not yet indexed

        with patch("app.worker.DocumentParser", return_value=mock_parser), \
             patch("app.worker.vector_store", mock_vector_store):
            result = ingest_document_pipeline(str(dummy_pdf), "tenant_a", "abc123")

        assert result["chunks_inserted"] == 0
        mock_vector_store.insert_chunks.assert_not_called()


# ---------------------------------------------------------------------------
# Task 4.4 — Worker Smoke Test
# ---------------------------------------------------------------------------

class TestWorkerSmokeTest:
    """Verify the worker can dispatch tasks and that Redis is reachable.

    Two layers of coverage:
      - Eager-mode tests: task_always_eager=True makes .delay() execute
        synchronously in the calling process. No broker required. This tests
        that the full dispatch path (.delay() → task → AsyncResult) works.
      - Redis ping test (integration): attempts a live PING to the configured
        Redis URL. Skips automatically when Docker is not running so it never
        blocks CI in environments without infrastructure.
    """

    @pytest.fixture(autouse=True)
    def eager_mode(self):
        """Run tasks inline when .delay() is called — no broker needed.

        task_eager_propagates is intentionally left False (the default) so that
        task exceptions are captured in the AsyncResult rather than re-raised at
        .delay() time. This lets test_failed_task_state_is_failure inspect
        result.state before calling result.get().
        """
        celery_app.conf.update(task_always_eager=True)
        yield
        celery_app.conf.update(task_always_eager=False)

    @pytest.fixture
    def dispatch_mocks(self, tmp_path):
        """A real temp file + mocked collaborators ready for a .delay() call."""
        dummy_pdf = tmp_path / "smoke.pdf"
        dummy_pdf.write_bytes(b"fake pdf content")

        mock_chunks = [{"text": "smoke test", "page_number": 1, "chunk_index": 0}]

        mock_parser = MagicMock()
        mock_parser.extract_text_per_page.return_value = [{"page_number": 1, "text": "smoke test"}]
        mock_parser.chunk_pages.return_value = mock_chunks

        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = [[0.1] * 384]

        mock_vector_store = MagicMock()
        mock_vector_store.document_exists.return_value = False  # not yet indexed
        mock_vector_store.insert_chunks.return_value = 1

        with patch("app.worker.DocumentParser", return_value=mock_parser), \
             patch("app.worker.embedder", mock_embedder), \
             patch("app.worker.vector_store", mock_vector_store):
            yield {"path": str(dummy_pdf)}

    # --- dispatch mechanics ---

    def test_delay_returns_async_result(self, dispatch_mocks):
        """Calling .delay() returns an AsyncResult, not a bare value."""
        from celery.result import AsyncResult
        result = ingest_document_pipeline.delay(
            dispatch_mocks["path"], "tenant_smoke", "hash_smoke"
        )
        assert isinstance(result, AsyncResult)

    def test_eager_task_completes_successfully(self, dispatch_mocks):
        """In eager mode the task executes and result.get() returns the summary dict."""
        result = ingest_document_pipeline.delay(
            dispatch_mocks["path"], "tenant_smoke", "hash_smoke"
        )
        assert result.successful()
        assert result.get() == {
            "status": "ingested",
            "file_hash": "hash_smoke",
            "chunks_inserted": 1,
            "tenant_id": "tenant_smoke",
        }

    def test_eager_task_state_is_success(self, dispatch_mocks):
        """Task state is SUCCESS after eager execution — not PENDING or FAILURE."""
        result = ingest_document_pipeline.delay(
            dispatch_mocks["path"], "tenant_smoke", "hash_smoke"
        )
        assert result.state == "SUCCESS"

    def test_failed_task_state_is_failure(self, tmp_path):
        """A task that raises records FAILURE state and re-raises via result.get()."""
        dummy_pdf = tmp_path / "bad.pdf"
        dummy_pdf.write_bytes(b"fake")

        mock_parser = MagicMock()
        mock_parser.extract_text_per_page.side_effect = RuntimeError("boom")

        mock_vs = MagicMock()
        mock_vs.document_exists.return_value = False  # let the pipeline proceed so it can raise

        with patch("app.worker.DocumentParser", return_value=mock_parser), \
             patch("app.worker.vector_store", mock_vs):
            result = ingest_document_pipeline.delay(str(dummy_pdf), "tenant_smoke", "hash_bad")

        assert result.state == "FAILURE"
        with pytest.raises(RuntimeError, match="boom"):
            result.get()

    # --- Redis connectivity (integration, skips when Docker is down) ---

    @pytest.mark.integration
    def test_redis_broker_is_reachable(self):
        """Redis accepts a PING at the configured broker URL."""
        import redis as redis_lib
        try:
            client = redis_lib.from_url(settings.redis_url, socket_connect_timeout=2)
            assert client.ping() is True
        except (redis_lib.ConnectionError, redis_lib.TimeoutError) as exc:
            pytest.skip(f"Redis not available (start Docker): {exc}")


# ---------------------------------------------------------------------------
# Task 4.5 — Idempotent Ingestion Guard
# ---------------------------------------------------------------------------

class TestIdempotencyGuard:
    """Verify that ingest_document_pipeline skips re-processing a document that
    is already indexed in ChromaDB for the same tenant.

    The guard lives in two places:
      1. VectorStoreManager.document_exists() — the DB-layer primitive.
      2. ingest_document_pipeline() — calls document_exists() before doing any
         work so re-submitted jobs are a no-op even if FastAPI's pre-check is
         bypassed (e.g. two concurrent uploads of the same file).
    """

    @pytest.fixture
    def pipeline_mocks(self, tmp_path):
        """Temp file + collaborators for calling ingest_document_pipeline directly."""
        dummy_pdf = tmp_path / "dup.pdf"
        dummy_pdf.write_bytes(b"fake pdf content")

        mock_parser = MagicMock()
        mock_parser.extract_text_per_page.return_value = [{"page_number": 1, "text": "hello"}]
        mock_parser.chunk_pages.return_value = [{"text": "hello", "page_number": 1, "chunk_index": 0}]

        mock_embedder = MagicMock()
        mock_embedder.embed_batch.return_value = [[0.1] * 384]

        return {"path": str(dummy_pdf), "parser": mock_parser, "embedder": mock_embedder}

    # --- already-exists path ---

    def test_returns_already_exists_status_when_document_found(self, pipeline_mocks):
        """Task returns status='already_exists' when document_exists() is True."""
        mock_vs = MagicMock()
        mock_vs.document_exists.return_value = True

        with patch("app.worker.DocumentParser", return_value=pipeline_mocks["parser"]), \
             patch("app.worker.embedder", pipeline_mocks["embedder"]), \
             patch("app.worker.vector_store", mock_vs):
            result = ingest_document_pipeline(pipeline_mocks["path"], "tenant_a", "hash_dup")

        assert result["status"] == "already_exists"

    def test_returns_zero_chunks_inserted_when_already_exists(self, pipeline_mocks):
        mock_vs = MagicMock()
        mock_vs.document_exists.return_value = True

        with patch("app.worker.vector_store", mock_vs):
            result = ingest_document_pipeline(pipeline_mocks["path"], "tenant_a", "hash_dup")

        assert result["chunks_inserted"] == 0

    def test_skips_embedding_when_already_exists(self, pipeline_mocks):
        """No work should be done if the document is already indexed."""
        mock_vs = MagicMock()
        mock_vs.document_exists.return_value = True

        with patch("app.worker.embedder", pipeline_mocks["embedder"]), \
             patch("app.worker.vector_store", mock_vs):
            ingest_document_pipeline(pipeline_mocks["path"], "tenant_a", "hash_dup")

        pipeline_mocks["embedder"].embed_batch.assert_not_called()

    def test_skips_insert_when_already_exists(self, pipeline_mocks):
        mock_vs = MagicMock()
        mock_vs.document_exists.return_value = True

        with patch("app.worker.vector_store", mock_vs):
            ingest_document_pipeline(pipeline_mocks["path"], "tenant_a", "hash_dup")

        mock_vs.insert_chunks.assert_not_called()

    def test_deletes_temp_file_when_already_exists(self, pipeline_mocks):
        """Temp file is cleaned up on the early-return path too."""
        mock_vs = MagicMock()
        mock_vs.document_exists.return_value = True

        with patch("app.worker.vector_store", mock_vs):
            ingest_document_pipeline(pipeline_mocks["path"], "tenant_a", "hash_dup")

        assert not os.path.exists(pipeline_mocks["path"])

    def test_preserves_file_hash_and_tenant_in_result(self, pipeline_mocks):
        mock_vs = MagicMock()
        mock_vs.document_exists.return_value = True

        with patch("app.worker.vector_store", mock_vs):
            result = ingest_document_pipeline(pipeline_mocks["path"], "tenant_x", "hash_xyz")

        assert result["file_hash"] == "hash_xyz"
        assert result["tenant_id"] == "tenant_x"

    # --- normal path still works ---

    def test_proceeds_normally_when_document_does_not_exist(self, pipeline_mocks):
        """When document_exists() is False the task runs the full pipeline."""
        mock_vs = MagicMock()
        mock_vs.document_exists.return_value = False
        mock_vs.insert_chunks.return_value = 1

        with patch("app.worker.DocumentParser", return_value=pipeline_mocks["parser"]), \
             patch("app.worker.embedder", pipeline_mocks["embedder"]), \
             patch("app.worker.vector_store", mock_vs):
            result = ingest_document_pipeline(pipeline_mocks["path"], "tenant_a", "hash_new")

        assert result["status"] == "ingested"
        assert result["chunks_inserted"] == 1
        mock_vs.insert_chunks.assert_called_once()

    # --- document_exists is called with the right arguments ---

    def test_document_exists_called_with_file_hash_and_tenant(self, pipeline_mocks):
        mock_vs = MagicMock()
        mock_vs.document_exists.return_value = True

        with patch("app.worker.vector_store", mock_vs):
            ingest_document_pipeline(pipeline_mocks["path"], "tenant_q", "hash_q")

        mock_vs.document_exists.assert_called_once_with("hash_q", "tenant_q")
