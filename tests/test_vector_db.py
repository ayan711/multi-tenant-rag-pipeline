"""
Tests for app/vector_db.py — Tasks 3.1, 3.2, 3.3: Client Setup, Collection Init,
and Metadata Injection.

All tests mock chromadb.HttpClient so the suite runs without a live
ChromaDB server. We verify the behaviour our code is responsible for:
correct construction arguments, collection initialisation, heartbeat delegation,
and the structure of the collection.add() call made by insert_chunks().
"""

from unittest.mock import MagicMock, patch

import pytest

from app.config import settings
from app.vector_db import VectorStoreManager


@pytest.fixture
def mock_http_client():
    """Patch chromadb.HttpClient at the import site in app.vector_db."""
    with patch("app.vector_db.chromadb.HttpClient") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        yield mock_cls, mock_instance


class TestVectorStoreManagerInit:

    def test_default_args_come_from_settings(self, mock_http_client):
        mock_cls, _ = mock_http_client
        VectorStoreManager()
        mock_cls.assert_called_once_with(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )

    def test_custom_host_and_port_override_defaults(self, mock_http_client):
        mock_cls, _ = mock_http_client
        VectorStoreManager(host="custom-host", port=9999)
        mock_cls.assert_called_once_with(host="custom-host", port=9999)

    def test_client_stored_on_instance(self, mock_http_client):
        _, mock_instance = mock_http_client
        manager = VectorStoreManager()
        assert manager._client is mock_instance

    def test_two_instances_each_create_their_own_client(self, mock_http_client):
        mock_cls, _ = mock_http_client
        VectorStoreManager()
        VectorStoreManager()
        assert mock_cls.call_count == 2


class TestCollectionInit:

    def test_get_or_create_called_with_default_collection_name(self, mock_http_client):
        _, mock_instance = mock_http_client
        VectorStoreManager()
        mock_instance.get_or_create_collection.assert_called_once_with(
            name=settings.chroma_collection
        )

    def test_get_or_create_called_with_custom_collection_name(self, mock_http_client):
        _, mock_instance = mock_http_client
        VectorStoreManager(collection_name="custom_collection")
        mock_instance.get_or_create_collection.assert_called_once_with(
            name="custom_collection"
        )

    def test_collection_stored_on_instance(self, mock_http_client):
        _, mock_instance = mock_http_client
        mock_collection = MagicMock()
        mock_instance.get_or_create_collection.return_value = mock_collection
        manager = VectorStoreManager()
        assert manager._collection is mock_collection

    def test_get_or_create_called_exactly_once_per_instance(self, mock_http_client):
        _, mock_instance = mock_http_client
        VectorStoreManager()
        assert mock_instance.get_or_create_collection.call_count == 1


class TestInsertChunks:

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_chunks(self, n: int = 3) -> list[dict]:
        return [
            {"text": f"chunk text {i}", "page_number": 1, "chunk_index": i}
            for i in range(n)
        ]

    def _make_embeddings(self, n: int = 3) -> list[list[float]]:
        return [[float(i)] * 384 for i in range(n)]

    # ── Composite ID construction ─────────────────────────────────────────────

    def test_ids_use_composite_key(self, mock_http_client):
        _, mock_instance = mock_http_client
        mock_collection = MagicMock()
        mock_instance.get_or_create_collection.return_value = mock_collection

        chunks = self._make_chunks(3)
        embeddings = self._make_embeddings(3)
        VectorStoreManager().insert_chunks(chunks, embeddings, "tenant_a", "abc123")

        call_kwargs = mock_collection.add.call_args.kwargs
        assert call_kwargs["ids"] == [
            "abc123_1_0",
            "abc123_1_1",
            "abc123_1_2",
        ]

    def test_id_uses_file_hash_from_argument(self, mock_http_client):
        _, mock_instance = mock_http_client
        mock_collection = MagicMock()
        mock_instance.get_or_create_collection.return_value = mock_collection

        chunks = [{"text": "x", "page_number": 2, "chunk_index": 0}]
        VectorStoreManager().insert_chunks(chunks, [[0.0] * 384], "t", "deadbeef")

        ids = mock_collection.add.call_args.kwargs["ids"]
        assert ids[0].startswith("deadbeef_")

    # ── Metadata payload ──────────────────────────────────────────────────────

    def test_metadata_contains_required_fields(self, mock_http_client):
        _, mock_instance = mock_http_client
        mock_collection = MagicMock()
        mock_instance.get_or_create_collection.return_value = mock_collection

        chunks = [{"text": "hello", "page_number": 5, "chunk_index": 2}]
        VectorStoreManager().insert_chunks(chunks, [[0.1] * 384], "tenant_x", "hash99")

        metadatas = mock_collection.add.call_args.kwargs["metadatas"]
        assert metadatas == [
            {"tenant_id": "tenant_x", "file_hash": "hash99", "page_number": 5}
        ]

    def test_metadata_tenant_id_matches_argument(self, mock_http_client):
        _, mock_instance = mock_http_client
        mock_collection = MagicMock()
        mock_instance.get_or_create_collection.return_value = mock_collection

        chunks = self._make_chunks(2)
        VectorStoreManager().insert_chunks(chunks, self._make_embeddings(2), "tenant_z", "h")

        metadatas = mock_collection.add.call_args.kwargs["metadatas"]
        assert all(m["tenant_id"] == "tenant_z" for m in metadatas)

    def test_metadata_page_number_per_chunk(self, mock_http_client):
        _, mock_instance = mock_http_client
        mock_collection = MagicMock()
        mock_instance.get_or_create_collection.return_value = mock_collection

        chunks = [
            {"text": "a", "page_number": 1, "chunk_index": 0},
            {"text": "b", "page_number": 3, "chunk_index": 1},
        ]
        VectorStoreManager().insert_chunks(chunks, self._make_embeddings(2), "t", "h")

        metadatas = mock_collection.add.call_args.kwargs["metadatas"]
        assert metadatas[0]["page_number"] == 1
        assert metadatas[1]["page_number"] == 3

    # ── Documents & embeddings pass-through ───────────────────────────────────

    def test_documents_are_chunk_texts(self, mock_http_client):
        _, mock_instance = mock_http_client
        mock_collection = MagicMock()
        mock_instance.get_or_create_collection.return_value = mock_collection

        chunks = [
            {"text": "first", "page_number": 1, "chunk_index": 0},
            {"text": "second", "page_number": 1, "chunk_index": 1},
        ]
        VectorStoreManager().insert_chunks(chunks, self._make_embeddings(2), "t", "h")

        docs = mock_collection.add.call_args.kwargs["documents"]
        assert docs == ["first", "second"]

    def test_embeddings_passed_through_unchanged(self, mock_http_client):
        _, mock_instance = mock_http_client
        mock_collection = MagicMock()
        mock_instance.get_or_create_collection.return_value = mock_collection

        chunks = self._make_chunks(2)
        vecs = [[0.1] * 384, [0.9] * 384]
        VectorStoreManager().insert_chunks(chunks, vecs, "t", "h")

        assert mock_collection.add.call_args.kwargs["embeddings"] is vecs

    # ── Return value ──────────────────────────────────────────────────────────

    def test_returns_count_of_inserted_chunks(self, mock_http_client):
        _, mock_instance = mock_http_client
        mock_collection = MagicMock()
        mock_instance.get_or_create_collection.return_value = mock_collection

        chunks = self._make_chunks(5)
        result = VectorStoreManager().insert_chunks(
            chunks, self._make_embeddings(5), "t", "h"
        )
        assert result == 5

    def test_single_chunk_returns_one(self, mock_http_client):
        _, mock_instance = mock_http_client
        mock_collection = MagicMock()
        mock_instance.get_or_create_collection.return_value = mock_collection

        chunks = [{"text": "only", "page_number": 1, "chunk_index": 0}]
        result = VectorStoreManager().insert_chunks(chunks, [[0.0] * 384], "t", "h")
        assert result == 1

    # ── collection.add() called exactly once ──────────────────────────────────

    def test_collection_add_called_once(self, mock_http_client):
        _, mock_instance = mock_http_client
        mock_collection = MagicMock()
        mock_instance.get_or_create_collection.return_value = mock_collection

        VectorStoreManager().insert_chunks(
            self._make_chunks(4), self._make_embeddings(4), "t", "h"
        )
        mock_collection.add.assert_called_once()


class TestSearchTenantContext:

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_query_result(
        self, texts: list[str], page_numbers: list[int], distances: list[float]
    ) -> dict:
        """Build the dict shape that collection.query() actually returns."""
        return {
            "ids": [[f"id_{i}" for i in range(len(texts))]],
            "documents": [texts],
            "metadatas": [
                [{"tenant_id": "t", "file_hash": "h", "page_number": p}
                 for p in page_numbers]
            ],
            "distances": [distances],
        }

    def _manager_with_collection(self, mock_http_client, query_return):
        """Return a VectorStoreManager whose collection.query() yields query_return."""
        _, mock_instance = mock_http_client
        mock_collection = MagicMock()
        mock_instance.get_or_create_collection.return_value = mock_collection
        mock_collection.query.return_value = query_return
        return VectorStoreManager(), mock_collection

    # ── where= filter ────────────────────────────────────────────────────────

    def test_where_filter_always_includes_tenant_id(self, mock_http_client):
        result = self._make_query_result(["a"], [1], [0.1])
        manager, mock_collection = self._manager_with_collection(mock_http_client, result)
        manager.search_tenant_context([0.0] * 384, "tenant_a")
        assert mock_collection.query.call_args.kwargs["where"] == {"tenant_id": "tenant_a"}

    def test_different_tenants_produce_different_where_filters(self, mock_http_client):
        result = self._make_query_result(["x"], [1], [0.1])
        manager, mock_collection = self._manager_with_collection(mock_http_client, result)
        manager.search_tenant_context([0.0] * 384, "tenant_a")
        manager.search_tenant_context([0.0] * 384, "tenant_b")
        calls = mock_collection.query.call_args_list
        assert calls[0].kwargs["where"] == {"tenant_id": "tenant_a"}
        assert calls[1].kwargs["where"] == {"tenant_id": "tenant_b"}

    # ── query_embeddings ──────────────────────────────────────────────────────

    def test_query_embedding_is_wrapped_in_list(self, mock_http_client):
        result = self._make_query_result(["a"], [1], [0.1])
        manager, mock_collection = self._manager_with_collection(mock_http_client, result)
        vec = [0.5] * 384
        manager.search_tenant_context(vec, "t")
        assert mock_collection.query.call_args.kwargs["query_embeddings"] == [vec]

    # ── n_results ─────────────────────────────────────────────────────────────

    def test_n_results_default_is_five(self, mock_http_client):
        result = self._make_query_result(["a"], [1], [0.1])
        manager, mock_collection = self._manager_with_collection(mock_http_client, result)
        manager.search_tenant_context([0.0] * 384, "t")
        assert mock_collection.query.call_args.kwargs["n_results"] == 5

    def test_custom_n_results_is_passed_through(self, mock_http_client):
        result = self._make_query_result(["a"], [1], [0.1])
        manager, mock_collection = self._manager_with_collection(mock_http_client, result)
        manager.search_tenant_context([0.0] * 384, "t", n_results=8)
        assert mock_collection.query.call_args.kwargs["n_results"] == 8

    # ── include= fields ───────────────────────────────────────────────────────

    def test_include_specifies_all_required_fields(self, mock_http_client):
        result = self._make_query_result(["a"], [1], [0.1])
        manager, mock_collection = self._manager_with_collection(mock_http_client, result)
        manager.search_tenant_context([0.0] * 384, "t")
        include = mock_collection.query.call_args.kwargs.get("include", [])
        assert "documents" in include
        assert "metadatas" in include
        assert "distances" in include

    # ── return shape ──────────────────────────────────────────────────────────

    def test_returns_flat_list_not_list_of_lists(self, mock_http_client):
        result = self._make_query_result(["a", "b"], [1, 2], [0.1, 0.3])
        manager, mock_collection = self._manager_with_collection(mock_http_client, result)
        out = manager.search_tenant_context([0.0] * 384, "t")
        assert isinstance(out, list)
        assert all(isinstance(r, dict) for r in out)

    def test_returns_correct_number_of_results(self, mock_http_client):
        result = self._make_query_result(["a", "b", "c"], [1, 2, 3], [0.1, 0.2, 0.3])
        manager, mock_collection = self._manager_with_collection(mock_http_client, result)
        out = manager.search_tenant_context([0.0] * 384, "t")
        assert len(out) == 3

    def test_result_fields_text_page_number_distance(self, mock_http_client):
        result = self._make_query_result(["first chunk", "second chunk"], [1, 4], [0.1, 0.3])
        manager, mock_collection = self._manager_with_collection(mock_http_client, result)
        out = manager.search_tenant_context([0.0] * 384, "t")
        assert out[0] == {"text": "first chunk", "page_number": 1, "distance": 0.1}
        assert out[1] == {"text": "second chunk", "page_number": 4, "distance": 0.3}

    def test_text_comes_from_documents(self, mock_http_client):
        result = self._make_query_result(["specific content"], [3], [0.05])
        manager, mock_collection = self._manager_with_collection(mock_http_client, result)
        out = manager.search_tenant_context([0.0] * 384, "t")
        assert out[0]["text"] == "specific content"

    def test_page_number_comes_from_metadata(self, mock_http_client):
        result = self._make_query_result(["text"], [7], [0.2])
        manager, mock_collection = self._manager_with_collection(mock_http_client, result)
        out = manager.search_tenant_context([0.0] * 384, "t")
        assert out[0]["page_number"] == 7

    def test_distance_comes_from_distances(self, mock_http_client):
        result = self._make_query_result(["text"], [1], [0.42])
        manager, mock_collection = self._manager_with_collection(mock_http_client, result)
        out = manager.search_tenant_context([0.0] * 384, "t")
        assert out[0]["distance"] == pytest.approx(0.42)

    def test_empty_result_set_returns_empty_list(self, mock_http_client):
        _, mock_instance = mock_http_client
        mock_collection = MagicMock()
        mock_instance.get_or_create_collection.return_value = mock_collection
        mock_collection.query.return_value = {
            "ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]
        }
        out = VectorStoreManager().search_tenant_context([0.0] * 384, "t")
        assert out == []


class TestDeleteDocument:

    def _manager_with_mock_collection(self, mock_http_client):
        _, mock_instance = mock_http_client
        mock_collection = MagicMock()
        mock_instance.get_or_create_collection.return_value = mock_collection
        return VectorStoreManager(), mock_collection

    # ── collection.delete() is called ────────────────────────────────────────

    def test_delete_calls_collection_delete_once(self, mock_http_client):
        manager, mock_collection = self._manager_with_mock_collection(mock_http_client)
        manager.delete_document("abc123", "tenant_a")
        mock_collection.delete.assert_called_once()

    # ── compound where= filter ────────────────────────────────────────────────

    def test_where_filter_is_and_clause(self, mock_http_client):
        manager, mock_collection = self._manager_with_mock_collection(mock_http_client)
        manager.delete_document("abc123", "tenant_a")
        where = mock_collection.delete.call_args.kwargs["where"]
        assert "$and" in where

    def test_where_filter_includes_file_hash(self, mock_http_client):
        manager, mock_collection = self._manager_with_mock_collection(mock_http_client)
        manager.delete_document("abc123", "tenant_a")
        where = mock_collection.delete.call_args.kwargs["where"]
        conditions = where["$and"]
        assert any(c.get("file_hash", {}).get("$eq") == "abc123" for c in conditions)

    def test_where_filter_includes_tenant_id(self, mock_http_client):
        manager, mock_collection = self._manager_with_mock_collection(mock_http_client)
        manager.delete_document("abc123", "tenant_a")
        where = mock_collection.delete.call_args.kwargs["where"]
        conditions = where["$and"]
        assert any(c.get("tenant_id", {}).get("$eq") == "tenant_a" for c in conditions)

    def test_different_file_hashes_produce_different_filters(self, mock_http_client):
        manager, mock_collection = self._manager_with_mock_collection(mock_http_client)
        manager.delete_document("hash_one", "t")
        manager.delete_document("hash_two", "t")
        calls = mock_collection.delete.call_args_list
        where_0 = calls[0].kwargs["where"]["$and"]
        where_1 = calls[1].kwargs["where"]["$and"]
        hashes = lambda c: next(v["$eq"] for d in c for k, v in d.items() if k == "file_hash")
        assert hashes(where_0) == "hash_one"
        assert hashes(where_1) == "hash_two"

    def test_different_tenant_ids_produce_different_filters(self, mock_http_client):
        manager, mock_collection = self._manager_with_mock_collection(mock_http_client)
        manager.delete_document("h", "tenant_x")
        manager.delete_document("h", "tenant_y")
        calls = mock_collection.delete.call_args_list
        get_tenant = lambda call: next(
            v["$eq"]
            for d in call.kwargs["where"]["$and"]
            for k, v in d.items()
            if k == "tenant_id"
        )
        assert get_tenant(calls[0]) == "tenant_x"
        assert get_tenant(calls[1]) == "tenant_y"

    # ── return value ──────────────────────────────────────────────────────────

    def test_returns_none(self, mock_http_client):
        manager, _ = self._manager_with_mock_collection(mock_http_client)
        result = manager.delete_document("h", "t")
        assert result is None


class TestDocumentExists:
    """Task 4.5 — document_exists() checks whether any chunks for a
    {file_hash, tenant_id} pair are already stored in ChromaDB."""

    def _manager_with_collection(self, mock_http_client, get_return):
        _, mock_instance = mock_http_client
        mock_collection = MagicMock()
        mock_instance.get_or_create_collection.return_value = mock_collection
        mock_collection.get.return_value = get_return
        return VectorStoreManager(), mock_collection

    # --- return value ---

    def test_returns_true_when_ids_found(self, mock_http_client):
        manager, _ = self._manager_with_collection(mock_http_client, {"ids": ["abc_1_0"]})
        assert manager.document_exists("abc", "tenant_a") is True

    def test_returns_false_when_no_ids_found(self, mock_http_client):
        manager, _ = self._manager_with_collection(mock_http_client, {"ids": []})
        assert manager.document_exists("abc", "tenant_a") is False

    # --- collection.get() call arguments ---

    def test_uses_and_filter_with_file_hash_and_tenant(self, mock_http_client):
        manager, mock_collection = self._manager_with_collection(mock_http_client, {"ids": []})
        manager.document_exists("hashXYZ", "tenant_b")
        where = mock_collection.get.call_args.kwargs["where"]
        assert "$and" in where
        conditions = where["$and"]
        assert any(c.get("file_hash", {}).get("$eq") == "hashXYZ" for c in conditions)
        assert any(c.get("tenant_id", {}).get("$eq") == "tenant_b" for c in conditions)

    def test_requests_ids_only_via_empty_include(self, mock_http_client):
        """Minimal response — we only need to know if any row matches."""
        manager, mock_collection = self._manager_with_collection(mock_http_client, {"ids": []})
        manager.document_exists("h", "t")
        assert mock_collection.get.call_args.kwargs.get("include") == []

    def test_requests_limit_one(self, mock_http_client):
        """Fetching 1 ID is enough; no need to scan all matching rows."""
        manager, mock_collection = self._manager_with_collection(mock_http_client, {"ids": []})
        manager.document_exists("h", "t")
        assert mock_collection.get.call_args.kwargs.get("limit") == 1

    def test_different_file_hashes_produce_different_filters(self, mock_http_client):
        manager, mock_collection = self._manager_with_collection(mock_http_client, {"ids": []})
        manager.document_exists("hash_one", "t")
        manager.document_exists("hash_two", "t")
        calls = mock_collection.get.call_args_list
        get_hash = lambda call: next(
            v["$eq"] for d in call.kwargs["where"]["$and"]
            for k, v in d.items() if k == "file_hash"
        )
        assert get_hash(calls[0]) == "hash_one"
        assert get_hash(calls[1]) == "hash_two"

    def test_same_file_hash_different_tenants_are_independent(self, mock_http_client):
        """A document stored for tenant_a does not count as existing for tenant_b."""
        manager, mock_collection = self._manager_with_collection(mock_http_client, {"ids": []})
        manager.document_exists("shared_hash", "tenant_a")
        manager.document_exists("shared_hash", "tenant_b")
        calls = mock_collection.get.call_args_list
        get_tenant = lambda call: next(
            v["$eq"] for d in call.kwargs["where"]["$and"]
            for k, v in d.items() if k == "tenant_id"
        )
        assert get_tenant(calls[0]) == "tenant_a"
        assert get_tenant(calls[1]) == "tenant_b"


class TestHeartbeat:

    def test_returns_value_from_client_heartbeat(self, mock_http_client):
        _, mock_instance = mock_http_client
        mock_instance.heartbeat.return_value = 1_234_567_890
        manager = VectorStoreManager()
        result = manager.heartbeat()
        assert result == 1_234_567_890

    def test_delegates_to_client_exactly_once(self, mock_http_client):
        _, mock_instance = mock_http_client
        mock_instance.heartbeat.return_value = 0
        VectorStoreManager().heartbeat()
        mock_instance.heartbeat.assert_called_once()
