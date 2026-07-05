"""
Integration tests for Tasks 3.5 and 3.6 — Tenant Isolation and Document Deletion.

Uses chromadb.EphemeralClient (in-process, no server needed) so real
insert + query operations run against actual ChromaDB logic, not mocks.

Goal: verify that the where={"tenant_id": ...} filter in
search_tenant_context() structurally prevents cross-tenant data leakage —
even when query vectors are numerically closer to the other tenant's data.
"""

import chromadb
import pytest
from unittest.mock import patch

from app.vector_db import VectorStoreManager


TENANT_A = "tenant_alpha"
TENANT_B = "tenant_beta"

# Tenant A owns seeds 0.1–0.3; Tenant B owns 0.4–0.6.
# A query vector of 0.5 is numerically closest to B's chunks — if isolation
# is broken, tenant A would receive B's results. That's the key test.
_A_SEEDS = [0.1, 0.2, 0.3]
_B_SEEDS = [0.4, 0.5, 0.6]


def _vec(seed: float, dim: int = 384) -> list[float]:
    return [seed] * dim


@pytest.fixture
def populated_manager():
    """
    VectorStoreManager backed by an EphemeralClient pre-loaded with
    3 chunks per tenant. The EphemeralClient is patched in place of
    HttpClient so no Docker service is required.
    """
    ephemeral = chromadb.EphemeralClient()

    with patch("app.vector_db.chromadb.HttpClient", return_value=ephemeral):
        manager = VectorStoreManager(collection_name="test_isolation")

    chunks_a = [
        {"text": "Tenant A: security policy overview", "page_number": 1, "chunk_index": 0},
        {"text": "Tenant A: access control procedures", "page_number": 2, "chunk_index": 1},
        {"text": "Tenant A: incident response plan", "page_number": 3, "chunk_index": 2},
    ]
    manager.insert_chunks(chunks_a, [_vec(s) for s in _A_SEEDS], TENANT_A, "hash_a")

    chunks_b = [
        {"text": "Tenant B: quarterly financial results", "page_number": 1, "chunk_index": 0},
        {"text": "Tenant B: budget allocation report", "page_number": 2, "chunk_index": 1},
        {"text": "Tenant B: revenue projections", "page_number": 3, "chunk_index": 2},
    ]
    manager.insert_chunks(chunks_b, [_vec(s) for s in _B_SEEDS], TENANT_B, "hash_b")

    return manager


class TestTenantIsolation:

    # ── Basic scoping ─────────────────────────────────────────────────────────

    def test_tenant_a_results_contain_only_tenant_a_texts(self, populated_manager):
        results = populated_manager.search_tenant_context(_vec(0.15), TENANT_A, n_results=2)
        for r in results:
            assert "Tenant A" in r["text"]

    def test_tenant_b_results_contain_only_tenant_b_texts(self, populated_manager):
        results = populated_manager.search_tenant_context(_vec(0.45), TENANT_B, n_results=2)
        for r in results:
            assert "Tenant B" in r["text"]

    def test_tenant_a_cannot_see_tenant_b_texts(self, populated_manager):
        results = populated_manager.search_tenant_context(_vec(0.15), TENANT_A, n_results=2)
        for r in results:
            assert "Tenant B" not in r["text"]

    def test_tenant_b_cannot_see_tenant_a_texts(self, populated_manager):
        results = populated_manager.search_tenant_context(_vec(0.45), TENANT_B, n_results=2)
        for r in results:
            assert "Tenant A" not in r["text"]

    # ── The critical case: filter beats distance ───────────────────────────────

    def test_filter_overrides_distance_ordering(self, populated_manager):
        # Query vector 0.5 is numerically closest to tenant B's chunks (0.4–0.6).
        # If isolation is broken, a tenant A query here would return B's results.
        results = populated_manager.search_tenant_context(_vec(0.5), TENANT_A, n_results=2)
        assert len(results) > 0
        for r in results:
            assert "Tenant A" in r["text"], (
                "Distance ordering leaked tenant B's results into tenant A's query"
            )

    # ── No overlap between tenants ────────────────────────────────────────────

    def test_same_query_vector_returns_disjoint_text_sets(self, populated_manager):
        results_a = populated_manager.search_tenant_context(_vec(0.3), TENANT_A, n_results=2)
        results_b = populated_manager.search_tenant_context(_vec(0.3), TENANT_B, n_results=2)
        texts_a = {r["text"] for r in results_a}
        texts_b = {r["text"] for r in results_b}
        assert texts_a.isdisjoint(texts_b), "Tenants share text results — isolation broken"

    # ── Domain-specific keywords don't cross tenants ──────────────────────────

    def test_tenant_a_never_sees_financial_keywords(self, populated_manager):
        results = populated_manager.search_tenant_context(_vec(0.5), TENANT_A, n_results=2)
        texts = " ".join(r["text"] for r in results).lower()
        assert "financial" not in texts
        assert "budget" not in texts
        assert "revenue" not in texts

    def test_tenant_b_never_sees_security_keywords(self, populated_manager):
        results = populated_manager.search_tenant_context(_vec(0.1), TENANT_B, n_results=2)
        texts = " ".join(r["text"] for r in results).lower()
        assert "security" not in texts
        assert "access control" not in texts
        assert "incident" not in texts

    # ── Result count and non-emptiness ────────────────────────────────────────

    def test_results_are_non_empty_for_tenant_with_data(self, populated_manager):
        results = populated_manager.search_tenant_context(_vec(0.1), TENANT_A, n_results=2)
        assert len(results) > 0

    def test_n_results_respected_within_tenant_scope(self, populated_manager):
        results = populated_manager.search_tenant_context(_vec(0.1), TENANT_A, n_results=2)
        assert len(results) == 2

    def test_can_retrieve_all_chunks_for_a_tenant(self, populated_manager):
        # n_results=3 should return all 3 of tenant A's chunks
        results = populated_manager.search_tenant_context(_vec(0.2), TENANT_A, n_results=3)
        assert len(results) == 3

    # ── Result shape ──────────────────────────────────────────────────────────

    def test_results_have_expected_fields(self, populated_manager):
        results = populated_manager.search_tenant_context(_vec(0.1), TENANT_A, n_results=1)
        assert len(results) == 1
        r = results[0]
        assert "text" in r
        assert "page_number" in r
        assert "distance" in r

    def test_distance_is_non_negative(self, populated_manager):
        results = populated_manager.search_tenant_context(_vec(0.1), TENANT_A, n_results=2)
        for r in results:
            assert r["distance"] >= 0.0

    def test_results_ordered_closest_first(self, populated_manager):
        # Query at 0.1, which is identical to tenant A's first chunk.
        # First result should have the smallest distance.
        results = populated_manager.search_tenant_context(_vec(0.1), TENANT_A, n_results=3)
        distances = [r["distance"] for r in results]
        assert distances == sorted(distances)


@pytest.fixture
def two_doc_manager():
    """
    EphemeralClient-backed manager with two documents for tenant A and
    one document for tenant B. Used for deletion tests.

    Tenant A doc 1 (hash_a1): 2 chunks, seeds 0.1 / 0.2
    Tenant A doc 2 (hash_a2): 2 chunks, seeds 0.3 / 0.4
    Tenant B doc 1 (hash_b1): 2 chunks, seeds 0.5 / 0.6
    """
    ephemeral = chromadb.EphemeralClient()
    with patch("app.vector_db.chromadb.HttpClient", return_value=ephemeral):
        manager = VectorStoreManager(collection_name="test_deletion")

    manager.insert_chunks(
        [{"text": "A-doc1 chunk 0", "page_number": 1, "chunk_index": 0},
         {"text": "A-doc1 chunk 1", "page_number": 2, "chunk_index": 1}],
        [_vec(0.1), _vec(0.2)],
        TENANT_A, "hash_a1",
    )
    manager.insert_chunks(
        [{"text": "A-doc2 chunk 0", "page_number": 1, "chunk_index": 0},
         {"text": "A-doc2 chunk 1", "page_number": 2, "chunk_index": 1}],
        [_vec(0.3), _vec(0.4)],
        TENANT_A, "hash_a2",
    )
    manager.insert_chunks(
        [{"text": "B-doc1 chunk 0", "page_number": 1, "chunk_index": 0},
         {"text": "B-doc1 chunk 1", "page_number": 2, "chunk_index": 1}],
        [_vec(0.5), _vec(0.6)],
        TENANT_B, "hash_b1",
    )
    return manager


class TestDocumentDeletion:

    # ── Deleted document is gone ──────────────────────────────────────────────

    def test_deleted_doc_chunks_no_longer_returned(self, two_doc_manager):
        two_doc_manager.delete_document("hash_a1", TENANT_A)
        results = two_doc_manager.search_tenant_context(_vec(0.1), TENANT_A, n_results=2)
        texts = [r["text"] for r in results]
        assert not any("A-doc1" in t for t in texts)

    # ── Sibling document for same tenant is untouched ─────────────────────────

    def test_other_doc_same_tenant_is_untouched(self, two_doc_manager):
        two_doc_manager.delete_document("hash_a1", TENANT_A)
        results = two_doc_manager.search_tenant_context(_vec(0.35), TENANT_A, n_results=2)
        texts = [r["text"] for r in results]
        assert any("A-doc2" in t for t in texts)

    # ── Other tenant's data is untouched ─────────────────────────────────────

    def test_other_tenant_data_untouched_after_delete(self, two_doc_manager):
        two_doc_manager.delete_document("hash_a1", TENANT_A)
        results = two_doc_manager.search_tenant_context(_vec(0.5), TENANT_B, n_results=2)
        texts = [r["text"] for r in results]
        assert any("B-doc1" in t for t in texts)

    # ── Cross-tenant delete guard ─────────────────────────────────────────────

    def test_wrong_tenant_cannot_delete_document(self, two_doc_manager):
        # Tenant B tries to delete tenant A's doc — compound filter means no rows match.
        two_doc_manager.delete_document("hash_a1", TENANT_B)
        # Tenant A's doc should still be retrievable.
        results = two_doc_manager.search_tenant_context(_vec(0.1), TENANT_A, n_results=2)
        texts = [r["text"] for r in results]
        assert any("A-doc1" in t for t in texts)

    # ── Idempotent delete ─────────────────────────────────────────────────────

    def test_deleting_already_deleted_doc_does_not_raise(self, two_doc_manager):
        two_doc_manager.delete_document("hash_a1", TENANT_A)
        # Second delete of the same hash should be a no-op, not an error.
        two_doc_manager.delete_document("hash_a1", TENANT_A)

    def test_deleting_nonexistent_hash_does_not_raise(self, two_doc_manager):
        two_doc_manager.delete_document("does_not_exist", TENANT_A)
