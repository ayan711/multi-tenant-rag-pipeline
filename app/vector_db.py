# ─────────────────────────────────────────────────────────────────────────────
# vector_db.py — ChromaDB Vector Store Client
# ─────────────────────────────────────────────────────────────────────────────
#
# All interactions with the ChromaDB vector database are centralised here.
# No other module talks to ChromaDB directly — they all go through this wrapper.
# This keeps the database logic in one place and makes it easy to swap the
# vector store later without touching the rest of the codebase.
#
# Key responsibilities:
#   1. INSERT — Store embedded chunks with rich metadata (tenant_id, file_hash,
#      page_number) so they can be filtered at query time.
#
#   2. SEARCH — Given a query vector, find the N most semantically similar
#      chunks, but ONLY within a specific tenant's data. This is the core
#      multi-tenancy guarantee: the `where={"tenant_id": ...}` filter is
#      applied at the database layer, not the application layer.
#
#   3. DELETE — Remove all vectors belonging to a specific document, identified
#      by its file_hash + tenant_id combination.
#
# Why ChromaDB over Pinecone/Weaviate?
#   It's self-hosted (free), runs in Docker, and supports metadata filtering
#   out of the box. Perfect for a zero-cost local setup.
#
# Implemented in: Task 3.1 through Task 3.6.
# ─────────────────────────────────────────────────────────────────────────────

import chromadb

from app.config import settings


class VectorStoreManager:
    """Wrapper around the ChromaDB HTTP client.

    Single point of contact for all vector database operations.
    Instantiate once per process — the underlying HttpClient is thread-safe,
    so the same instance can be shared across FastAPI routes and Celery workers.
    """

    def __init__(
        self,
        host: str = settings.chroma_host,
        port: int = settings.chroma_port,
        collection_name: str = settings.chroma_collection,
    ) -> None:
        # HttpClient speaks to the ChromaDB server over HTTP, so the
        # chroma_server container must be running before this is called.
        self._client = chromadb.HttpClient(host=host, port=port)
        
        # get_or_create_collection is idempotent: first run creates the
        # collection; every subsequent run returns the existing one.
        self._collection = self._client.get_or_create_collection(
            name=collection_name
        )

    def insert_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        tenant_id: str,
        file_hash: str,
    ) -> int:
        """Store embedded chunks in ChromaDB with tenant-scoped metadata.

        Each chunk gets a composite ID so re-ingesting the same document
        overwrites existing rows instead of creating duplicates — ChromaDB
        upserts on duplicate IDs.

        Args:
            chunks:     Dicts from DocumentParser: {text, page_number, chunk_index}.
            embeddings: Parallel list of 384-dim vectors from LocalEmbedder.
            tenant_id:  Caller's tenant — stored as metadata for query filtering.
            file_hash:  SHA-256 of the source file — used to scope deletions later.

        Returns:
            Number of chunks inserted.
        """
        ids = [
            f"{file_hash}_{c['page_number']}_{c['chunk_index']}" for c in chunks
        ]
        metadatas = [
            {
                "tenant_id": tenant_id,
                "file_hash": file_hash,
                "page_number": c["page_number"],
            }
            for c in chunks
        ]
        documents = [c["text"] for c in chunks]

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return len(chunks)

    def search_tenant_context(
        self,
        query_embedding: list[float],
        tenant_id: str,
        n_results: int = 5,
    ) -> list[dict]:
        """Return the n most similar chunks restricted to the given tenant.

        The where= clause is applied at the ChromaDB layer — callers cannot
        accidentally omit it because it is not an optional argument.

        Args:
            query_embedding: 384-dim vector of the user's query.
            tenant_id:       Only return chunks owned by this tenant.
            n_results:       Number of closest chunks to return (default 5).

        Returns:
            List of {text, page_number, distance} dicts, closest first.
        """
        results = self._collection.query(
            query_embeddings=[query_embedding],  # list-of-lists: one sublist per query vector
            n_results=n_results,
            where={"tenant_id": tenant_id},
            include=["documents", "metadatas", "distances"],
        )
        # query() returns parallel lists-of-lists; [0] selects the single query's results.
        return [
            {
                "text": doc,
                "page_number": meta["page_number"],
                "distance": dist,
            }
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def delete_document(self, file_hash: str, tenant_id: str) -> None:
        """Remove every chunk belonging to one document for one tenant.

        The compound where= filter prevents a tenant from deleting another
        tenant's document even if they know the file_hash.

        Args:
            file_hash: SHA-256 of the document whose chunks should be removed.
            tenant_id: Tenant that owns the document — acts as a delete guard.
        """
        # $and requires explicit $eq operators; ChromaDB shorthand {"k": "v"}
        # only works for single-field filters.
        self._collection.delete(
            where={
                "$and": [
                    {"file_hash": {"$eq": file_hash}},
                    {"tenant_id": {"$eq": tenant_id}},
                ]
            }
        )

    def document_exists(self, file_hash: str, tenant_id: str) -> bool:
        """Return True if at least one chunk for this file/tenant pair is already stored.

        Used as an idempotency check before enqueuing a new ingestion task.
        Fetching limit=1 with include=[] (IDs only) keeps the round-trip minimal.

        Args:
            file_hash: SHA-256 of the file to check.
            tenant_id: Tenant scope — a file_hash that exists for another tenant
                       does NOT count as existing for this one.
        """
        result = self._collection.get(
            where={
                "$and": [
                    {"file_hash": {"$eq": file_hash}},
                    {"tenant_id": {"$eq": tenant_id}},
                ]
            },
            include=[],  # IDs only — we only need to know if any row matches
            limit=1,
        )
        return len(result["ids"]) > 0

    def heartbeat(self) -> int:
        """Return the ChromaDB server's nanosecond timestamp.

        A successful call proves the TCP connection and server are alive.
        Used by the health-check endpoint (Task 5.7).
        """
        return self._client.heartbeat()
