# Enterprise RAG — Implementation Checklist

Production-grade, zero-cost Document Q&A engine. Work through tasks sequentially — each one is independently testable before moving on.

---

## Progress

| Phase | Done | Total |
|-------|------|-------|
| 1 — Infrastructure & Environment | 5 | 5 |
| 2 — Data Extraction & Embedding | 6 | 6 |
| 3 — Vector Database Layer | 6 | 6 |
| 4 — Async Processing Engine | 5 | 5 |
| 5 — FastAPI Gateway | 9 | 9 |
| 6 — System Validation | 5 | 5 |
| 7 — Demo & Interview Readiness | 1 | 2 |

---

## 🏗️ Phase 1 — Local Infrastructure & Environment

- [x] **Task 1.1** — Project Directory Structure Setup
  - Create `app/` and `app/services/` packages with `__init__.py` files.
  - Scaffold empty core modules: `main.py`, `worker.py`, `config.py`, `parser.py`, `embedder.py`, `vector_db.py`.

- [x] **Task 1.2** — Dependency Configuration
  - Create `requirements.txt` with pinned versions for FastAPI, PyMuPDF, ChromaDB, Sentence-Transformers, Celery, Redis, and OpenAI.

- [x] **Task 1.3** — Docker Compose Environment Provisioning
  - Define `redis_broker` (alpine, port `6379`) and `chroma_server` (port `8000`) in `docker-compose.yml`.
  - Mount `./chroma_data` as a bind volume for ChromaDB persistence.

- [x] **Task 1.4** — Environment Variables Isolation
  - Create `.env` with `OPENAI_API_KEY`, `REDIS_URL`, `CHROMA_HOST`, `CHROMA_PORT`.
  - Implement `config.py` using `pydantic-settings` to validate and expose settings at startup.

- [x] **Task 1.5** — Infrastructure Smoke Test
  - Run `docker-compose up -d` and verify both containers are healthy and accepting TCP connections.

---

## 🧱 Phase 2 — Structural Data Extraction & Local Embedding

- [x] **Task 2.1** — PDF Ingestion Engine
  - Implement `DocumentParser.extract_text_per_page()` in `parser.py` using PyMuPDF (`fitz`).
  - Clean each page's text (rejoin hyphenated line-breaks, collapse whitespace, skip blank pages).

- [x] **Task 2.2** — Chunking Sequence Logic
  - Implement `DocumentParser.chunk_pages()` — sliding window, 400 words per chunk, 50-word overlap.
  - Return `{text, page_number, chunk_index}` dicts; `chunk_index` is globally sequential across all pages.

- [x] **Task 2.3** — Local Vector Model Initialization
  - Implement `LocalEmbedder` class in `embedder.py` using `sentence-transformers`.
  - Load `all-MiniLM-L6-v2` locally on instantiation.

- [x] **Task 2.4** — Vectorization Methods
  - Single-string `embed(text)` → 384-dimensional `list[float]`.
  - Batch `embed_batch(texts)` → list of vectors, optimized for CPU.

- [x] **Task 2.5** — In-Memory Ingestion Verification
  - Temporary script: pipe a local PDF through parser → embedder, print vector shapes to confirm the pipeline works end-to-end.

- [x] **Task 2.6** — Chunking Strategy Comparison
  - Run the same PDF through the fixed-size chunker and a sentence-boundary chunker (`nltk.sent_tokenize`).
  - Compare retrieval quality on 5 test questions. Record findings — this is a common interview topic.

---

## 🗄️ Phase 3 — Secure Multi-Tenant Vector Database Layer

- [x] **Task 3.1** — Vector Database Client Setup
  - Implement `VectorStoreManager` in `vector_db.py` using `chromadb.HttpClient`.
  - Connect using `CHROMA_HOST` / `CHROMA_PORT` from settings.

- [x] **Task 3.2** — Collection Initialization
  - Get or create a single collection named `enterprise_knowledge_base`.

- [x] **Task 3.3** — Metadata Injection
  - Build `insert_chunks()`: store vectors with composite ID `{file_hash}_{page_number}_{chunk_index}`.
  - Metadata payload per row: `tenant_id`, `file_hash`, `page_number`.

- [x] **Task 3.4** — Tenant-Scoped Queries
  - Build `search_tenant_context()`: always filter with `where={"tenant_id": tenant_id}` to enforce isolation at the DB layer.

- [x] **Task 3.5** — Isolation Testing
  - Insert mock data for two tenant IDs, then verify that queries for one tenant cannot return the other's vectors.

- [x] **Task 3.6** — Document Deletion Endpoint
  - `DELETE /api/v1/docs/{file_hash}?tenant_id=...` — calls `collection.delete(where={...})` and returns `204 No Content`.

---

## ⚡ Phase 4 — Asynchronous Processing Engine

- [x] **Task 4.1** — Celery Application Setup
  - Instantiate Celery in `worker.py` with Redis as both broker and result backend.

- [x] **Task 4.2** — Singleton Worker Resources
  - Instantiate `LocalEmbedder` and `VectorStoreManager` once at module level so model weights load only once per worker process.

- [x] **Task 4.3** — Ingestion Pipeline Task
  - Implement `@celery_app.task ingest_document_pipeline(path, tenant_id, file_hash)`.
  - Steps: extract pages → chunk → embed batch → insert to ChromaDB → delete temp file.

- [x] **Task 4.4** — Worker Smoke Test
  - Start worker: `celery -A app.worker.celery_app worker --loglevel=info`.
  - Verify it connects to Redis and picks up a test task.

- [x] **Task 4.5** — Idempotent Ingestion Guard
  - Before enqueuing, check if vectors for `{file_hash, tenant_id}` already exist in ChromaDB.
  - Return `200 Already Exists` with the `file_hash` if so — skip redundant processing.

---

## 🌐 Phase 5 — FastAPI Web Gateway & Completion Engine

- [x] **Task 5.1** — FastAPI Application Bootstrap
  - Instantiate FastAPI in `main.py`, register error handlers, and define the base router.

- [x] **Task 5.2** — File Upload Endpoint
  - `POST /api/v1/docs/upload` — accepts `tenant_id` (form field) and `file` (multipart).
  - Reject non-PDF files with `400`.

- [x] **Task 5.3** — SHA-256 Fingerprinting
  - Hash the uploaded file bytes immediately after read.
  - Save to a temp path named by the hash for deduplication.

- [x] **Task 5.4** — Async Task Dispatch
  - Call `ingest_document_pipeline.delay(path, tenant_id, file_hash)`.
  - Return `202 Accepted` with `{task_id}`.

- [x] **Task 5.5** — RAG Query Endpoint
  - `POST /api/v1/query` — Pydantic body: `{query, tenant_id}`.
  - Inject `LocalEmbedder` and `VectorStoreManager` singletons via `Depends`.

- [x] **Task 5.6** — Grounded LLM Synthesis
  - Embed query → scoped ChromaDB search → build system prompt from retrieved chunks.
  - Call `gpt-4o-mini` at `temperature=0.0`; prompt forbids answers outside the provided context.

- [x] **Task 5.7** — Health Check Endpoint
  - `GET /health` — ping Redis (`PING`) and ChromaDB (`client.heartbeat()`).
  - Return `{redis: ok/error, chroma: ok/error}` — used by Docker/Kubernetes readiness probes.

- [x] **Task 5.8** — Streaming Query Response
  - Refactor `/api/v1/query` to use `stream=True` + `StreamingResponse` with an async generator.

- [x] **Task 5.9** — Task Status Endpoint
  - `GET /api/v1/tasks/{task_id}` — proxy `AsyncResult(task_id)`.
  - Return `{status: PENDING|SUCCESS|FAILURE, result?, error?}`.

---

## 🚀 Phase 6 — System Validation & Interview Demos

- [x] **Task 6.1** — Full Stack Startup
  - Docker services up → Celery worker running → Uvicorn on port `8001`.

- [x] **Task 6.2** — Upload Smoke Test
  - `curl` a real PDF to `/api/v1/docs/upload` and confirm a `task_id` comes back.

- [x] **Task 6.3** — Query Verification
  - Ask a question about the uploaded PDF and verify the answer is grounded in the document.

- [x] **Task 6.4** — Multi-Tenant Isolation Audit
  - Ask the same question under a different `tenant_id` — confirm no data leaks across tenants.

- [x] **Task 6.5** — Retrieval Quality Evaluation
  - Write 5 questions with known answers for a test PDF.
  - Compare accuracy at `n_results=3` vs `n_results=8` — observe precision/recall tradeoff firsthand.

---

## 🎓 Phase 7 — Demo & Interview Readiness

- [x] **Task 7.1** — Streamlit UI
  - Single-page app (`ui.py`): file uploader + chat input wired to the FastAPI endpoints.

- [ ] **Task 7.2** — Architecture Diagram & README
  - [x] Data-flow diagrams in `architecture.md`: system overview, ingestion, query (RAG), task-status poll, deletion, tenant isolation, Streamlit UI session, deployment topology — updated for Gemini/storage-abstraction/UI and annotated with WHY notes per diagram.
  - [ ] `README.md` covering setup, env vars, and how to run each component. (`RUNNING.md` covers the "how to run" part already; README still pending.)
