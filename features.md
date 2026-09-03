# Enterprise RAG — Product Feature Specification

A production-grade, zero-cost-infrastructure Document Q&A engine built on local embeddings, ChromaDB, and OpenAI completion. This document describes every feature the system supports, organized by capability domain.

---

## 1. Document Ingestion

### 1.1 PDF Upload
- Accepts PDF files via a multipart HTTP POST to `/api/v1/docs/upload`.
- Enforces strict `.pdf`-only file type validation at the gateway layer — all other formats are rejected with a `400 Bad Request`.
- Accepts a `tenant_id` form field alongside the file to scope the document to a specific tenant.

### 1.2 SHA-256 Deduplication Fingerprinting
- Computes a SHA-256 hash of the raw binary file contents immediately after upload.
- Stores the file on disk using the hash as the filename, preventing duplicate files from occupying storage.
- Before enqueuing ingestion, checks ChromaDB for existing vectors with the same `file_hash` and `tenant_id` — returns `200 Already Exists` without re-processing if found.

### 1.3 Asynchronous Non-Blocking Ingestion
- Heavy processing (parsing, embedding, storing) is offloaded to a Celery background worker via `.delay()`.
- The HTTP gateway responds immediately with `202 Processing Queued` and a `task_id` — the client is never blocked waiting for ingestion to complete.

### 1.4 Structural PDF Parsing
- Uses PyMuPDF (`fitz`) to extract text from each page of the PDF preserving structural document lines.
- Handles multi-page documents by iterating page-by-page and collecting clean text output.

### 1.5 Sliding Window Text Chunking
- Splits extracted page text into overlapping chunks using a sliding window algorithm.
- Default boundaries: **400 words per chunk** with a **50-word overlap** between adjacent chunks.
- Each chunk is returned as a structured dictionary containing:
  - `text` — the chunk content
  - `page_number` — source page in the original PDF
  - `chunk_index` — sequential position within the document

---

## 2. Vector Embeddings

### 2.1 Local Embedding Model
- Uses the `sentence-transformers` library with the `all-MiniLM-L6-v2` model running entirely on local CPU.
- No external embedding API calls — zero per-token embedding cost.
- Model weights are loaded once per Celery worker lifespan (lazy-loaded outside task functions) to avoid repeated cold starts.

### 2.2 Single-String Embedding
- Converts any text string to a **384-dimensional float vector** for use at query time.

### 2.3 Batch Embedding
- Processes arrays of chunk texts in a single batched call, optimized for sequential CPU throughput during ingestion.

---

## 3. Multi-Tenant Vector Storage

### 3.1 ChromaDB HTTP Vector Store
- Connects to a self-hosted ChromaDB server (Docker) via `chromadb.HttpClient`.
- All vectors are stored in a single unified collection: `enterprise_knowledge_base`.

### 3.2 Composite Document ID Keying
- Every vector is stored with a deterministic unique ID in the format: `{file_hash}_{page_number}_{chunk_index}`.
- Prevents duplicate vector insertions and enables targeted deletion by document.

### 3.3 Per-Vector Metadata Injection
- Every stored vector carries a strict metadata payload:
  - `tenant_id` — the owning tenant
  - `file_hash` — SHA-256 fingerprint of the source document
  - `page_number` — source page for citation traceability

### 3.4 Tenant-Isolated Similarity Search
- All vector similarity queries include a hard `where={"tenant_id": tenant_id}` metadata filter.
- Programmatically enforced — tenant A can never retrieve vectors belonging to tenant B regardless of semantic similarity.

### 3.5 Document Deletion
- Accepts a `DELETE /api/v1/docs/{file_hash}?tenant_id=` request.
- Removes all ChromaDB vectors matching the `file_hash` and `tenant_id` combination via metadata-filtered delete.
- Returns `204 No Content` on success — supports GDPR data-erasure and document lifecycle management.

---

## 4. Retrieval-Augmented Generation (RAG) Query Engine

### 4.1 Semantic Query Endpoint
- Accepts a JSON POST to `/api/v1/query` with `query` (string) and `tenant_id` (string) fields.
- Input is validated via a Pydantic schema — malformed requests are rejected before any processing.

### 4.2 Tenant-Scoped Retrieval
- The query string is embedded locally using the same `all-MiniLM-L6-v2` model used during ingestion.
- ChromaDB is queried with the embedding vector filtered strictly to the requesting tenant's data.

### 4.3 Context-Grounded Completion
- Retrieved chunks are assembled into a system prompt that instructs the LLM to answer **only from the provided context** and explicitly forbids drawing on external knowledge.
- Calls OpenAI's `gpt-4o-mini` model at `temperature=0.0` for deterministic, factual answers.

### 4.4 Streaming Response
- The OpenAI call is made with `stream=True`.
- Tokens are yielded progressively via FastAPI's `StreamingResponse` using an async generator.
- Eliminates the perception of latency on long answers — the client receives the first token within milliseconds.

### 4.5 Singleton Dependency Injection
- The local embedder and ChromaDB client are instantiated once per application process and injected into route handlers via FastAPI's `Depends()` mechanism.
- Prevents model re-loading on every request.

---

## 5. Async Task Management

### 5.1 Celery + Redis Task Queue
- Celery is configured with Redis as both the message broker and result backend.
- Background ingestion jobs are dispatched to the queue and processed independently of the web gateway.

### 5.2 Task Status Polling
- `GET /api/v1/tasks/{task_id}` proxies Celery's `AsyncResult` to expose task lifecycle state.
- Returns a JSON body with:
  - `status` — one of `PENDING`, `STARTED`, `SUCCESS`, `FAILURE`
  - `result` — output payload on success
  - `error` — exception message on failure
- Allows clients to poll for ingestion completion without webhooks or websockets.

### 5.3 Worker Resilience
- Workers can be scaled horizontally by launching multiple Celery worker processes against the same Redis broker.
- Temporary staging files are cleaned up by the worker after successful ChromaDB insertion.

---

## 6. Observability & Operations

### 6.1 Health Check Endpoint
- `GET /health` actively pings both infrastructure dependencies:
  - Redis — via a `PING` command
  - ChromaDB — via `client.heartbeat()`
- Returns a structured JSON status for each dependency.
- Compatible with Docker `HEALTHCHECK`, Kubernetes liveness/readiness probes, and uptime monitors.

### 6.2 Structured Application Logging
- Each component (parser, embedder, worker, API routes) emits structured log lines covering key lifecycle events: file received, task enqueued, chunks embedded, vectors stored, query executed.
- Log level is configurable via environment variable.

---

## 7. Security & Configuration

### 7.1 Environment Variable Isolation
- All secrets and connection coordinates (`OPENAI_API_KEY`, `REDIS_URL`, `CHROMA_HOST`, `CHROMA_PORT`) are stored in a `.env` file, never hardcoded.
- Loaded and validated at startup via a `pydantic-settings` `BaseSettings` model — the app fails fast on missing or malformed config.

### 7.2 File Type Enforcement
- Only `.pdf` files are accepted at the upload endpoint. Extension is validated server-side, not trusted from the client's `Content-Type` header.

### 7.3 Tenant Isolation Guarantee
- Tenant data isolation is enforced at the database query layer, not the application layer — a bug in business logic cannot cause cross-tenant data leakage because the filter is applied within the ChromaDB query itself.

### 7.4 Idempotent Document Handling
- Re-uploading an identical document for the same tenant is a no-op — the system detects the duplicate via SHA-256 hash before any processing or storage occurs.

---

## 8. Infrastructure

### 8.1 Docker Compose Local Stack
- A single `docker-compose.yml` provisions the full stateless infrastructure:
  - `redis_broker` — Redis on port `6379` (message broker + Celery result backend)
  - `chroma_server` — ChromaDB on port `8000` with telemetry disabled and a local volume (`./chroma_data`) for vector persistence across restarts.

### 8.2 Uvicorn Web Server
- FastAPI is served by Uvicorn on port `8001`, separate from ChromaDB's port to avoid conflicts.

### 8.3 Stateless Application Layer
- The FastAPI gateway and Celery workers are stateless — they hold no in-process data between requests beyond singleton model instances.
- Horizontal scaling requires only adding more worker processes behind the same Redis broker.

---

## 9. Developer & Demo Experience

### 9.1 Streamlit UI
- A single-page `ui.py` Streamlit application provides a visual interface for the API.
- Features:
  - File uploader widget that triggers the async ingestion flow and displays the returned `task_id`.
  - Chat input box that calls `/api/v1/query` and streams the answer back to the screen.
- Intended for demos and manual testing — not a production frontend.

### 9.2 Chunking Strategy Experimentation
- A comparison script runs the same PDF through both the fixed-size sliding window chunker and a sentence-boundary chunker (`nltk.sent_tokenize`).
- Retrieval quality is manually evaluated against a 5-question golden set to observe the impact of chunk boundary decisions on answer accuracy.

### 9.3 Retrieval Quality Golden Set
- A curated set of 5 test questions with known ground-truth answers, run against a fixed test PDF.
- Used to benchmark retrieval accuracy at different `n_results` values (3 vs 8) — makes the precision/recall tradeoff observable and measurable.

### 9.4 Architecture Diagram & README
- A data-flow diagram covers both pipeline paths:
  - **Ingest path**: PDF → Parser → Chunker → Embedder → ChromaDB
  - **Query path**: Query → Embedder → ChromaDB → OpenAI → Streaming Response
- `README.md` documents environment setup, `.env` configuration, and the exact commands to start each component (Docker, Celery worker, Uvicorn).

---

## API Surface Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/docs/upload` | Upload a PDF; returns `202` with `task_id` |
| `GET` | `/api/v1/tasks/{task_id}` | Poll ingestion task status |
| `POST` | `/api/v1/query` | Semantic Q&A against tenant documents |
| `DELETE` | `/api/v1/docs/{file_hash}` | Remove a document's vectors by hash |
| `GET` | `/health` | Infrastructure liveness check |
