# Enterprise RAG

A production-grade, zero-cost Document Q&A (RAG) engine. Upload a PDF, ask questions, get answers grounded strictly in that document — nothing invented, nothing pulled from outside the source.

Built as a learning project to practice production RAG patterns end-to-end: async ingestion, multi-tenant vector isolation, streaming synthesis, and the operational scaffolding (health checks, task polling, idempotency) that a toy notebook version skips.

## Why "zero-cost"

- **Embeddings** run locally via `sentence-transformers` — no per-token embedding API cost.
- **Vector storage** is self-hosted ChromaDB in Docker — no managed vector DB bill.
- **Only the final answer synthesis** calls an external LLM (Gemini's free/cheap tier), and only after retrieval has already narrowed the context to a handful of chunks.

## Stack

| Layer | Technology |
|---|---|
| Web gateway | FastAPI + Uvicorn |
| PDF parsing | PyMuPDF (`fitz`) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, local, 384-dim) |
| Vector store | ChromaDB (self-hosted, Docker) |
| Async processing | Celery + Redis |
| Answer synthesis | Gemini (`gemini-3.5-flash`) via the OpenAI-compatible endpoint, `temperature=0.0` |
| Demo UI | Streamlit |

## How it works

**Ingestion** — a PDF is uploaded, hashed (SHA-256) for deduplication, and handed to a Celery worker that parses it page-by-page, splits it into overlapping 400-word chunks (50-word overlap), embeds each chunk locally, and stores the vectors in ChromaDB tagged with `tenant_id` and `file_hash` metadata. The HTTP request returns immediately with a `task_id` — it never blocks on parsing/embedding.

**Query** — a question is embedded with the same local model, ChromaDB is searched with a hard `where={"tenant_id": ...}` filter (so one tenant can never retrieve another tenant's chunks), and the retrieved passages are assembled into a prompt that instructs Gemini to answer *only* from that context. The answer streams back token-by-token over SSE.

Full request/response sequence diagrams for every flow (ingestion, query, task polling, deletion, tenant isolation, the Streamlit session, and deployment topology) — each annotated with the *why* behind the design, not just the *what* — live in [architecture.md](architecture.md).

## API surface

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v1/docs/upload` | Upload a PDF (`file` + `tenant_id`); returns `202` with `task_id` |
| `GET` | `/api/v1/tasks/{task_id}` | Poll ingestion task status (`PENDING`/`SUCCESS`/`FAILURE`) |
| `POST` | `/api/v1/query` | Semantic Q&A against a tenant's documents (SSE stream) |
| `DELETE` | `/api/v1/docs/{file_hash}?tenant_id=` | Remove a document's vectors for one tenant |
| `GET` | `/health` | Liveness check — pings Redis and ChromaDB |

Full feature-by-feature spec (validation rules, response shapes, edge cases): [features.md](features.md).

## Quickstart

Requires Python 3.11+, Docker, and a [Gemini API key](https://aistudio.google.com/apikey).

```bash
# 1. Clone and create a virtualenv
git clone <repo-url> && cd enterprise-rag
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies (PyTorch first — CPU build)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# edit .env and set GEMINI_API_KEY

# 4. Start infrastructure (Redis + ChromaDB)
docker compose up -d
```

Then, in separate terminals:

```bash
# Celery worker — --pool=solo is required on macOS, see RUNNING.md
celery -A app.worker.celery_app worker --loglevel=info --pool=solo

# FastAPI gateway
uvicorn app.main:app --port 8001 --reload

# Streamlit demo UI (optional)
streamlit run ui.py
```

Verify everything's up: `curl http://localhost:8001/health` → `{"redis": "ok", "chroma": "ok"}`.

Step-by-step run instructions per process, plus a combined health-check snippet and the macOS `--pool=solo` gotcha explained in full: [RUNNING.md](RUNNING.md).

## Testing

```bash
pytest
```

Unit tests cover the parser, chunker, and embedder in isolation; integration tests exercise ChromaDB tenant isolation, the Celery pipeline, and the API routes end-to-end. Tests live in `tests/`, mirroring the `app/` structure.

ChromaDB is mocked globally for the whole suite (`tests/conftest.py`), so no Docker services are required to run `pytest` locally. The one test that touches a real Redis (`tests/test_worker.py::test_redis_broker_is_reachable`, marked `@pytest.mark.integration`) skips itself when Redis isn't reachable rather than failing.

**CI:** every push/PR to `main` runs the full suite via GitHub Actions ([.github/workflows/tests.yml](.github/workflows/tests.yml)) — no external services or secrets required.

`scripts/` holds standalone evaluation tools used during development, not part of the test suite:

- `verify_pipeline.py` — pipes a sample PDF through parse → chunk → embed to sanity-check the pipeline manually.
- `compare_chunking.py` — compares the fixed-size sliding-window chunker against a sentence-boundary chunker on retrieval quality.
- `eval_retrieval_quality.py` — runs a 5-question golden set against `sample_data/` PDFs to measure precision/recall at different `n_results`.

## Known limitations

- **Composite vector ID has no tenant in it.** IDs are `{file_hash}_{page}_{chunk_index}` — if two different tenants upload a byte-identical PDF, the second tenant's insert is silently dropped by ChromaDB's non-upserting `add()`, even though the ingestion task reports `SUCCESS`. Read-path tenant isolation (the `where` filter) is unaffected; this is a write-path collision only. Full writeup: `learning.md`, Task 7.1.
- **`S3Storage` is stubbed, not implemented.** The storage layer (`app/storage/`) is built behind an abstract `StorageBackend` specifically so a future S3 backend is a drop-in swap, but only `LocalStorage` (filesystem) is wired up today.
- **Not a production frontend.** `ui.py` is a Streamlit demo client for manual testing and interview walkthroughs — it talks to the same public API any other HTTP client would, with no privileged access.

## Project docs

| File | Purpose |
|---|---|
| [tasks.md](tasks.md) | Development checklist — source of truth for what's built |
| [features.md](features.md) | Full feature specification |
| [architecture.md](architecture.md) | Architecture + sequence diagrams, with design-rationale notes |
| [RUNNING.md](RUNNING.md) | Per-process run instructions and troubleshooting |
| [learning.md](learning.md) | Running notes on concepts, gotchas, and interview angles per task |

## License

Apache 2.0 — see [LICENSE](LICENSE).
