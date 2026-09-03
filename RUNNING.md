# Running the Stack & Viewing Logs

Four things need to be running for the full app to work. Each gets its own
terminal tab so you can watch its logs live.

## 1. Infrastructure (Redis + ChromaDB)

```bash
cd "Enterprise RAG"
docker compose up -d      # starts both containers if not already up
docker compose logs -f    # tail both containers' logs (Ctrl-C to stop tailing, containers keep running)
docker compose ps         # check status without tailing
```

## 2. Celery worker

```bash
source .venv/bin/activate
celery -A app.worker.celery_app worker --loglevel=info --pool=solo
```

Logs print directly to this terminal: `Task ... received` when an ingestion
job starts, `succeeded in Xs: {...}` when it finishes.

> **macOS gotcha:** always pass `--pool=solo`. The default `--pool=prefork`
> forks worker processes *after* `sentence-transformers`/PyTorch has already
> loaded at module import (Task 4.2's singleton pattern) — forking a process
> with native ML libraries already initialized segfaults on macOS
> (`Worker exited prematurely: signal 11 (SIGSEGV)`). Not an issue in a Linux
> container deployment, but a very common local-dev trap. See `learning.md`,
> Task 6.1.

## 3. FastAPI gateway

```bash
source .venv/bin/activate
uvicorn app.main:app --port 8001 --reload
```

`--reload` picks up code changes automatically. Logs show one line per
request, e.g. `POST /api/v1/query HTTP/1.1" 200 OK`.

Health check: `curl http://localhost:8001/health` → `{"redis": "ok", "chroma": "ok"}`.

## 4. Streamlit UI

```bash
source .venv/bin/activate
streamlit run ui.py
```

Opens `http://localhost:8501` in your browser automatically and prints its
own logs to this terminal. Override the API it talks to with
`RAG_API_BASE_URL` (defaults to `http://localhost:8001`):

```bash
RAG_API_BASE_URL=http://localhost:8001 streamlit run ui.py
```

## Quick health check, all four at once

```bash
docker compose ps                                    # both containers "healthy"
pgrep -fl "celery -A app.worker.celery_app"           # worker process present
curl -s http://localhost:8001/health                  # {"redis":"ok","chroma":"ok"}
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8501
```

## Known issue to watch for while testing

Re-ingesting a byte-identical PDF under a *different* `tenant_id` silently
fails to create separate rows — the composite ChromaDB ID is
`{file_hash}_{page}_{chunk}` with no `tenant_id` in it (`app/vector_db.py`,
`insert_chunks`). The Celery task still reports success, but the second
tenant's document won't show up in their queries. Not yet fixed — see
`learning.md`, Task 7.1, for the full writeup. Test with tenants that don't
share identical source files, or a tenant that hasn't ingested that file yet.
