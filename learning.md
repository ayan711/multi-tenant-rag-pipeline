# Enterprise RAG — Learning Notes

Key gotchas and non-obvious decisions per task. Skip what you already know; re-read before interviews.

---

## Phase 1 — Infrastructure & Environment

### Task 1.2: Dependencies

**PyTorch must be installed separately** — `torch` comes in CPU-only and CUDA variants. Plain `pip install torch` may pull the wrong one. Always install it first with an explicit index URL before running `pip install -r requirements.txt`.

**Broker vs result backend** — Celery uses Redis in two distinct roles:

| Role | Stores | Used by |
|---|---|---|
| Broker | Task messages (function + args) | FastAPI writes; worker reads and deletes |
| Result backend | Task state + return value | Worker writes on finish; FastAPI reads via `AsyncResult` |

Both point to the same Redis URL in this project. In production you'd split them to control memory pressure independently.

**`temperature=0.0`** — makes the LLM fully deterministic. For RAG the model's job is to summarise what the document says, not be creative. Zero temperature prevents hallucination drift.

**`python-multipart`** — FastAPI silently requires this to parse `multipart/form-data`. Missing it causes a confusing 422 on upload, not an import error.

---

### Task 1.3: Docker Compose

**"Up" ≠ "Healthy"** — Docker marks a container Up the moment the process starts. ChromaDB takes a few seconds to initialise. Always look for `(healthy)` in `docker compose ps`, not just `Up`.

**Port conflict** — ChromaDB defaults to 8000 inside its container; FastAPI also defaults to 8000 as a host process. They'd collide. Fix: run FastAPI on 8001. Only the left side of Docker's `HOST:CONTAINER` mapping is yours to change.

---

### Task 1.4: Environment Variables

**`SecretStr`** — wraps the API key so `str(settings.openai_api_key)` prints `**********`. Call `.get_secret_value()` explicitly when you need the raw string. You can `grep` for `.get_secret_value()` in code review to audit every place the key is used.

**`@lru_cache`** — memoises a function: first call executes and stores the result; every subsequent call with the same arguments returns the cached object instantly.

```python
@lru_cache
def get_settings() -> Settings:
    return Settings()   # reads .env from disk

s1 = get_settings()    # runs — reads .env
s2 = get_settings()    # returns cached — no disk I/O
assert s1 is s2        # True — literally the same object
```

**LRU = Least Recently Used** — the eviction policy when `maxsize` is set. When the cache is full, the least recently used entry is dropped. With no `maxsize` (our case) it never evicts — effectively a permanent singleton. **No external server needed** — it's a plain dictionary in process RAM, maintained by Python itself.

Used in two places in this project:

| Function | Caches | Why |
|---|---|---|
| `get_settings()` | `Settings` object | Reads `.env` once; all modules share one instance |
| `get_local_storage()` | `LocalStorage` instance | Lazy singleton — created on first request, not at import |

Also makes the dependency swappable in tests without a real `.env`: `app.dependency_overrides[get_settings] = lambda: Settings(openai_api_key="test")`.

**`Field(ge=1, le=65535)`** — pydantic validates this at startup. A bad `CHROMA_PORT=99999` crashes immediately with a clear error, not mysteriously inside a live request.

---

### Task 1.5: Smoke Test

**Port conflict fix** — another project's Redis was already on 6379. Changed the left side of Docker's mapping to 6380 and updated `REDIS_URL` in `.env`. The container still listens on 6379 internally.

**`version:` key** — remove it from `docker-compose.yml`. It was meaningful for the old v1 CLI (`docker-compose`). The current `docker compose` ignores it and warns on every command.

---

## Phase 2 — PDF Extraction & Embedding

### Task 2.1: PDF Ingestion Engine

**Why PyMuPDF** — wraps the MuPDF C engine. Fastest parser and handles complex layouts well. Imported as `fitz` — legacy naming quirk from when the package was called python-fitz.

**Hyphenated line breaks** — PDFs are designed for printing. When a word wraps at the right margin the renderer inserts `re-\nsponsible`. Without `re.sub(r"-\n\s*", "", text)` the embedder sees two tokens instead of one. This is the most common PDF extraction artifact.

> **Interview angle:** *"How do you handle noisy PDFs?"* — The hyphenation join and whitespace collapse are the two fixes that matter for almost every real PDF.

---

### Task 2.2: Chunking

**Sliding window formula:** `step = chunk_size - overlap`. With defaults (400/50): chunks start at 0, 350, 700… Words 350–399 appear in both the first and second chunk. That's the overlap at work.

**`chunk_index` is global, not per-page** — the composite ChromaDB ID is `{file_hash}_{page}_{chunk_index}`. A per-page index would collide on multi-page documents.

> **Interview angle:** *"How did you choose chunk size?"* — 400 words (~600 tokens) fits the embedding model's window with room to spare. The 50-word overlap (~12%) covers most sentence-spanning boundaries. Tune empirically by testing retrieval on known questions (Task 6.5).

---

### Task 2.3: Local Embedding Model

**Model reference:**

| Property | Value |
|---|---|
| Model | `all-MiniLM-L6-v2` (distilled BERT, 6 layers) |
| Output | 384-dim vector per chunk |
| Speed | ~30ms per chunk on CPU |
| Download | ~80 MB, cached in `~/.cache/huggingface/` |

**First-call warmup** — `SentenceTransformer(...)` takes ~0.5–1s to load weights from disk. This is why `LocalEmbedder` is instantiated once at worker startup (Task 4.2), not on every task.

**Switching models mid-stream breaks everything** — new model outputs a different vector dimension. ChromaDB rejects inserts and queries with a dimension mismatch. The only fix is a full re-index: drop the collection, re-embed every chunk, reinsert.

---

### Task 2.4: Vectorization

**`embed_batch` over looping `embed`** — `encode()` amortises tokeniser overhead across the whole batch. For a 150-chunk document, one batch call is significantly faster than 150 individual calls.

**`.tolist()` is required** — `encode()` returns a `numpy.ndarray`. ChromaDB's client expects `list[float]`. Passing a numpy array causes a type error at insert time.

---

### Task 2.6: Chunking Strategy Comparison

| Property | Fixed-size | Sentence-boundary |
|---|---|---|
| Split mechanism | Word-count sliding window | Accumulate sentences until `max_words` |
| Mid-sentence cuts | Yes | Never |
| Predictability | High | Low — one long sentence = unexpectedly large chunk |
| Dependency | None | `nltk` + `punkt_tab` corpus |

**Experiment finding on ISO 27001** — both strategies produced identical rankings. Each page averaged 256 words, well under the 400-word limit, so neither chunker ever needed to split. Conclusion: strategy only matters when pages are long enough to require splitting.

**`sent_tokenize` gotcha** — Punkt model makes mistakes on abbreviations in technical documents: `"Fig. 1"` and `"e.g."` may trigger false sentence splits. Mitigation: provide a custom abbreviation list via `PunktParameters`, or switch to spaCy.

---

## Phase 3 — Multi-Tenant Vector Database

### Task 3.1: ChromaDB Client

**`HttpClient` not `PersistentClient`** — Celery workers and FastAPI routes are separate processes. `HttpClient` is the only mode that lets multiple processes share one ChromaDB instance. `PersistentClient` is single-process only.

---

### Task 3.2: Collection Initialization

**`get_or_create_collection`** — safe to call on every startup. `create_collection` raises if the collection already exists, crashing every worker after the first.

**One collection, all tenants** — multi-tenancy is enforced via `where={"tenant_id": ...}` on every query, not via separate collections. Adding a new tenant requires zero schema changes. The risk: forgetting the filter. Mitigated by making `tenant_id` a required argument on `search_tenant_context()` (Task 3.4).

---

### Task 3.3: Metadata Injection

**Composite ID: `{file_hash}_{page_number}_{chunk_index}`**
- Globally unique: `chunk_index` is sequential across the whole document
- Idempotent: ChromaDB upserts on duplicate IDs — re-ingesting overwrites instead of duplicating
- Deletable by document: `where={"file_hash": ...}` removes all chunks of one file in one call

**Why `where=` filters degrade at scale (ChromaDB)**

ChromaDB builds one HNSW graph across ALL tenants. A query for tenant A traverses the entire graph, including B and C nodes, then post-filters. If A owns 5% of vectors, 95% of traversal steps are wasted. **Qdrant's native multi-tenancy** routes each query to a per-tenant HNSW segment — zero wasted steps, isolation is structural not a filter.

> **Interview angle:** *"At what scale would you move away from ChromaDB's where= filter?"* — Past ~1M total vectors or when query latency becomes noticeable. Qdrant with `payload_index` + `tenant` partitioning is the drop-in architectural upgrade.

---

### Task 3.4: Tenant-Scoped Queries

**Lists-of-lists gotcha** — `collection.query()` supports batch queries. Even for a single query vector, results are wrapped: `results["documents"][0]` is your result list. `results["documents"]` is a list-of-lists. Forgetting `[0]` passes nested lists to the prompt builder.

**Why the filter lives in the method, not the caller** — `tenant_id` is a required positional arg on `search_tenant_context()`. The Python interpreter won't compile a call that omits it. Isolation is enforced in the type signature, not documentation.

---

### Task 3.5: Isolation Testing

**`EphemeralClient` over mocks** — unit tests with `MagicMock` prove the filter argument is passed; they can't prove ChromaDB's filter actually evaluates correctly. `EphemeralClient` runs the real HNSW engine in-process. No Docker needed.

**Patching `HttpClient` to return `EphemeralClient`:**
```python
ephemeral = chromadb.EphemeralClient()
with patch("app.vector_db.chromadb.HttpClient", return_value=ephemeral):
    manager = VectorStoreManager(collection_name="test_isolation")
```
`VectorStoreManager.__init__` calls `chromadb.HttpClient(host, port)` — which now returns the ephemeral instance. Everything downstream runs against real ChromaDB.

---

### Task 3.6: Document Deletion

**Always use compound filter for delete** — `where={"file_hash": hash}` alone crosses tenant boundaries. If two tenants uploaded the same file, one tenant's delete wipes the other's copy. Always scope with `$and`:
```python
{"$and": [{"file_hash": {"$eq": hash}}, {"tenant_id": {"$eq": tid}}]}
```

**`$and` requires explicit `$eq`** — the shorthand `{"field": "value"}` only works for a single field. Two or more conditions require the operator form.

**`collection.delete()` returns `None`** — no count of rows removed. If you need the count, do a `collection.get(where=filter, include=[])` first.

---

## Phase 4 — Asynchronous Processing

### Task 4.1: Celery Setup

Broker stores task messages; result backend stores task state/return values. Both point to the same Redis URL here.

**JSON serialization over pickle** — pickle can execute arbitrary code on deserialisation. A malicious task message on a shared broker could compromise the worker. `task_serializer="json"` restricts to basic types only.

**`task_track_started=True`** — without this, task state jumps directly from `PENDING` to `SUCCESS/FAILURE`. With it, Celery writes a `STARTED` record the moment the worker picks up the job — gives the polling endpoint a third meaningful state.

---

### Task 4.2: Singleton Worker Resources

Module-level `embedder` and `vector_store` are initialised once when the worker process starts, not on each task:

| Placement | Load cost per task |
|---|---|
| Inside task function | ~1–2s (model reload every time) |
| Module level | ~0ms (loaded once at startup) |

**`conftest.py` must patch before any test module imports `app.worker`** — `VectorStoreManager.__init__` calls `chromadb.HttpClient` at import time. The fix is a module-level patch in `conftest.py`, started before pytest collects tests:
```python
patch("chromadb.HttpClient", return_value=mock_instance).start()
```
Individual tests can apply narrower patches on top; they take precedence within their scope.

---

### Task 4.3: Ingestion Pipeline

```
FastAPI                              Celery worker
  │──ingest_document_pipeline.delay()──▶│
  │         (Redis queue)                ├─ extract_pages()     → list[{page, text}]
  │                                      ├─ chunk_pages()       → list[{text, page, idx}]
  │                                      ├─ embed_batch()       → list[list[float]]
  │                                      ├─ insert_chunks()     → int (rows inserted)
  │                                      └─ os.unlink(path)     ← always, via finally
```

**`try / finally` for cleanup** — if the pipeline raises mid-way, `finally` still deletes the temp file. Without it, crashes leave files on disk permanently.

**Empty PDF guard** — `collection.add(ids=[], ...)` raises `ValueError`. A blank or image-only PDF produces zero chunks. Guard with `if chunks:` before calling embed and insert.

**Explicit task name** — `@celery_app.task(name="enterprise_rag.ingest_document_pipeline")`. Without this, Celery derives the name from the module path. Renaming or moving the module breaks in-flight tasks in the queue silently.

**Patching module-level singletons in tests:**
```python
with patch("app.worker.embedder", mock_embedder):
    ingest_document_pipeline(...)
```
The target is `app.worker.embedder` (the attribute on the module), not `app.embedder.LocalEmbedder` (the class). The task function looks up `embedder` in the module's global namespace.

---

### Task 4.4: Worker Smoke Test

**`task_always_eager=True`** — makes `.delay()` execute synchronously in the calling process. No Redis needed. The returned `AsyncResult` still has `.state`, `.get()` etc., so assertions work identically to real async dispatch.

**`task_eager_propagates`** — leave it `False` (the default). With `True`, exceptions from the task re-raise at `.delay()`, making it impossible to assert `result.state == "FAILURE"` before calling `.get()`.

**`AsyncResult` API reference:**
```python
result = task.delay(args)
result.state         # PENDING | STARTED | SUCCESS | FAILURE
result.successful()  # True if SUCCESS
result.get()         # return value, or re-raises on FAILURE
```
`AsyncResult(task_id)` constructed from a string is how the status-polling endpoint (Task 5.9) reads state from Redis.

---

### Task 4.5: Idempotent Ingestion Guard

**Why two layers** — the FastAPI layer checks `document_exists()` before enqueuing (cheapest path). But two concurrent uploads can both pass that check before either finishes indexing — a race window. The Celery layer re-checks at task start and exits immediately if already indexed.

**`collection.get()` not `collection.query()`** — `query()` requires an embedding vector (expensive). `get(where=filter, limit=1, include=[])` is a metadata-only lookup with no vector math. Returns just the `ids` key, no payload.

**`MagicMock` is truthy** — `mock_vs.document_exists(...)` returns a `MagicMock`, which is truthy. The idempotency guard fires immediately and skips the pipeline. Always set `mock_vs.document_exists.return_value = False` in fixtures where you want the full pipeline to run.

---

## Phase 5 — FastAPI Web Gateway

### Task 5.1: FastAPI Bootstrap

Three error handlers normalise all responses to `{"error": "..."}`: `HTTPException` (explicit raises), `RequestValidationError` (Pydantic validation failures → 422), and a bare `Exception` catch-all (500). FastAPI's default 422 body exposes Pydantic internals — overriding it gives clients one consistent shape.

---

### Task 5.2: File Upload Endpoint

**PDF magic bytes** — every valid PDF file begins with the bytes `%PDF` (e.g. `%PDF-1.7`). This is the file's true identity, independent of its filename or the MIME type the client claims. Two-layer check:

| Check | Catches |
|---|---|
| `content_type == "application/pdf"` | Clients that didn't set the MIME type |
| `data.startswith(b"%PDF")` | Files renamed to `.pdf` but actually text, images, etc. |

Either failing alone rejects the upload. MIME type is client-controlled and trivially spoofed; magic bytes are ground truth.

**`Form(...)` not `Body(...)` for companion fields** — multipart requests don't carry a JSON body. Any non-file field alongside `File(...)` must use `Form(...)`.

---

### Task 5.3: SHA-256 Fingerprinting

`hashlib.sha256(data).hexdigest()` produces a 64-char hex string. Naming the temp file `<hash>.pdf` means identical uploads map to the same path — the second upload hits the `if not path.exists()` guard and skips the write. The same hash is then passed to the Celery task, where Task 4.5's guard deduplicates at the ChromaDB layer.

---

### Storage Layer: Strategy Pattern + Dependency Injection

**Strategy Pattern** — define a family of interchangeable behaviours behind one interface; swap them without touching the code that uses them.

```python
# Contract (what callers depend on)
class StorageBackend(ABC):
    def save(self, file_hash: str, data: bytes) -> str: ...
    def exists(self, file_hash: str) -> bool: ...

# Strategy A — active today
class LocalStorage(StorageBackend):
    def save(self, file_hash, data): ...   # writes to /tmp/<hash>.pdf

# Strategy B — plug in tomorrow
class S3Storage(StorageBackend):
    def save(self, file_hash, data): ...   # uploads to s3://bucket/<hash>.pdf
```

**Dependency Injection via `Depends`** — the route declares what it needs, not how to get it. FastAPI resolves the concrete implementation at request time:

```python
@router.post("/upload")
async def upload_document(
    ...
    storage: StorageBackend = Depends(get_local_storage),  # ← injected
):
    locator = storage.save(file_hash, data)  # doesn't know if it's local or S3
```

To switch to S3: one line change in `main.py` (`Depends(get_s3_storage)`). The route handler is untouched.

**Why this matters in tests** — `app.dependency_overrides` lets you inject a `MagicMock` in place of any backend, without monkey-patching or touching global state:

```python
app.dependency_overrides[get_local_storage] = lambda: mock_storage
```

> **Interview angle:** *"How do you make infrastructure swappable without rewriting business logic?"* — Strategy pattern for the contract, dependency injection for wiring. The handler depends only on the ABC; which concrete class runs is decided at the composition root (`main.py`). Tests exploit the same seam via `dependency_overrides`.

---

### Task 5.5: RAG Query Endpoint

**`@lru_cache(maxsize=1)` for FastAPI singletons** — `LocalEmbedder` loads ~80 MB of model weights on construction. Wrapping the provider function with `@lru_cache(maxsize=1)` means the first `Depends(get_embedder)` call pays the construction cost; every subsequent call returns the cached instance instantly. This is the standard FastAPI pattern for expensive singletons.

| Approach | Cost per request |
|---|---|
| `Depends(lambda: LocalEmbedder())` | ~2s (model reload each request) |
| `Depends(get_embedder)` with `@lru_cache(maxsize=1)` | ~0ms after warm-up |

**`dependency_overrides` for clean testing** — `app.dependency_overrides[get_embedder] = lambda: mock_embedder` replaces the provider at the FastAPI level. The route handler never imports or references the real class in tests. Always `.clear()` overrides in `finally` or `yield` to avoid state leaking across tests.

**Pydantic v2 `@field_validator` puts `ValueError` in `ctx`** — when a validator raises `ValueError("msg")`, Pydantic v2 stores the exception object itself in the error's `ctx` dict: `{'ctx': {'error': ValueError(...)}}`. That `ValueError` is not JSON-serializable. The fix in the exception handler: stringify any non-native ctx values before passing to `JSONResponse`.

**Tenant isolation is structural, not optional** — `search_tenant_context` takes `tenant_id` as a required positional argument; there is no version of the method that skips the `where=` filter. Isolation cannot be accidentally omitted by a future developer.

> **Interview angle:** *"How does your query endpoint prevent cross-tenant data leaks?"* — Two layers: (1) the `where={"tenant_id": tenant_id}` filter is baked into `search_tenant_context` — callers cannot bypass it, and (2) the Pydantic validator rejects blank `tenant_id` values so the filter is never invoked with an empty string that might match broadly.

---

### Task 5.6: Grounded LLM Synthesis

**`temperature=0.0` is the RAG contract** — at temperature 0 the model is fully deterministic and picks the single highest-probability token at each step. For RAG, the model's job is extraction and compression, not creativity. Higher temperature would let it drift away from the source text.

**System prompt design for grounding** — the prompt must do three things:
1. Restrict the model to the provided context (`"ONLY the context excerpts below"`)
2. Give it a precise fallback phrase for missing answers — makes behaviour testable
3. Label each chunk by page number so the model can cite its source if needed

```
[Page 1] ...chunk text...
[Page 3] ...chunk text...
```

**`AsyncOpenAI` vs `OpenAI`** — FastAPI route handlers are `async def`, so they run in an event loop. Using the sync `OpenAI` client inside an async handler blocks the event loop for the duration of the HTTP call — all other requests queue up. `AsyncOpenAI` wraps `httpx.AsyncClient` and `await`s the call, releasing the event loop while the network round-trip completes.

**Mocking async OpenAI with `AsyncMock`** — `AsyncMock` (from `unittest.mock`) is needed because `client.chat.completions.create(...)` is a coroutine. `MagicMock` would return a regular object, and `await mock_client.chat.completions.create(...)` would raise `TypeError: object MagicMock can't be used in 'await' expression`. Rule: any `async def` method must be mocked with `AsyncMock`.

**Service layer placement** — synthesis logic (`build_context_block`, `synthesise`) lives in `app/services/synthesis.py`, not in the router. The router becomes a thin orchestrator: call embedder → call vector store → call synthesiser → return response. Each layer is independently testable.

> **Interview angle:** *"Why use a strict system prompt instead of trusting the model to stay grounded?"* — LLMs are trained to be helpful and will fill gaps with plausible-sounding information even when asked not to. A system prompt that says "ONLY use the provided context" and specifies an exact fallback phrase shifts grounding from the model's implicit behaviour to an explicit instruction it can be tested against. It also makes the fallback phrase machine-checkable — you can assert on it in integration tests.

---

### Task 5.7: Health Check Endpoint

**`/health` is top-level, not versioned** — Kubernetes readiness and liveness probes are configured at the infra layer, not by API clients. They shouldn't be coupled to an API version (`/api/v1/health`). If you rev the API, the probe URL stays stable.

**503 vs 200 for unhealthy state** — Kubernetes only looks at the HTTP status code for probe results. Returning 200 with `{"redis": "error"}` in the body would cause k8s to think the pod is healthy and keep routing traffic to it. Returning 503 removes the pod from the load balancer until it recovers.

**Catch-all `except Exception`** — health checks must never propagate exceptions to the caller. If Redis raises `AuthenticationError` or ChromaDB raises `ValueError`, the endpoint should report `"error: <detail>"` and return 503, not 500. The broad catch is intentional here.

**`redis.asyncio` vs `redis.Redis`** — the sync `redis.Redis.ping()` blocks the event loop for the duration of the TCP round-trip. In an `async def` handler this stalls all other requests. `redis.asyncio.Redis` is a drop-in async replacement; `await client.ping()` releases the event loop during the wait.

**`decode_responses=True`** — without this, `redis.asyncio` returns raw `bytes` (e.g. `b"PONG"`). Setting it to `True` automatically decodes responses to `str`, which is what you want for most application-layer use.

> **Interview angle:** *"How does your health check help with zero-downtime deployments?"* — A k8s readiness probe hitting `/health` ensures a new pod only receives traffic once both Redis and ChromaDB connections are confirmed live. If a deployment starts but ChromaDB is unreachable, the probe returns 503, the pod stays out of the load balancer rotation, and the old pods keep serving — no user-visible outage.

---

### Task 5.8: Streaming Query Response

**Why stream at all** — a non-streamed `gpt-4o-mini` completion for a multi-paragraph answer can take several seconds with nothing sent to the client until the very end. Streaming sends each token the moment OpenAI emits it, so the client can start rendering immediately — same total latency, far better perceived latency.

**`stream=True` changes the return type, not just the timing** — `await client.chat.completions.create(..., stream=True)` no longer returns a single `ChatCompletion`; it returns an `AsyncStream` you iterate with `async for`. Each chunk is a `ChatCompletionChunk` where the text lives at `chunk.choices[0].delta.content` (not `.message.content`) and is often `None` (e.g. the first chunk just announces the role) — those must be filtered out or you leak empty-string tokens.

**Retrieval happens before the generator, not inside it** — `embedder.embed()` and `search_tenant_context()` still run as plain synchronous code in the route body, *before* `StreamingResponse` is constructed. This is deliberate: if there are no matching documents, the route can still raise `HTTPException(404)` normally. Once a `StreamingResponse` starts sending bytes, the HTTP status code is already committed to `200` — you can't downgrade it to a 404 mid-stream.

**Server-Sent Events (SSE) as the wire format** — `media_type="text/event-stream"` with each chunk framed as `data: {json}\n\n` is the standard way to stream structured events over plain HTTP (no WebSocket upgrade needed, works through most proxies). This endpoint sends three event types in order:

```
data: {"type": "context", "chunks": [...]}   ← sources, sent once, up front
data: {"type": "token", "content": "Tenant"} ← repeated per token
data: {"type": "done"}                        ← terminal event
```

Sending `context` first lets the UI show "answering from page 1, 2..." before the answer text starts arriving.

**Testing a stream with `TestClient`** — Starlette's `TestClient` fully drains the async generator synchronously and hands back the concatenated bytes as `response.text`. Tests don't need real async streaming infrastructure — just split on `\n\n`, strip the `data: ` prefix, and `json.loads` each event, same as a real SSE client would.

> **Interview angle:** *"Why not just return the full answer as JSON like before?"* — Time-to-first-byte vs. total latency. A blocking JSON response makes the user stare at nothing until the entire answer is generated. Streaming trades a slightly more complex client (needs an SSE/event parser instead of `response.json()`) for the answer appearing token-by-token, which is the UX users now expect from LLM products. The context chunks are still sent — just as the first event instead of a response field — so source attribution isn't lost.

---

### Task 5.9: Task Status Endpoint

**`AsyncResult` is a lookup, not a subscription** — `AsyncResult(task_id, app=celery_app)` doesn't ask "does this ID exist?"; it just reads whatever key currently sits in the result backend (Redis) under that ID. An ID that was never dispatched and an ID for a task still queued both read back identically as `PENDING` — Celery has no way to distinguish "unknown" from "not started yet."

**Splitting `celery_app.py` out of `worker.py`** — `worker.py` builds `LocalEmbedder()` and `VectorStoreManager()` at module level (Task 4.2's lazy-loading pattern), so simply `from app.worker import celery_app` in the FastAPI process would silently load the ~80 MB embedding model and open a ChromaDB connection just to answer a status-check GET request. Moving the bare `Celery(...)` instance + config into `app/celery_app.py` lets both processes import the app object without either one paying for the other's heavy singletons.

**Task config is bound once, then frozen** — Celery copies settings like `task_track_started` and `task_store_eager_result` onto the `Task` class the *first time it's bound to the app* (`Task.bind()`, which runs at `@celery_app.task` decoration time — i.e. module import). `Task.bind()` only applies a setting if the attribute is currently `None`; after that first import it's permanently set, and later `celery_app.conf.update(...)` calls have no effect on already-bound tasks. This bit a test directly: toggling `task_store_eager_result=True` inside a test fixture did nothing, because `ingest_document_pipeline` had already bound to the app (as `False`) the moment `tests/test_tasks.py` imported `app.worker`. Fix: set it once, permanently, in `app/celery_app.py`'s initial config.

**Why `task_store_eager_result` matters at all** — with `task_always_eager=True` (used in tests to run tasks synchronously, no broker needed), `.delay()` returns an `EagerResult` whose `.state`/`.result` are correct in memory — but by default Celery does *not* also write that result to Redis, since eager mode assumes you'll just read the object it handed back. This endpoint does a **separate** `AsyncResult(task_id)` lookup later, which only finds anything if eager tasks are told to persist to the backend too. The setting only has any effect when eager mode is on, so it's a no-op in the deployed worker (which always dispatches through the real broker).

**`.result` means different things depending on `.state`** — on `SUCCESS`, `.result` is the task's return value (here, the ingestion summary dict). On `FAILURE`, `.result` is the *exception instance* Celery caught, not a dict — the endpoint must `str()` it before it can go into a JSON response. Reading `.result` on `PENDING`/`STARTED` is `None` — there's nothing to report yet.

> **Interview angle:** *"How would a client know when an async job is done?"* — Polling `GET /tasks/{id}` and checking `status` is the simplest pattern: no persistent connection, works through any HTTP infra, and the client controls its own polling interval. The tradeoff vs. streaming/WebSocket push is latency (the client finds out on its next poll, not the instant it happens) for much simpler infrastructure — no open connections to manage across a job that might take minutes.

---

### Task 5.4: Async Task Dispatch

**Dispatch by name, not by import** — the upload route calls `celery_app.send_task("enterprise_rag.ingest_document_pipeline", args=[...])` instead of `from app.worker import ingest_document_pipeline; ingest_document_pipeline.delay(...)`. `send_task` only needs the task's registered *name* (a string) to serialize a message onto the broker; it never imports `worker.py`. Importing `worker.py` from the FastAPI process would trigger its module-level `LocalEmbedder()` and `VectorStoreManager()` singletons (Task 4.2) — paying for an 80 MB model load and a ChromaDB connection in a process that never runs the task body. This is the same reason `celery_app.py` was split out of `worker.py` back in Task 5.9.

**Idempotency check moves earlier, not just deeper** — Task 4.5 already guards inside the task body, but checking `vector_store.document_exists()` in the route *before* `storage.save()` and dispatch means a re-upload of an already-ingested file skips the disk write and the broker round-trip entirely, not just the embedding work. Two guards, two different costs saved.

**`send_task` returns an `AsyncResult`-shaped object without needing the task class** — `celery_app.send_task(...).id` is enough to hand back to the client for polling via the Task 5.9 status endpoint; the dispatching process never needs to know what the task actually does.

> **Interview angle:** *"Why not just import the Celery task function directly and call `.delay()`?"* — That's the natural first instinct, but it silently couples your API process to your worker process's dependencies. `send_task(name, args=...)` is the decoupled version: the API only needs to agree with the worker on a task *name* and argument *shape*, not share Python imports. This is also what lets the API and worker be deployed as separate services with different dependency sets in production.

---

## Phase 6 — System Validation & Interview Demos

### Task 6.1: Full Stack Startup

**A "healthy" container can still be functionally broken** — ChromaDB's Docker healthcheck only curls `/api/v1/heartbeat`, which returns 200 as soon as the HTTP server accepts connections. It says nothing about whether the underlying SQLite schema actually initialized. In this session the persisted `chroma_data/chroma.sqlite3` was 0 bytes while the container reported `Up (healthy)` for 12 days — every query failed with `OperationalError('no such table: tenants')` the moment a real client (the Celery worker, at `VectorStoreManager()` construction) tried to validate its tenant. A plain `docker compose restart chroma_server` was enough to re-run initialization against the existing (empty) volume and fix it. Lesson: a healthcheck only proves what it actually checks — a heartbeat endpoint proves the process is up, not that the database underneath it works.

**Celery's prefork pool crashes on macOS when native ML libraries are already loaded** — starting `celery worker` with the default `--pool=prefork` (concurrency=10) segfaulted immediately (`Worker exited prematurely: signal 11 (SIGSEGV)`) as soon as a real task ran. Cause: `worker.py` loads `sentence-transformers`/PyTorch at module level (Task 4.2), *before* Celery forks its child worker processes; forking a process that has already initialized certain native/Accelerate-backed libraries is unsafe on macOS and corrupts child process memory. Fix: run with `--pool=solo` (single process, no fork) for local dev. In a Linux container deployment this usually isn't an issue, but it's a very common local-dev gotcha with Celery + PyTorch on macOS.

**Port collisions are process-specific, not just service-specific** — `uvicorn --port 8001` "succeeded" (no bind error, log said "Uvicorn running on http://0.0.0.0:8001") even though another unrelated project's dev server was already bound to `127.0.0.1:8001`. Binding to the wildcard address (`0.0.0.0`) and binding to a specific address (`127.0.0.1`) can coexist on the same port on macOS; whichever listener matches most specifically wins for a given request. `curl localhost:8001` was silently being served by the *other* process the whole time — `lsof -nP -i :8001` was what actually revealed two different PIDs holding the same port.

> **Interview angle:** *"How do you know your local dev stack is actually healthy, not just 'up'?"* — Container/process status and health checks tell you a process is running and responding to its own shallow probe. Actually proving the stack works means exercising it end-to-end: dispatch a real task through the real broker, hit the real DB with a real query, and check the response — which is exactly what Tasks 6.2–6.4 do next.

---

### Task 6.2: Upload Smoke Test

**The smoke test is the first real proof the pieces are wired together** — every earlier task tested its own layer in isolation with mocks (parser mocked in worker tests, ChromaDB mocked in unit tests). Uploading a real PDF through the actual HTTP endpoint and polling the actual task status endpoint is the first time the full chain — upload → hash → dispatch → worker → parse → chunk → embed → insert → task result — runs with no mocks at all, end to end.

**Idempotency guard is also a smoke test** — re-uploading the exact same file after it succeeded returns `{"status": "already_exists", "file_hash": ...}` immediately, without dispatching a second task — free confirmation that Task 4.5's guard (checked in the route this time, per Task 5.4) actually fires on real data.

---

### Task 6.3: Query Verification

**Retrieval and generation are separable failure points** — when the OpenAI API key in `.env` turned out to be invalid (401 `AuthenticationError`), the SSE stream still successfully emitted the `context` event with correctly-retrieved, relevant chunks — the failure was entirely in the synthesis half. This is a direct consequence of Task 5.8's design: retrieval happens as plain code before the generator starts, so a downstream synthesis failure doesn't corrupt or hide the fact that retrieval worked. It also means "the answer is wrong" and "the answer is missing" are diagnosable separately by looking at the `context` event first.

**Swapping LLM providers only touched the `base_url`** — Gemini exposes an OpenAI-compatible endpoint at `https://generativelanguage.googleapis.com/v1beta/openai/`. Because `app/services/synthesis.py` already depended on the generic `openai.AsyncOpenAI` client shape (not an OpenAI-specific detail), moving from OpenAI to Gemini required zero changes to the streaming/synthesis code — only `app/dependencies.py`'s client construction (new `api_key` + `base_url`) and `app/config.py`'s field names (`gemini_api_key`, `gemini_model`, `gemini_base_url`). This is the practical payoff of coding against the OpenAI wire protocol as a de facto standard rather than a vendor SDK feature.

> **Interview angle:** *"How would you design an LLM integration to be provider-agnostic?"* — Depend on the OpenAI request/response shape (which most providers now mirror via a compatibility layer — Gemini, and others) rather than a vendor SDK's unique features. Keep the API key and base URL in config, not code. The real test of this design showed up here: a dead API key forced a live provider swap, and the fix was two config fields, not a rewrite.

---

### Task 6.4: Multi-Tenant Isolation Audit

**The isolation guarantee is provable at the retrieval layer, not just the response** — asking the identical question under `tenant_id="globex"` (which only has the Constitution of India ingested) returned SSE `context` chunks exclusively from that document — zero ISO 27001 content — and the LLM correctly answered "I don't have enough information," because `search_tenant_context`'s `where={"tenant_id": ...}` filter (Task 3.4) never handed it any ISO 27001 chunks to begin with. Checking the retrieved *chunks*, not just the final answer, is what actually proves isolation — a model could theoretically still produce a safe-looking refusal even if leaked context had been on the way. Repeating the test in the reverse direction (Constitution-specific question under `tenant_id="acme"`) confirmed the guarantee holds both ways.

> **Interview angle:** *"How would you prove to an auditor that your multi-tenant RAG system doesn't leak data across tenants?"* — Don't just show that answers look correct; show the retrieved context chunks per tenant and prove the `where=` filter is unconditional (Task 3.4's design — `tenant_id` is a required parameter with no bypass). A reproducible test that queries the same question across two tenants with disjoint document sets and inspects the raw retrieved chunks is stronger evidence than trusting the LLM's phrasing.

---

### Task 6.5: Retrieval Quality Evaluation

**A cheap correctness proxy avoids needing an LLM to grade retrieval** — instead of asking Gemini to judge whether each answer was "correct" (expensive, and couples a retrieval experiment to a second component's reliability), `scripts/eval_retrieval_quality.py` checks whether a keyword known to appear near the real answer in the source PDF shows up anywhere in the retrieved chunks' text. It's a blunt instrument (a keyword match isn't the same as semantic correctness) but it's fast, deterministic, and isolates the retrieval layer from the generation layer.

**The precision/recall tradeoff is directly observable, not just theoretical** — running the same 5 questions against the same ingested document at `n_results=3` vs `n_results=8` produced a real flip: "What does Annex A contain?" missed its expected keyword at n=3 (its correct chunk ranked 4th–8th by cosine distance) and hit at n=8. That's recall improving with a larger `n_results` — at the direct cost of sending more, and more marginally-relevant, context to the LLM (visible in the widening distance range: 1.16–1.41 at n=3 vs 1.16–1.47 at n=8).

**Distance range width is a rough relevance signal** — a tight distance range (e.g. Q2's 0.50–0.85) suggests all returned chunks are genuinely close to the query; a wide range (Q3, Q5: spans over 0.3) suggests the tail results are much less relevant than the top hit and are along for the ride mostly to hit the `n_results` count. This isn't a formal metric but it's a quick way to eyeball whether a larger `n_results` is adding useful context or just noise for a given query.

> **Interview angle:** *"How do you choose `n_results` for a RAG system?"* — It's a precision/recall tradeoff with a real cost on both sides: too small and you risk missing the answer entirely (recall failure, unrecoverable — the LLM can't cite what it never saw); too large and you dilute the context with less-relevant chunks, which can distract the LLM, increase token cost, and in extreme cases (very long context) directly hurt answer quality. The right value depends on how densely the answer tends to be represented across chunks — measuring hit-rate at different `n_results` on a small golden set of known-answer questions is a lightweight way to tune it empirically instead of guessing.

---

## Phase 7 — Demo & Interview Readiness

### Task 7.1: Streamlit UI

**A Streamlit script becomes unit-testable by guarding page rendering behind `if __name__ == "__main__"`** — `streamlit run ui.py` executes the file with `__name__ == "__main__"` (same as any script run directly), but `import ui` from a test module does not. Putting every `st.*` call inside `main()` (plus `_handle_upload`/`_handle_question` helpers) and only calling `main()` under that guard means the HTTP-calling helper functions (`upload_document`, `get_task_status`, `stream_query`, `parse_sse_line`) can be imported and tested with plain mocks — no Streamlit session, no `AppTest` harness needed for this level of coverage.

**`ui.py` deliberately has zero imports from `app/`** — it talks to the FastAPI gateway over plain HTTP (`requests`), exactly like any external client would. Importing `app.main` directly would have been easier to wire up, but it would also silently pull in Celery/ChromaDB/the embedding model into the demo UI's process — the same "don't import the heavy module just to reuse one thing" lesson as Task 5.4's `send_task`-by-name pattern, applied one layer further out.

**Consuming an SSE stream from a synchronous script is just line-splitting** — `requests.post(..., stream=True)` + `response.iter_lines()` yields the wire-format lines exactly as the server sent them (`data: {...}` frames separated by blank lines). No SSE client library needed — `parse_sse_line()` just checks for the `data:` prefix, strips it, and `json.loads`s the rest; blank separator lines return `None` and get skipped by the caller.

**Discovered while manually verifying this task — not fixed, flagging for later:** re-ingesting a byte-identical PDF under a *different* `tenant_id` silently fails to create separate rows. `VectorStoreManager.insert_chunks` builds each row's ID as `f"{file_hash}_{page_number}_{chunk_index}"` (Task 3.3) — purely content-derived, with no `tenant_id` in the ID. Two tenants uploading the same file produce identical IDs, and ChromaDB's `collection.add()` does not upsert on a duplicate ID (unlike `.upsert()`) — it appears to silently keep the first tenant's row and drop the new insert, even though the Celery task still reports `{"status": "ingested", "chunks_inserted": 26}` (that count is just `len(chunks)` client-side, not a server-confirmed write count). Net effect: the second tenant's upload looks successful end-to-end (task reaches `SUCCESS`) but its document is not actually queryable under its own `tenant_id` — the isolation guarantee tested in Task 6.4 only holds when tenants never upload identical file bytes. Fix would be including `tenant_id` in the composite ID (e.g. `f"{tenant_id}_{file_hash}_{page_number}_{chunk_index}"`) or switching `add()` to `upsert()` plus a tenant-aware existence check.

> **Interview angle:** *"Your idempotency/dedup key is derived from file content (a hash) — what happens when two different tenants upload the same file?"* — If the dedup key doesn't also incorporate the tenant, a content-addressed ID scheme silently collapses two tenants' rows into one on write, which is a much quieter failure than a query-time leak: nothing errors, the task reports success, and the bug only surfaces when the second tenant tries to actually query their data. The fix is cheap (namespace the ID by tenant) but the lesson is general: any identifier derived purely from *what* was uploaded, with no *who* uploaded it, needs an explicit argument for why cross-tenant collisions are safe — "it happened to work in every test so far" isn't one.

---

### Task 7.2: Architecture Diagrams + Code Comment Pass + README

**Diagrams rot the moment an underlying dependency swaps** — `architecture.md` (written back in Phase 1–3) still said `OpenAI`/`gpt-4o-mini` and `/tmp staging` after the project had already moved to Gemini (Task 6.3) and a pluggable `StorageBackend` (Task 5.2/5.3's later refactor). Nothing broke — the diagram just quietly stopped matching the code. A diagram is documentation, not a one-time deliverable; it needs the same "does this still match reality" check as any comment referencing specific behavior.

**Verifying a diagram means rendering it, not just eyeballing the text** — Mermaid's sequence-diagram syntax will happily *look* right in a text editor while being subtly broken (an unescaped character, a mismatched `alt`/`end`). Running each diagram block through `mmdc` (mermaid-cli) and actually looking at the rendered PNG caught this before it shipped — the same "run it, don't just read it" principle as verifying the Streamlit UI in Task 7.1.

**A documentation pass on an already-well-commented codebase mostly confirms, not adds** — auditing every file in `app/` for missing WHY-comments found that routers, services, and worker/config modules were already thoroughly commented from when they were originally built (Tasks 3–6 consistently explained non-obvious decisions inline as they were written). The actual gap was the two trivial-looking Pydantic schema files (`schemas/query.py`, `schemas/tasks.py`) — plain field lists with validators, no comments at all, because "just a data model" doesn't obviously call for a WHY-comment even when a validator or field encodes a real design decision (e.g. why blank strings are rejected before they become confusing 404s downstream, or why `result: Any` is deliberately untyped).

> **Interview angle:** *"How do you keep architecture docs from going stale as the system evolves?"* — Treat them like code that references specific behavior: whenever a dependency, provider, or abstraction changes (LLM provider swap, storage backend added), grep the docs for the old name/assumption instead of assuming diagrams age independently of the code. And verify diagrams the same way you'd verify code — render them, don't just proofread the source.

**A README's job is to be the *only* doc a stranger reads before deciding whether to keep reading** — this project already had `RUNNING.md` (per-process run steps), `architecture.md` (diagrams + rationale), and `features.md` (full spec) before `README.md` existed at all. The temptation with a README written last is to re-explain everything those files already cover; the more useful version is a compressed front door — what the project is, why it's shaped this way, the one quickstart path, and then a table of links out — so each existing doc stays the single source of truth for its own topic instead of drifting out of sync with a duplicate copy in the README.

**Documenting a known bug in the README is a design choice, not an admission of failure** — the composite-ID tenant-collision issue (Task 7.1) already lived in `learning.md` and `RUNNING.md`'s "known issue" section; surfacing it again in the README's own "Known limitations" section means someone evaluating the project from the README alone (the most likely entry point) sees it before they hit it themselves in a demo, rather than discovering it looks like an unexplained bug.

> **Interview angle:** *"What belongs in a README versus a separate architecture doc?"* — A README optimizes for a five-minute first read: what/why, one quickstart path, and links onward. Depth (sequence diagrams, full API validation rules, per-task history) belongs in dedicated docs that a README should point to, not absorb — duplicating content across both is what makes docs go stale, because a change now has two places to update and only one gets remembered.

---

### Ad hoc: Dependency Construction Failures Should Not Look Like Route Bugs

**`Depends()` resolves before the route body runs, so a raising provider never lets the endpoint execute** — FastAPI builds every `Depends()` argument first; if one raises, the route function is never entered. Before this fix, `get_vector_store()` ([dependencies.py](app/dependencies.py)) constructing `VectorStoreManager()` — which opens an HTTP connection to ChromaDB in `__init__` — would let a raw `ConnectionError` (or similar) bubble past FastAPI's default handling into `main.py`'s catch-all `Exception` handler, returning a generic `{"error": "Internal server error"}` 500. That's indistinguishable from an actual application bug (bad logic, unhandled edge case) even though the real cause is an external dependency being down.

**Fix: catch the construction failure inside the provider and re-raise as `HTTPException(503)`** — because FastAPI dependencies are allowed to raise `HTTPException` directly (it's treated exactly like a route raising one), wrapping `VectorStoreManager()` in a `try/except Exception` inside `get_vector_store()` and re-raising `HTTPException(503, detail="Vector store is unavailable")` gives callers a clean, correctly-coded signal ("the service is temporarily down, retry") instead of a 500 ("something is broken, file a bug").

**`lru_cache` only memoizes successful returns, never a raised exception** — this matters because it means the fix doesn't need any manual cache-invalidation logic: if construction fails once, the next request calls the provider fresh and can succeed as soon as ChromaDB is reachable again. Verified directly in [tests/test_dependencies.py](tests/test_dependencies.py) by patching `VectorStoreManager` to raise on one call and return a fake instance on the next, clearing `get_vector_store.cache_clear()` between tests so no prior test's cached instance leaks in.

> **Interview angle:** *"What's the difference between a 500 and a 503, and where should that distinction be enforced?"* — A 500 says "the request revealed a bug in this code"; a 503 says "the request was fine, but something this code depends on is unavailable right now." FastAPI's dependency-injection model makes this cheap to enforce correctly: since a raising `Depends()` provider never lets route logic run, catching *at the provider* — not wrapping the whole route in a generic `try/except` — is both more precise (only the specific network-backed construction is guarded) and reusable across every route that shares the dependency.

---

### Ad hoc: CI via GitHub Actions — No Service Containers Needed

**A test suite's mocking choices determine whether CI needs live infrastructure at all** — the instinct when adding CI to a project with Redis + ChromaDB dependencies is to reach for GitHub Actions' `services:` block (spins up service containers alongside the job). That would have worked but wasn't necessary here: `tests/conftest.py` already patches `chromadb.HttpClient` at module level for the entire suite (Task 4.2's note about `app.worker` instantiating `VectorStoreManager` at import time), and the one test that needs a real Redis (`test_worker.py::test_redis_broker_is_reachable`) is written to `pytest.skip()` on `ConnectionError`/`TimeoutError` rather than fail. Checked this by grepping every test file for direct `HttpClient`/`redis.` usage before writing the workflow, instead of assuming — the workflow ended up being a plain "install deps, run pytest" with no `services:` block at all.

**A required-but-fake config value still needs to exist for imports to succeed** — `app/config.py`'s `Settings.gemini_api_key` has no default (Task 1.4's fail-fast design), so `from app.config import settings` — which most of `app/` imports transitively — raises `ValidationError` in a bare CI environment with no `.env`. None of the 262 tests make a live Gemini call, so the fix is a placeholder string set via the workflow's `env:` block (`GEMINI_API_KEY: ci-placeholder-key`), not a real GitHub Actions secret. Worth distinguishing "config the code requires to construct objects" from "config that needs to be real" — only the latter belongs in `secrets.*`.

> **Interview angle:** *"Your service has hard runtime dependencies (a DB, a broker) — how do you avoid needing to stand up all of that just to run CI?"* — Push the mocking boundary to where it naturally belongs (a `conftest.py` patch that fires before any module import, per the `app.worker` module-level-singleton problem from Task 4.2), and make integration tests that truly need live infra self-skip rather than fail when it's absent, instead of gating the whole suite behind `-m "not integration"` or a CI-only skip flag. The payoff shows up exactly here: adding CI didn't require choosing a service-container strategy or provisioning anything, because the test suite was already structured to not need it.
