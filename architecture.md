# Enterprise RAG — Architecture & Sequence Diagrams

---

## 1. System Architecture

High-level view of every component, its role, and how they connect.

```mermaid
flowchart TB
    subgraph Client Layer
        CL["Client\n(curl / Postman / Streamlit UI)"]
    end

    subgraph Gateway Layer
        API["FastAPI\nUvicorn :8001\nmain.py"]
    end

    subgraph Async Layer
        RD["Redis\n:6379\nBroker + Result Backend"]
        CW["Celery Worker\nworker.py"]
    end

    subgraph ML Layer
        EM["LocalEmbedder\nall-MiniLM-L6-v2\n384-dim vectors"]
    end

    subgraph Storage Layer
        FS["Local File Storage\n/tmp staging"]
        CH["ChromaDB\n:8000\nenterprise_knowledge_base"]
    end

    subgraph External
        OA["OpenAI API\ngpt-4o-mini\ntemperature=0.0"]
    end

    CL -->|"HTTP POST/GET/DELETE"| API
    API -->|"enqueue task"| RD
    API -->|"embed query"| EM
    API -->|"scoped similarity search"| CH
    API -->|"chat completion stream"| OA
    API -->|"health ping"| RD
    API -->|"health ping"| CH

    RD -->|"dispatch job"| CW
    CW -->|"read PDF"| FS
    CW -->|"batch embed chunks"| EM
    CW -->|"insert vectors + metadata"| CH
    CW -->|"write result state"| RD
    CW -->|"delete temp file"| FS
```

### Component Responsibilities

| Component | Technology | Responsibility |
|-----------|-----------|----------------|
| **FastAPI Gateway** | FastAPI + Uvicorn | HTTP routing, input validation, auth boundary, request fan-out |
| **Redis** | Redis (Alpine) | Celery message broker, task result backend, health check target |
| **Celery Worker** | Celery | Background ingestion: parse → chunk → embed → store |
| **LocalEmbedder** | sentence-transformers | Converts text to 384-dim vectors; loaded once per process |
| **ChromaDB** | ChromaDB (Docker) | Persistent vector store with metadata-filtered tenant isolation |
| **Local File Storage** | Filesystem `/tmp` | Temporary staging for uploaded PDFs during ingestion |
| **OpenAI API** | gpt-4o-mini | Context-grounded answer synthesis at query time |

---

## 2. Data Flow: Document Ingestion

The full lifecycle from HTTP upload to vectors persisted in ChromaDB.

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant FastAPI
    participant Redis
    participant CeleryWorker
    participant DocumentParser
    participant LocalEmbedder
    participant ChromaDB
    participant FileStorage

    Client->>FastAPI: POST /api/v1/docs/upload\n(file=doc.pdf, tenant_id=acme)

    FastAPI->>FastAPI: Validate file extension (.pdf only)
    FastAPI->>FastAPI: Compute SHA-256(file_bytes) → file_hash

    FastAPI->>ChromaDB: Query — does {file_hash, tenant_id} exist?
    ChromaDB-->>FastAPI: Not found

    FastAPI->>FileStorage: Save file as /tmp/{file_hash}.pdf
    FastAPI->>Redis: ingest_document_pipeline.delay(\n  path, tenant_id, file_hash\n)
    Redis-->>FastAPI: task_id
    FastAPI-->>Client: 202 Accepted\n{ task_id: "abc-123" }

    Note over Redis, CeleryWorker: Async — worker picks up job from queue

    Redis->>CeleryWorker: Dispatch ingest_document_pipeline task
    CeleryWorker->>FileStorage: Read /tmp/{file_hash}.pdf
    CeleryWorker->>DocumentParser: extract_pages(pdf_bytes)
    DocumentParser-->>CeleryWorker: [ page_text, ... ]

    CeleryWorker->>DocumentParser: chunk_pages(pages)\n400-word window, 50-word overlap
    DocumentParser-->>CeleryWorker: [ {text, page_number, chunk_index}, ... ]

    CeleryWorker->>LocalEmbedder: batch_embed([ chunk_text, ... ])
    LocalEmbedder-->>CeleryWorker: [ [float × 384], ... ]

    CeleryWorker->>ChromaDB: insert_chunks(\n  ids={hash}_{page}_{idx},\n  vectors, metadata={tenant_id, file_hash, page_number}\n)
    ChromaDB-->>CeleryWorker: OK

    CeleryWorker->>FileStorage: Delete /tmp/{file_hash}.pdf
    CeleryWorker->>Redis: Write result → SUCCESS
```

---

## 3. Data Flow: Semantic Query (RAG)

The full lifecycle from query string to streamed grounded answer.

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant FastAPI
    participant LocalEmbedder
    participant ChromaDB
    participant OpenAI

    Client->>FastAPI: POST /api/v1/query\n{ query: "What is...", tenant_id: "acme" }

    FastAPI->>FastAPI: Pydantic schema validation

    FastAPI->>LocalEmbedder: embed(query_string)
    LocalEmbedder-->>FastAPI: query_vector [float × 384]

    FastAPI->>ChromaDB: similarity_search(\n  vector=query_vector,\n  where={ tenant_id: "acme" },\n  n_results=5\n)
    ChromaDB-->>FastAPI: top-5 chunks\n[ {text, page_number, file_hash}, ... ]

    FastAPI->>FastAPI: Build system prompt:\n"Answer ONLY from context below.\n[chunk_1]\n[chunk_2]..."

    FastAPI->>OpenAI: chat.completions.create(\n  model=gpt-4o-mini,\n  temperature=0.0,\n  stream=True,\n  messages=[system_prompt, user_query]\n)

    loop Streaming tokens
        OpenAI-->>FastAPI: token chunk
        FastAPI-->>Client: StreamingResponse chunk
    end

    FastAPI-->>Client: [stream end]
```

---

## 4. Data Flow: Task Status Poll

How a client checks the outcome of an async ingestion job.

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant FastAPI
    participant Redis

    Client->>FastAPI: GET /api/v1/tasks/{task_id}

    FastAPI->>Redis: AsyncResult(task_id).state + .result

    alt Task still running
        Redis-->>FastAPI: state=PENDING
        FastAPI-->>Client: 200 { status: "PENDING" }
    else Task succeeded
        Redis-->>FastAPI: state=SUCCESS, result={...}
        FastAPI-->>Client: 200 { status: "SUCCESS", result: {...} }
    else Task failed
        Redis-->>FastAPI: state=FAILURE, traceback="..."
        FastAPI-->>Client: 200 { status: "FAILURE", error: "..." }
    end
```

---

## 5. Data Flow: Document Deletion

How a document and all its vectors are purged from the system.

```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant FastAPI
    participant ChromaDB

    Client->>FastAPI: DELETE /api/v1/docs/{file_hash}?tenant_id=acme

    FastAPI->>ChromaDB: collection.delete(\n  where={ tenant_id: "acme", file_hash: "{hash}" }\n)
    ChromaDB-->>FastAPI: Deleted N vectors

    FastAPI-->>Client: 204 No Content
```

---

## 6. Multi-Tenant Isolation Model

Illustrates why tenant data cannot bleed across boundaries — the filter is enforced at the vector store layer, not the application layer.

```mermaid
flowchart LR
    subgraph ChromaDB Collection: enterprise_knowledge_base
        direction TB
        V1["Vector\ntenant_id: acme\nfile_hash: abc\npage: 1"]
        V2["Vector\ntenant_id: acme\nfile_hash: abc\npage: 2"]
        V3["Vector\ntenant_id: globex\nfile_hash: xyz\npage: 1"]
        V4["Vector\ntenant_id: globex\nfile_hash: xyz\npage: 2"]
    end

    QA["Query from tenant: acme"]
    QB["Query from tenant: globex"]

    QA -->|"where={tenant_id: acme}"| V1
    QA -->|"where={tenant_id: acme}"| V2
    QA -. "blocked by filter" .-> V3
    QA -. "blocked by filter" .-> V4

    QB -->|"where={tenant_id: globex}"| V3
    QB -->|"where={tenant_id: globex}"| V4
    QB -. "blocked by filter" .-> V1
    QB -. "blocked by filter" .-> V2
```

---

## 7. Deployment Topology

How all processes run together locally.

```mermaid
flowchart TB
    subgraph Docker Compose
        R["redis_broker\nport 6379"]
        C["chroma_server\nport 8000\nvolume: ./chroma_data"]
    end

    subgraph Host Process 1
        UV["uvicorn main:app\nport 8001"]
    end

    subgraph Host Process 2
        CE["celery -A app.worker.celery_app worker\n--loglevel=info"]
    end

    subgraph Host Process 3
        ST["streamlit run ui.py\nport 8501"]
    end

    UV <-->|"TCP"| R
    UV <-->|"HTTP"| C
    CE <-->|"TCP"| R
    CE <-->|"HTTP"| C
    ST -->|"HTTP"| UV
```
