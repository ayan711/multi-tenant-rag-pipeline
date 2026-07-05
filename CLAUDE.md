# Enterprise RAG — Claude Context

## Project Overview

A production-grade, zero-cost Document Q&A (RAG) engine. Core idea: upload a PDF, ask questions, get answers grounded strictly in that document. Built for learning and interview prep.

**Stack:** FastAPI · PyMuPDF · sentence-transformers (local) · ChromaDB · Celery · Redis · OpenAI (gpt-4o-mini) · Docker Compose · Streamlit (demo UI)

---

## Collaboration Rules

- **Task-by-task only.** User picks the next task. Never start the next task until explicitly told.
- **Learning-first.** Add short inline comments only on genuinely non-obvious lines — explain the WHY in one line, not a paragraph.
- **After every task:** update `learning.md` with key concepts, gotchas, and one interview angle per task. Keep each concept to 2–4 sentences max; use a table or bullet list instead of prose paragraphs where possible.
- **After every task:** check the box in `tasks.md` for the completed task.
- **Tests alongside every task.** For pure-logic tasks (parser, chunker, embedder) write `pytest` unit tests. For infrastructure tasks (ChromaDB, Redis, Celery) write integration tests or lightweight mocks. Tests live in `tests/` mirroring `app/` structure (e.g. `tests/test_parser.py`).
- **Review loop:** user reviews the code and may leave comments. Address all feedback before marking a task done.

---

## Project Files

| File | Purpose |
|------|---------|
| `tasks.md` | Master development checklist — source of truth for what to build |
| `features.md` | Product feature specification |
| `architecture.md` | Architecture + sequence diagrams (Mermaid) |
| `learning.md` | Running learning notes — updated after each task |
| `CLAUDE.md` | This file — session context and progress tracker |

---

## Progress Tracker

### Phase 1: Local Infrastructure & Environment
- [x] Task 1.1 — Project Directory Structure Setup
- [x] Task 1.2 — Dependency Configuration
- [x] Task 1.3 — Docker Compose Environment Provisioning
- [x] Task 1.4 — Environment Variables Isolation
- [x] Task 1.5 — Infrastructure Smoke Test

### Phase 2: Structural Data Extraction & Local Embedding
- [x] Task 2.1 — PDF Ingestion Engine
- [x] Task 2.2 — Chunking Sequence Logic
- [x] Task 2.3 — Local Vector Model Initialization
- [x] Task 2.4 — Vectorization Methods
- [x] Task 2.5 — In-Memory Ingestion Verification
- [x] Task 2.6 — Chunking Strategy Comparison

### Phase 3: Secure Multi-Tenant Vector Database Layer
- [x] Task 3.1 — Standalone Vector Database Client Setup
- [x] Task 3.2 — Multi-Tenant Collection Mapping
- [x] Task 3.3 — Secured Metadata Injection Routing
- [x] Task 3.4 — Isolated Metadata Scoping Queries
- [x] Task 3.5 — Isolation Testing
- [x] Task 3.6 — Document Deletion Endpoint

### Phase 4: Asynchronous Processing Engine
- [x] Task 4.1 — Distributed Task Factory Setup
- [x] Task 4.2 — Worker Instance Lazy Loading
- [x] Task 4.3 — Ingestion Pipeline Consolidation
- [x] Task 4.4 — Background Worker Simulation Test
- [x] Task 4.5 — Idempotent Ingestion Guard

### Phase 5: FastAPI Web Gateway & Completion Engine
- [x] Task 5.1 — Web Server Foundation Initialization
- [x] Task 5.2 — Asynchronous File Upload API Route
- [x] Task 5.3 — Unique Fingerprint Identification Architecture
- [ ] Task 5.4 — Non-Blocking Celery Offloading
- [x] Task 5.5 — Secure RAG Completion API Route
- [x] Task 5.6 — Zero-Temperature Synthesis Grounding
- [x] Task 5.7 — Health Check Endpoint
- [x] Task 5.8 — Streaming Query Response
- [ ] Task 5.9 — Task Status Polling Endpoint

### Phase 6: System Validation & Interview Demos
- [ ] Task 6.1 — Full Stack Runtime Execution
- [ ] Task 6.2 — End-to-End File Processing Upload Check
- [ ] Task 6.3 — Knowledge Ingestion Query Verification
- [ ] Task 6.4 — Multi-Tenant Logical Security Firewall Audit
- [ ] Task 6.5 — Retrieval Quality Evaluation (Golden Set)

### Phase 7: Demo & Interview Readiness
- [ ] Task 7.1 — Minimal Streamlit UI
- [ ] Task 7.2 — Architecture Diagram & README

---

## Current Status

**Last completed task:** Task 5.8 — Streaming Query Response
**Next task:** Awaiting user instruction.

---

## Directory Structure (target)

```
enterprise-rag/
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI gateway
│   ├── worker.py        # Celery task definitions
│   ├── config.py        # pydantic-settings env loader
│   ├── parser.py        # PDF parsing + chunking
│   ├── embedder.py      # sentence-transformers wrapper
│   └── vector_db.py     # ChromaDB client wrapper
├── docker-compose.yml
├── requirements.txt
├── .env                 # secrets (never commit)
├── ui.py                # Streamlit demo UI
└── chroma_data/         # ChromaDB persistence volume
```

---

## Key Design Decisions (for reference)

| Decision | Choice | Why |
|----------|--------|-----|
| Embedding model | `all-MiniLM-L6-v2` (local) | Zero API cost, 384-dim, fast on CPU |
| Vector store | ChromaDB (self-hosted Docker) | Free, persistent, supports metadata filters |
| Tenant isolation | `where={"tenant_id": ...}` on every query | Enforced at DB layer, not app layer |
| Async ingestion | Celery + Redis | Keeps HTTP response fast; PDF processing is slow |
| LLM | `gpt-4o-mini` at `temperature=0.0` | Low cost, deterministic, grounded answers |
| Chunk size | 400 words, 50-word overlap | Balances context richness vs retrieval noise |
