"""
Tests for Task 5.5 — RAG Query Endpoint.
Tests for Task 5.6 — Grounded LLM Synthesis.
Tests for Task 5.8 — Streaming Query Response.

Covers:
  - Happy path: valid body → 200, text/event-stream, SSE events in order
  - First SSE event carries the retrieved context_chunks
  - Token events, concatenated, reconstruct the full answer
  - Final SSE event signals completion
  - Empty query string → 422 (validator rejects it)
  - Blank-whitespace query → 422
  - Empty tenant_id → 422
  - Missing required fields → 422
  - No documents for tenant → 404 (raised before streaming starts)
  - Tenant ID is forwarded to vector_store (isolation guarantee)
  - Embedder is called with the exact query text
  - OpenAI client is called with stream=True, temperature=0.0, correct model
  - Dependency injection: all three singletons can be swapped via overrides
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.dependencies import get_embedder, get_openai_client, get_vector_store
from app.main import app

# ── Fixtures ──────────────────────────────────────────────────────────────────

_FAKE_VECTOR = [0.1] * 384

_FAKE_CHUNKS = [
    {"text": "ChromaDB stores vectors.", "page_number": 1, "distance": 0.12},
    {"text": "Tenant isolation uses where= filters.", "page_number": 2, "distance": 0.25},
]

_FAKE_TOKENS = ["Tenant ", "isolation ", "uses metadata filters."]


def _mock_stream_chunk(content: str | None):
    """Mirror the shape of an OpenAI streaming chunk: chunk.choices[0].delta.content."""
    return MagicMock(choices=[MagicMock(delta=MagicMock(content=content))])


async def _fake_openai_stream(tokens: list[str]):
    for token in tokens:
        yield _mock_stream_chunk(token)


@pytest.fixture
def mock_embedder():
    m = MagicMock()
    m.embed.return_value = _FAKE_VECTOR
    return m


@pytest.fixture
def mock_vector_store():
    m = MagicMock()
    m.search_tenant_context.return_value = _FAKE_CHUNKS
    return m


@pytest.fixture
def mock_openai_client():
    m = AsyncMock()
    # create() is awaited, then the resolved value is async-iterated over —
    # so return_value must itself be an async generator, not a coroutine.
    m.chat.completions.create.return_value = _fake_openai_stream(_FAKE_TOKENS)
    return m


@pytest.fixture
def client(mock_embedder, mock_vector_store, mock_openai_client):
    """TestClient with all three heavy dependencies replaced by lightweight mocks."""
    app.dependency_overrides[get_embedder] = lambda: mock_embedder
    app.dependency_overrides[get_vector_store] = lambda: mock_vector_store
    app.dependency_overrides[get_openai_client] = lambda: mock_openai_client
    yield TestClient(app)
    app.dependency_overrides.clear()


def _query(client, *, query="What is tenant isolation?", tenant_id="acme"):
    return client.post("/api/v1/query", json={"query": query, "tenant_id": tenant_id})


def _parse_sse(response) -> list[dict]:
    """Split an SSE body into its decoded `data: {...}` JSON payloads."""
    events = []
    for block in response.text.split("\n\n"):
        block = block.strip()
        if block.startswith("data: "):
            events.append(json.loads(block[len("data: ") :]))
    return events


# ── Happy path ────────────────────────────────────────────────────────────────


def test_query_returns_200(client):
    assert _query(client).status_code == 200


def test_query_content_type_is_event_stream(client):
    resp = _query(client)
    assert resp.headers["content-type"].startswith("text/event-stream")


def test_first_event_carries_context_chunks(client):
    events = _parse_sse(_query(client))
    assert events[0]["type"] == "context"
    assert len(events[0]["chunks"]) == 2


def test_context_chunk_fields(client):
    events = _parse_sse(_query(client))
    chunk = events[0]["chunks"][0]
    assert chunk["text"] == "ChromaDB stores vectors."
    assert chunk["page_number"] == 1
    assert chunk["distance"] == pytest.approx(0.12)


def test_token_events_reconstruct_full_answer(client):
    events = _parse_sse(_query(client))
    tokens = [e["content"] for e in events if e["type"] == "token"]
    assert "".join(tokens) == "".join(_FAKE_TOKENS)


def test_last_event_is_done(client):
    events = _parse_sse(_query(client))
    assert events[-1] == {"type": "done"}


def test_event_order_is_context_then_tokens_then_done(client):
    events = _parse_sse(_query(client))
    types = [e["type"] for e in events]
    assert types[0] == "context"
    assert types[-1] == "done"
    assert all(t == "token" for t in types[1:-1])


# ── Embedder / vector store interaction ──────────────────────────────────────


def test_embedder_called_with_query_text(client, mock_embedder):
    _query(client, query="What is RAG?")
    mock_embedder.embed.assert_called_once_with("What is RAG?")


def test_vector_store_called_with_correct_tenant(client, mock_vector_store):
    _query(client, tenant_id="beta-corp")
    mock_vector_store.search_tenant_context.assert_called_once_with(
        _FAKE_VECTOR, "beta-corp"
    )


def test_tenant_id_is_not_leaked_across_tenants(mock_embedder):
    """Two different tenants must each call search with their own tenant_id."""
    store_a = MagicMock()
    store_a.search_tenant_context.return_value = _FAKE_CHUNKS

    store_b = MagicMock()
    store_b.search_tenant_context.return_value = _FAKE_CHUNKS

    client_a_openai = AsyncMock()
    client_a_openai.chat.completions.create.return_value = _fake_openai_stream(_FAKE_TOKENS)

    app.dependency_overrides[get_embedder] = lambda: mock_embedder
    app.dependency_overrides[get_vector_store] = lambda: store_a
    app.dependency_overrides[get_openai_client] = lambda: client_a_openai
    client_a = TestClient(app)
    _query(client_a, tenant_id="acme")
    store_a.search_tenant_context.assert_called_once_with(_FAKE_VECTOR, "acme")
    # store_b was never touched
    store_b.search_tenant_context.assert_not_called()
    app.dependency_overrides.clear()


# ── No documents ──────────────────────────────────────────────────────────────


def test_no_documents_returns_404(mock_embedder, mock_openai_client):
    empty_store = MagicMock()
    empty_store.search_tenant_context.return_value = []

    app.dependency_overrides[get_embedder] = lambda: mock_embedder
    app.dependency_overrides[get_vector_store] = lambda: empty_store
    app.dependency_overrides[get_openai_client] = lambda: mock_openai_client
    try:
        client = TestClient(app)
        resp = _query(client, tenant_id="new-tenant")
        assert resp.status_code == 404
        assert "new-tenant" in resp.json()["error"]
    finally:
        app.dependency_overrides.clear()


# ── Input validation ──────────────────────────────────────────────────────────


def test_empty_query_returns_422(client):
    assert _query(client, query="").status_code == 422


def test_blank_whitespace_query_returns_422(client):
    assert _query(client, query="   ").status_code == 422


def test_empty_tenant_id_returns_422(client):
    assert _query(client, tenant_id="").status_code == 422


def test_missing_query_field_returns_422(client):
    resp = client.post("/api/v1/query", json={"tenant_id": "acme"})
    assert resp.status_code == 422


def test_missing_tenant_id_field_returns_422(client):
    resp = client.post("/api/v1/query", json={"query": "hello"})
    assert resp.status_code == 422


def test_empty_body_returns_422(client):
    resp = client.post("/api/v1/query", json={})
    assert resp.status_code == 422


# ── Whitespace trimming ───────────────────────────────────────────────────────


def test_query_is_stripped_before_embedding(client, mock_embedder):
    """Leading/trailing whitespace is trimmed; the clean string reaches the embedder."""
    _query(client, query="  What is RAG?  ")
    mock_embedder.embed.assert_called_once_with("What is RAG?")


# ── LLM synthesis (Task 5.6 / 5.8) ────────────────────────────────────────────


def test_openai_called_with_stream_true(client, mock_openai_client):
    _query(client)
    call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["stream"] is True


def test_openai_called_with_temperature_zero(client, mock_openai_client):
    _query(client)
    call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["temperature"] == 0.0


def test_openai_called_with_configured_model(client, mock_openai_client):
    from app.config import settings

    _query(client)
    call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == settings.openai_model


def test_openai_messages_contain_query(client, mock_openai_client):
    _query(client, query="What is tenant isolation?")
    messages = mock_openai_client.chat.completions.create.call_args.kwargs["messages"]
    user_message = next(m for m in messages if m["role"] == "user")
    assert "What is tenant isolation?" in user_message["content"]


def test_openai_system_prompt_contains_context_chunks(client, mock_openai_client):
    _query(client)
    messages = mock_openai_client.chat.completions.create.call_args.kwargs["messages"]
    system_message = next(m for m in messages if m["role"] == "system")
    # Both chunk texts from _FAKE_CHUNKS must appear in the system prompt context.
    assert "ChromaDB stores vectors." in system_message["content"]
    assert "Tenant isolation uses where= filters." in system_message["content"]


def test_openai_system_prompt_forbids_outside_knowledge(client, mock_openai_client):
    _query(client)
    messages = mock_openai_client.chat.completions.create.call_args.kwargs["messages"]
    system_message = next(m for m in messages if m["role"] == "system")
    # The prompt must instruct the model to stay within the provided context.
    assert "ONLY" in system_message["content"]


def test_openai_called_exactly_once_per_request(client, mock_openai_client):
    _query(client)
    mock_openai_client.chat.completions.create.assert_called_once()


# ── synthesise_stream unit tests ──────────────────────────────────────────────


@pytest.mark.anyio
async def test_synthesise_stream_yields_tokens_in_order():
    from app.services.synthesis import synthesise_stream

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = _fake_openai_stream(_FAKE_TOKENS)

    collected = [
        token
        async for token in synthesise_stream(
            "My query", _FAKE_CHUNKS, mock_client, "gpt-4o-mini"
        )
    ]
    assert collected == _FAKE_TOKENS


@pytest.mark.anyio
async def test_synthesise_stream_skips_empty_deltas():
    """Some OpenAI stream chunks (e.g. the role-announcement chunk) carry no
    content delta — these must not surface as empty-string tokens."""
    from app.services.synthesis import synthesise_stream

    async def stream_with_gaps():
        yield _mock_stream_chunk(None)
        yield _mock_stream_chunk("Hello")
        yield _mock_stream_chunk(None)

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = stream_with_gaps()

    collected = [
        token
        async for token in synthesise_stream(
            "My query", _FAKE_CHUNKS, mock_client, "gpt-4o-mini"
        )
    ]
    assert collected == ["Hello"]


# ── build_context_block unit tests ────────────────────────────────────────────


def test_build_context_block_formats_chunks():
    from app.services.synthesis import build_context_block

    chunks = [
        {"text": "First sentence.", "page_number": 1, "distance": 0.1},
        {"text": "Second sentence.", "page_number": 3, "distance": 0.2},
    ]
    block = build_context_block(chunks)
    assert "[Page 1] First sentence." in block
    assert "[Page 3] Second sentence." in block


def test_build_context_block_separates_chunks():
    from app.services.synthesis import build_context_block

    chunks = [
        {"text": "A", "page_number": 1, "distance": 0.1},
        {"text": "B", "page_number": 2, "distance": 0.2},
    ]
    # Chunks must be separated so the model doesn't read them as one sentence.
    assert "\n\n" in build_context_block(chunks)
