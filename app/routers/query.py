import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from app.config import settings
from app.dependencies import get_embedder, get_llm_client, get_vector_store
from app.embedder import LocalEmbedder
from app.schemas.query import ContextChunk, QueryRequest
from app.services.synthesis import synthesise_stream
from app.vector_db import VectorStoreManager

router = APIRouter(prefix="/query", tags=["query"])


@router.post("")
async def rag_query(
    body: QueryRequest,
    # Depends() injects the cached singletons from app/dependencies.py instead
    # of constructing a new embedding model / DB client on every request.
    embedder: LocalEmbedder = Depends(get_embedder),
    vector_store: VectorStoreManager = Depends(get_vector_store),
    llm_client: AsyncOpenAI = Depends(get_llm_client),
) -> StreamingResponse:
    # Turn the question into the same 384-dim vector space the stored chunks live in.
    query_vector = embedder.embed(body.query)
    # The where={"tenant_id": ...} filter inside this call is what actually
    # enforces isolation — nothing in this router does it separately.
    chunks = vector_store.search_tenant_context(query_vector, body.tenant_id)

    if not chunks:   
        raise HTTPException(
            status_code=404,
            detail=f"No documents found for tenant '{body.tenant_id}'. "
            "Upload a document before querying.",
        )

    # Retrieval (embed + search) happens above, outside the generator, so a
    # 404 still surfaces as a normal HTTP error instead of a broken stream.
    async def event_stream() -> AsyncIterator[str]:
        # Sent first so the client can show "answering from page 1, 2..."
        # before any answer text starts arriving.
        context_event = {
            "type": "context",
            "chunks": [ContextChunk(**c).model_dump() for c in chunks],
        }
        # SSE wire format is "data: <json>\n\n" — the trailing blank line is
        # the event terminator every SSE client (e.g. browser EventSource) expects.
        yield f"data: {json.dumps(context_event)}\n\n"

        # synthesise_stream calls OpenAI with stream=True and yields each answer
        # token as it's generated, instead of blocking for the full completion.
        async for token in synthesise_stream(
            body.query, chunks, llm_client, settings.gemini_model
        ):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        # Explicit end-of-stream marker — without it the client can't tell
        # "answer finished" apart from "connection dropped mid-answer".
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
