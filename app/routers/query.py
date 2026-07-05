import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from app.config import settings
from app.dependencies import get_embedder, get_openai_client, get_vector_store
from app.embedder import LocalEmbedder
from app.schemas.query import ContextChunk, QueryRequest
from app.services.synthesis import synthesise_stream
from app.vector_db import VectorStoreManager

router = APIRouter(prefix="/query", tags=["query"])


@router.post("")
async def rag_query(
    body: QueryRequest,
    embedder: LocalEmbedder = Depends(get_embedder),
    vector_store: VectorStoreManager = Depends(get_vector_store),
    openai_client: AsyncOpenAI = Depends(get_openai_client),
) -> StreamingResponse:
    query_vector = embedder.embed(body.query)
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
        context_event = {
            "type": "context",
            "chunks": [ContextChunk(**c).model_dump() for c in chunks],
        }
        yield f"data: {json.dumps(context_event)}\n\n"

        async for token in synthesise_stream(
            body.query, chunks, openai_client, settings.openai_model
        ):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
