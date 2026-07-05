from collections.abc import AsyncIterator

from openai import AsyncOpenAI

# Prompt instructs the model to answer strictly from context and give a
# specific fallback phrase when the answer isn't there — makes it testable.
_SYSTEM_PROMPT = """\
You are a precise question-answering assistant.
Answer the user's question using ONLY the context excerpts provided below.
Each excerpt is labeled with its source page number.
If the answer cannot be found in the provided context, respond with exactly:
"I don't have enough information in the provided documents to answer that."
Do not use any knowledge outside the provided context. Be concise.\
"""


def build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    return "\n\n".join(f"[Page {c['page_number']}] {c['text']}" for c in chunks)


async def synthesise_stream(
    query: str,
    chunks: list[dict],
    client: AsyncOpenAI,
    model: str,
) -> AsyncIterator[str]:
    """Call the LLM and yield answer tokens as they arrive, instead of
    waiting for the full completion — lets the client render the answer
    incrementally instead of staring at a blank screen during generation.

    temperature=0.0 makes the output deterministic — the model summarises what
    the documents say rather than generating creative text.
    """
    context = build_context_block(chunks)
    stream = await client.chat.completions.create(
        model=model,
        temperature=0.0,
        stream=True,
        messages=[
            {"role": "system", "content": f"{_SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"},
            {"role": "user", "content": query},
        ],
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
