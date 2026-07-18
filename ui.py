# ─────────────────────────────────────────────────────────────────────────────
# ui.py — Streamlit Demo UI (Task 7.1)
# ─────────────────────────────────────────────────────────────────────────────
#
# A thin HTTP client in front of the FastAPI gateway. Deliberately has zero
# imports from `app/` — it talks to the API the same way any external client
# would (plain `requests` calls), so it never pulls in Celery/ChromaDB/model
# weights just to render a page.
#
# Run with:  streamlit run ui.py
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import time
from collections.abc import Iterator

import requests
import streamlit as st

API_BASE_URL = os.environ.get("RAG_API_BASE_URL", "http://localhost:8001")
POLL_INTERVAL_SECONDS = 1.5

# ── API client helpers ──────────────────────────────────────────────────────
#
# Kept free of any `st.*` calls so they're plain functions the test suite can
# import and exercise without a running Streamlit session.


def parse_sse_line(raw_line: bytes | str) -> dict | None:
    """Parse one line from an SSE stream into its JSON payload.

    requests.iter_lines() also yields the blank lines that separate SSE
    frames — those (and any non "data:" line) return None so callers can
    skip them without special-casing blanks themselves.
    """
    if isinstance(raw_line, bytes):
        raw_line = raw_line.decode("utf-8")
    if not raw_line.startswith("data:"):
        return None
    return json.loads(raw_line[len("data:") :].strip())


def upload_document(tenant_id: str, filename: str, file_bytes: bytes) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/api/v1/docs/upload",
        data={"tenant_id": tenant_id},
        files={"file": (filename, file_bytes, "application/pdf")},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def get_task_status(task_id: str) -> dict:
    response = requests.get(f"{API_BASE_URL}/api/v1/tasks/{task_id}", timeout=10)
    response.raise_for_status()
    return response.json()


def stream_query(query: str, tenant_id: str) -> Iterator[dict]:
    """Yield parsed SSE events from POST /api/v1/query as they arrive."""
    with requests.post(
        f"{API_BASE_URL}/api/v1/query",
        json={"query": query, "tenant_id": tenant_id},
        stream=True,
        timeout=60,
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            event = parse_sse_line(raw_line)
            if event is not None:
                yield event


# ── Page rendering ───────────────────────────────────────────────────────────
#
# Wrapped in main() and only run under `if __name__ == "__main__"` (true when
# Streamlit execs this file) so `import ui` in tests just defines the helpers
# above without also trying to render a page outside a Streamlit session.


def main() -> None:
    st.set_page_config(page_title="Enterprise RAG", page_icon="📄")
    st.title("📄 Enterprise RAG — Document Q&A")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Tenant")
        tenant_id = st.text_input("Tenant ID", value="demo-tenant")

        st.header("Upload a document")
        uploaded_file = st.file_uploader("PDF", type="pdf")
        if uploaded_file is not None and st.button("Ingest"):
            _handle_upload(tenant_id, uploaded_file)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask a question about your documents..."):
        _handle_question(question, tenant_id)


def _handle_upload(tenant_id: str, uploaded_file) -> None:
    with st.spinner("Uploading..."):
        try:
            result = upload_document(tenant_id, uploaded_file.name, uploaded_file.getvalue())
        except requests.RequestException as exc:
            st.error(f"Upload failed: {exc}")
            return

    if result.get("status") == "already_exists":
        st.info(f"Already ingested (file_hash={result['file_hash']})")
        return

    task_id = result["task_id"]
    status_placeholder = st.empty()
    status = "PENDING"
    # Ingestion runs in a Celery worker (Phase 4) — poll rather than block,
    # since a large PDF can take several seconds to parse/embed/index.
    while status in ("PENDING", "STARTED", "RETRY"):
        status_placeholder.write(f"Status: {status}")
        time.sleep(POLL_INTERVAL_SECONDS)
        status = get_task_status(task_id)["status"]

    if status == "SUCCESS":
        status_placeholder.success("Document ingested and ready to query.")
    else:
        status_placeholder.error(f"Ingestion failed: {status}")


def _handle_question(question: str, tenant_id: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        context_placeholder = st.empty()
        answer_placeholder = st.empty()
        answer = ""
        try:
            for event in stream_query(question, tenant_id):
                if event["type"] == "context":
                    pages = sorted({c["page_number"] for c in event["chunks"]})
                    context_placeholder.caption(
                        f"Answering from page(s): {', '.join(map(str, pages))}"
                    )
                elif event["type"] == "token":
                    answer += event["content"]
                    answer_placeholder.markdown(answer)
        except requests.HTTPError as exc:
            # Router raises 404 (no docs for tenant) as {"error": "..."} —
            # surface that message instead of a raw stack-style error string.
            detail = exc.response.json().get("error", str(exc))
            answer = f"Error: {detail}"
            answer_placeholder.markdown(answer)
        except requests.RequestException as exc:
            answer = f"Error: {exc}"
            answer_placeholder.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
