"""
Tests for Task 7.1 — Streamlit UI.

ui.py's page-rendering code (main()) only runs under `streamlit run`, guarded
by `if __name__ == "__main__"`. These tests exercise the plain-function HTTP
helpers above it — parsing and request-building logic — without needing a
live FastAPI server or a Streamlit session.
"""

from unittest.mock import MagicMock, patch

import pytest

from ui import get_task_status, parse_sse_line, stream_query, upload_document

# ── parse_sse_line ───────────────────────────────────────────────────────────


def test_parse_sse_line_decodes_data_line():
    assert parse_sse_line('data: {"type": "done"}') == {"type": "done"}


def test_parse_sse_line_accepts_bytes():
    assert parse_sse_line(b'data: {"type": "done"}') == {"type": "done"}


def test_parse_sse_line_ignores_blank_line():
    assert parse_sse_line("") is None


def test_parse_sse_line_ignores_non_data_line():
    assert parse_sse_line(": comment") is None


def test_parse_sse_line_roundtrips_context_event():
    payload = '{"type": "context", "chunks": [{"text": "x", "page_number": 1, "distance": 0.1}]}'
    assert parse_sse_line(f"data: {payload}")["chunks"][0]["page_number"] == 1


# ── upload_document ───────────────────────────────────────────────────────────


def test_upload_document_posts_multipart_with_tenant_and_file():
    mock_response = MagicMock()
    mock_response.json.return_value = {"task_id": "abc", "file_hash": "deadbeef"}

    with patch("ui.requests.post", return_value=mock_response) as mock_post:
        result = upload_document("acme", "doc.pdf", b"%PDF-1.4")

    assert result == {"task_id": "abc", "file_hash": "deadbeef"}
    mock_response.raise_for_status.assert_called_once()
    _, kwargs = mock_post.call_args
    assert kwargs["data"] == {"tenant_id": "acme"}
    assert kwargs["files"]["file"] == ("doc.pdf", b"%PDF-1.4", "application/pdf")


def test_upload_document_raises_on_http_error():
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = __import__("requests").HTTPError("400")

    with patch("ui.requests.post", return_value=mock_response):
        with pytest.raises(Exception):
            upload_document("acme", "doc.pdf", b"not a pdf")


# ── get_task_status ───────────────────────────────────────────────────────────


def test_get_task_status_returns_parsed_body():
    mock_response = MagicMock()
    mock_response.json.return_value = {"task_id": "abc", "status": "SUCCESS"}

    with patch("ui.requests.get", return_value=mock_response) as mock_get:
        result = get_task_status("abc")

    assert result["status"] == "SUCCESS"
    assert "abc" in mock_get.call_args[0][0]


# ── stream_query ───────────────────────────────────────────────────────────


def test_stream_query_yields_parsed_events_and_skips_blanks():
    lines = [
        b'data: {"type": "context", "chunks": []}',
        b"",
        b'data: {"type": "token", "content": "Hi"}',
        b'data: {"type": "done"}',
    ]
    mock_response = MagicMock()
    mock_response.iter_lines.return_value = iter(lines)
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = False

    with patch("ui.requests.post", return_value=mock_response) as mock_post:
        events = list(stream_query("What is X?", "acme"))

    assert events == [
        {"type": "context", "chunks": []},
        {"type": "token", "content": "Hi"},
        {"type": "done"},
    ]
    _, kwargs = mock_post.call_args
    assert kwargs["json"] == {"query": "What is X?", "tenant_id": "acme"}
    assert kwargs["stream"] is True
