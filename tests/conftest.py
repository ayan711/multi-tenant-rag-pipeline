"""
Shared pytest fixtures for the Enterprise RAG test suite.

Fixtures that create real PDFs via fitz let us test PyMuPDF behavior
without mocking — more confidence, less setup noise.
"""

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Session-wide ChromaDB mock — must run at module level, not inside a fixture.
#
# app.worker instantiates VectorStoreManager at import time (Task 4.2), which
# calls chromadb.HttpClient(...). This patch is started here — before pytest
# collects any test file — so importing app.worker never requires a live server.
#
# Individual tests that need their own mock behaviour apply a narrower
# patch("app.vector_db.chromadb.HttpClient", ...) which takes precedence within
# its `with` block, then restores this global mock on exit.
# ---------------------------------------------------------------------------
_mock_chroma_instance = MagicMock()
_mock_chroma_instance.get_or_create_collection.return_value = MagicMock()
patch("chromadb.HttpClient", return_value=_mock_chroma_instance).start()

import fitz
import pytest


@pytest.fixture
def two_page_pdf(tmp_path):
    """A PDF with two content pages, each with known text."""
    path = tmp_path / "two_page.pdf"
    doc = fitz.open()
    for text in ["Content of page one.", "Content of page two."]:
        page = doc.new_page()
        page.insert_text((50, 72), text)
    doc.save(str(path))
    doc.close()
    return str(path)


@pytest.fixture
def blank_then_content_pdf(tmp_path):
    """A PDF whose first page is blank and second has text — tests blank-page skipping."""
    path = tmp_path / "blank_first.pdf"
    doc = fitz.open()
    doc.new_page()  # blank page — no text inserted
    page = doc.new_page()
    page.insert_text((50, 72), "Only real content here.")
    doc.save(str(path))
    doc.close()
    return str(path)


@pytest.fixture
def all_blank_pdf(tmp_path):
    """A PDF where every page is blank — extract_pages should return an empty list."""
    path = tmp_path / "all_blank.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.new_page()
    doc.save(str(path))
    doc.close()
    return str(path)
