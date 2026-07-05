"""
Tests for Task 2.5 — In-Memory Ingestion Verification.

Exercises the full parser → embedder pipeline using:
  - A minimal PDF created entirely in memory with fitz (no file on disk required).
  - A mocked SentenceTransformer so model weights are never downloaded.

Goal: confirm that the two modules compose correctly and that the output
shapes match what ChromaDB and the Celery worker will expect downstream.
"""

import numpy as np
import pytest
import fitz  # PyMuPDF
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.parser import DocumentParser
from app.embedder import LocalEmbedder


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_pdf(tmp_path: Path) -> Path:
    """
    Create a minimal two-page PDF in memory and write it to a temp file.
    Using fitz.Document() directly avoids any external test-data dependency.
    """
    doc = fitz.open()  # blank in-memory document

    page1 = doc.new_page()
    # insert_text requires a fitz.Point for the insertion coordinate
    page1.insert_text(fitz.Point(72, 72), "This is the first page. " * 30)

    page2 = doc.new_page()
    page2.insert_text(fitz.Point(72, 72), "This is the second page. " * 30)

    out = tmp_path / "test_doc.pdf"
    doc.save(str(out))
    doc.close()
    return out


@pytest.fixture
def mock_embedder():
    """
    Patch SentenceTransformer so no weights are downloaded.
    encode() returns deterministic numpy arrays shaped correctly for
    all-MiniLM-L6-v2 (384 dims).
    """
    with patch("app.embedder.SentenceTransformer") as mock_cls:
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 384

        # encode is called with a list of texts; return one row per text
        def fake_encode(texts, **kwargs):
            n = len(texts) if isinstance(texts, list) else 1
            return np.ones((n, 384), dtype=np.float32) * 0.1

        mock_instance.encode.side_effect = fake_encode
        mock_cls.return_value = mock_instance
        yield mock_cls, mock_instance


# ── Pipeline integration tests ────────────────────────────────────────────────

class TestParserEmbedderPipeline:

    def test_pages_extracted_from_pdf(self, tmp_pdf):
        pages = DocumentParser().extract_text_per_page(str(tmp_pdf))
        assert len(pages) == 2

    def test_pages_have_correct_keys(self, tmp_pdf):
        pages = DocumentParser().extract_text_per_page(str(tmp_pdf))
        for page in pages:
            assert "page_number" in page
            assert "text" in page

    def test_chunks_produced_from_pages(self, tmp_pdf):
        parser = DocumentParser()
        pages = parser.extract_text_per_page(str(tmp_pdf))
        chunks = parser.chunk_pages(pages, chunk_size=50, overlap=10)
        assert len(chunks) > 0

    def test_chunk_count_matches_vector_count(self, tmp_pdf, mock_embedder):
        parser = DocumentParser()
        pages = parser.extract_text_per_page(str(tmp_pdf))
        chunks = parser.chunk_pages(pages, chunk_size=50, overlap=10)
        texts = [c["text"] for c in chunks]

        embedder = LocalEmbedder()
        vectors = embedder.embed_batch(texts)

        assert len(vectors) == len(chunks)

    def test_each_vector_has_correct_dimension(self, tmp_pdf, mock_embedder):
        parser = DocumentParser()
        pages = parser.extract_text_per_page(str(tmp_pdf))
        chunks = parser.chunk_pages(pages, chunk_size=50, overlap=10)

        embedder = LocalEmbedder()
        vectors = embedder.embed_batch([c["text"] for c in chunks])

        assert all(len(v) == 384 for v in vectors)

    def test_vectors_are_lists_of_float(self, tmp_pdf, mock_embedder):
        parser = DocumentParser()
        pages = parser.extract_text_per_page(str(tmp_pdf))
        chunks = parser.chunk_pages(pages, chunk_size=50, overlap=10)

        embedder = LocalEmbedder()
        vectors = embedder.embed_batch([c["text"] for c in chunks])

        for v in vectors:
            assert isinstance(v, list)
            assert all(isinstance(x, float) for x in v)

    def test_chunk_indices_are_globally_sequential(self, tmp_pdf):
        parser = DocumentParser()
        pages = parser.extract_text_per_page(str(tmp_pdf))
        chunks = parser.chunk_pages(pages, chunk_size=50, overlap=10)

        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_page_numbers_are_preserved_in_chunks(self, tmp_pdf):
        parser = DocumentParser()
        pages = parser.extract_text_per_page(str(tmp_pdf))
        chunks = parser.chunk_pages(pages, chunk_size=50, overlap=10)

        page_numbers_in_chunks = {c["page_number"] for c in chunks}
        # Both pages should appear in the chunk list
        assert page_numbers_in_chunks == {1, 2}

    def test_blank_pdf_produces_no_chunks(self, tmp_path, mock_embedder):
        # A PDF with pages that have no text → parser skips them → no chunks → no vectors
        doc = fitz.open()
        doc.new_page()  # blank page, no text inserted
        blank_pdf = tmp_path / "blank.pdf"
        doc.save(str(blank_pdf))
        doc.close()

        parser = DocumentParser()
        pages = parser.extract_text_per_page(str(blank_pdf))
        chunks = parser.chunk_pages(pages)

        assert pages == []
        assert chunks == []
