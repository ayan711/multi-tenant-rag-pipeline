"""
Tests for app/parser.py — Task 2.1: PDF Ingestion Engine.

Split into two classes:
  - TestCleanText  : pure string logic, no I/O, fast.
  - TestExtractPages : uses real PDFs built by conftest fixtures.
"""

import pytest
from app.parser import DocumentParser


@pytest.fixture
def parser():
    return DocumentParser()


# ── _clean_text ────────────────────────────────────────────────────────────────

class TestCleanText:

    def test_rejoins_hyphenated_line_break(self, parser):
        assert parser._clean_text("impor-\ntant") == "important"

    def test_rejoins_hyphen_with_leading_whitespace_on_next_line(self, parser):
        # Some PDFs emit "word-\n   continuation" with indent on the wrapped line.
        assert parser._clean_text("re-\n   sponsible") == "responsible"

    def test_collapses_multiple_spaces(self, parser):
        assert parser._clean_text("hello   world") == "hello world"

    def test_collapses_newlines_to_single_space(self, parser):
        assert parser._clean_text("line one\nline two\n\nline three") == "line one line two line three"

    def test_strips_leading_and_trailing_whitespace(self, parser):
        assert parser._clean_text("  trimmed  ") == "trimmed"

    def test_returns_empty_string_for_blank_input(self, parser):
        assert parser._clean_text("   \n\n\t  ") == ""

    def test_passthrough_clean_text(self, parser):
        # Text with no noise should come through unchanged.
        assert parser._clean_text("clean text") == "clean text"


# ── extract_text_per_page ──────────────────────────────────────────────────────────────

class TestExtractPages:

    def test_returns_list_of_dicts(self, parser, two_page_pdf):
        pages = parser.extract_text_per_page(two_page_pdf)
        assert isinstance(pages, list)
        assert all(isinstance(p, dict) for p in pages)

    def test_each_dict_has_required_keys(self, parser, two_page_pdf):
        pages = parser.extract_text_per_page(two_page_pdf)
        for page in pages:
            assert "page_number" in page
            assert "text" in page

    def test_page_numbers_are_one_indexed(self, parser, two_page_pdf):
        pages = parser.extract_text_per_page(two_page_pdf)
        assert pages[0]["page_number"] == 1
        assert pages[1]["page_number"] == 2

    def test_pages_in_document_order(self, parser, two_page_pdf):
        pages = parser.extract_text_per_page(two_page_pdf)
        numbers = [p["page_number"] for p in pages]
        assert numbers == sorted(numbers)

    def test_text_is_non_empty_string(self, parser, two_page_pdf):
        pages = parser.extract_text_per_page(two_page_pdf)
        assert all(isinstance(p["text"], str) and len(p["text"]) > 0 for p in pages)

    def test_correct_page_count_for_two_page_pdf(self, parser, two_page_pdf):
        assert len(parser.extract_text_per_page(two_page_pdf)) == 2

    def test_skips_blank_pages(self, parser, blank_then_content_pdf):
        # The fixture has 1 blank + 1 content page; only the content page should appear.
        pages = parser.extract_text_per_page(blank_then_content_pdf)
        assert len(pages) == 1

    def test_blank_page_skipped_preserves_correct_page_number(self, parser, blank_then_content_pdf):
        # The content is on physical page 2, so page_number must be 2 (not 1).
        pages = parser.extract_text_per_page(blank_then_content_pdf)
        assert pages[0]["page_number"] == 2

    def test_all_blank_pdf_returns_empty_list(self, parser, all_blank_pdf):
        assert parser.extract_text_per_page(all_blank_pdf) == []

    def test_raises_on_missing_file(self, parser):
        with pytest.raises(Exception):  # fitz raises FileNotFoundError or RuntimeError
            parser.extract_text_per_page("/nonexistent/path/file.pdf")


# ── chunk_pages ────────────────────────────────────────────────────────────────
#
# These tests use plain Python dicts instead of real PDFs — chunk_pages is pure
# word-count logic with no I/O, so there is nothing to mock or create on disk.

def _words(n: int, prefix: str = "w") -> str:
    """Return a string of `n` unique words: 'w0 w1 w2 ...'"""
    return " ".join(f"{prefix}{i}" for i in range(n))


class TestChunkPages:

    # ── Output structure ───────────────────────────────────────────────────────

    def test_empty_pages_returns_empty_list(self, parser):
        assert parser.chunk_pages([]) == []

    def test_returns_list_of_dicts(self, parser):
        pages = [{"page_number": 1, "text": _words(50)}]
        result = parser.chunk_pages(pages)
        assert isinstance(result, list)
        assert all(isinstance(c, dict) for c in result)

    def test_each_chunk_has_required_keys(self, parser):
        pages = [{"page_number": 1, "text": _words(50)}]
        for chunk in parser.chunk_pages(pages):
            assert "text" in chunk
            assert "page_number" in chunk
            assert "chunk_index" in chunk

    def test_chunk_text_is_non_empty_string(self, parser):
        pages = [{"page_number": 1, "text": _words(50)}]
        for chunk in parser.chunk_pages(pages):
            assert isinstance(chunk["text"], str) and len(chunk["text"]) > 0

    # ── Chunk count ───────────────────────────────────────────────────────────

    def test_page_shorter_than_chunk_size_produces_one_chunk(self, parser):
        pages = [{"page_number": 1, "text": _words(100)}]
        assert len(parser.chunk_pages(pages)) == 1

    def test_exactly_chunk_size_words_produces_one_chunk(self, parser):
        pages = [{"page_number": 1, "text": _words(400)}]
        assert len(parser.chunk_pages(pages, chunk_size=400, overlap=50)) == 1

    def test_one_word_over_chunk_size_produces_two_chunks(self, parser):
        # 401 words: chunk0=words[0:400], chunk1=words[350:401] — step=350
        pages = [{"page_number": 1, "text": _words(401)}]
        assert len(parser.chunk_pages(pages, chunk_size=400, overlap=50)) == 2

    def test_single_word_produces_one_chunk(self, parser):
        pages = [{"page_number": 1, "text": "hello"}]
        assert len(parser.chunk_pages(pages)) == 1

    # ── Chunk size ────────────────────────────────────────────────────────────

    def test_first_chunk_is_exactly_chunk_size_words(self, parser):
        pages = [{"page_number": 1, "text": _words(800)}]
        chunks = parser.chunk_pages(pages, chunk_size=400, overlap=50)
        assert len(chunks[0]["text"].split()) == 400

    def test_custom_chunk_size_respected(self, parser):
        # chunk_size=10, overlap=2, step=8, 30 words → 4 chunks
        # chunk0=0:10, chunk1=8:18, chunk2=16:26, chunk3=24:30
        pages = [{"page_number": 1, "text": _words(30)}]
        chunks = parser.chunk_pages(pages, chunk_size=10, overlap=2)
        assert len(chunks) == 4
        assert len(chunks[0]["text"].split()) == 10

    # ── Overlap ───────────────────────────────────────────────────────────────

    def test_consecutive_chunks_share_overlap_words(self, parser):
        # Use small numbers so the test is easy to reason about:
        # chunk_size=10, overlap=3, step=7
        # chunk0 = words[0:10], chunk1 = words[7:17]
        # The last 3 words of chunk0 must equal the first 3 words of chunk1.
        pages = [{"page_number": 1, "text": _words(20)}]
        chunks = parser.chunk_pages(pages, chunk_size=10, overlap=3)
        tail = chunks[0]["text"].split()[-3:]
        head = chunks[1]["text"].split()[:3]
        assert tail == head

    def test_default_overlap_is_50_words(self, parser):
        # 800 words → chunk0=0:400, chunk1=350:750
        # tail of chunk0 == head of chunk1 (both are words[350:400])
        pages = [{"page_number": 1, "text": _words(800)}]
        chunks = parser.chunk_pages(pages)
        tail = chunks[0]["text"].split()[-50:]
        head = chunks[1]["text"].split()[:50]
        assert tail == head

    # ── Metadata ──────────────────────────────────────────────────────────────

    def test_page_number_preserved_in_chunk(self, parser):
        pages = [{"page_number": 7, "text": _words(50)}]
        assert parser.chunk_pages(pages)[0]["page_number"] == 7

    def test_chunk_index_starts_at_zero(self, parser):
        pages = [{"page_number": 1, "text": _words(50)}]
        assert parser.chunk_pages(pages)[0]["chunk_index"] == 0

    def test_chunk_index_increments_within_one_page(self, parser):
        # 401 words → 2 chunks; indexes must be 0 and 1
        pages = [{"page_number": 1, "text": _words(401)}]
        chunks = parser.chunk_pages(pages, chunk_size=400, overlap=50)
        assert [c["chunk_index"] for c in chunks] == [0, 1]

    def test_chunk_index_is_globally_sequential_across_pages(self, parser):
        # page 1: 401 words → 2 chunks (index 0, 1)
        # page 2: 50 words  → 1 chunk  (index 2)
        pages = [
            {"page_number": 1, "text": _words(401)},
            {"page_number": 2, "text": _words(50, prefix="x")},
        ]
        chunks = parser.chunk_pages(pages, chunk_size=400, overlap=50)
        assert [c["chunk_index"] for c in chunks] == [0, 1, 2]

    def test_page_numbers_correct_across_multiple_pages(self, parser):
        pages = [
            {"page_number": 3, "text": _words(50)},
            {"page_number": 5, "text": _words(50, prefix="x")},
        ]
        chunks = parser.chunk_pages(pages)
        assert chunks[0]["page_number"] == 3
        assert chunks[1]["page_number"] == 5

    def test_chunks_are_in_document_order(self, parser):
        # chunk_index should be monotonically increasing
        pages = [
            {"page_number": 1, "text": _words(401)},
            {"page_number": 2, "text": _words(401, prefix="x")},
        ]
        chunks = parser.chunk_pages(pages, chunk_size=400, overlap=50)
        indexes = [c["chunk_index"] for c in chunks]
        assert indexes == sorted(indexes)
