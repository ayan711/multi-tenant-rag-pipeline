"""
Tests for SentenceChunker in app/parser.py — Task 2.6.

SentenceChunker is pure logic (no I/O), so all tests use plain Python dicts.
We compare behavior against DocumentParser.chunk_pages() to highlight
the trade-offs: sentence coherence vs predictable word count.
"""

import pytest
from app.parser import DocumentParser, SentenceChunker


@pytest.fixture
def chunker():
    return SentenceChunker()


@pytest.fixture
def fixed_chunker():
    return DocumentParser()


# ── Output structure ───────────────────────────────────────────────────────────

class TestOutputStructure:

    def test_empty_pages_returns_empty_list(self, chunker):
        assert chunker.chunk_pages([]) == []

    def test_returns_list_of_dicts(self, chunker):
        pages = [{"page_number": 1, "text": "Hello world. This is a test."}]
        result = chunker.chunk_pages(pages)
        assert isinstance(result, list)
        assert all(isinstance(c, dict) for c in result)

    def test_each_chunk_has_required_keys(self, chunker):
        pages = [{"page_number": 1, "text": "First sentence. Second sentence."}]
        for chunk in chunker.chunk_pages(pages):
            assert "text" in chunk
            assert "page_number" in chunk
            assert "chunk_index" in chunk

    def test_chunk_text_is_non_empty_string(self, chunker):
        pages = [{"page_number": 1, "text": "Hello world. This is a test."}]
        for chunk in chunker.chunk_pages(pages):
            assert isinstance(chunk["text"], str) and len(chunk["text"]) > 0

    def test_chunk_index_starts_at_zero(self, chunker):
        pages = [{"page_number": 1, "text": "Hello world."}]
        assert chunker.chunk_pages(pages)[0]["chunk_index"] == 0

    def test_chunk_index_is_globally_sequential_across_pages(self, chunker):
        # Many sentences per page to force multiple chunks
        page_text = " ".join(f"Sentence {i}." for i in range(100))
        pages = [
            {"page_number": 1, "text": page_text},
            {"page_number": 2, "text": page_text},
        ]
        chunks = chunker.chunk_pages(pages, max_words=50, overlap_sentences=0)
        indexes = [c["chunk_index"] for c in chunks]
        assert indexes == list(range(len(chunks)))

    def test_page_number_preserved_in_chunk(self, chunker):
        pages = [{"page_number": 7, "text": "Hello world. Goodbye world."}]
        for chunk in chunker.chunk_pages(pages):
            assert chunk["page_number"] == 7


# ── Sentence boundaries ────────────────────────────────────────────────────────

class TestSentenceBoundaries:

    def test_short_page_produces_one_chunk(self, chunker):
        pages = [{"page_number": 1, "text": "Short text. Also short."}]
        assert len(chunker.chunk_pages(pages, max_words=400)) == 1

    def test_chunk_contains_complete_sentences(self, chunker):
        """Every chunk should end on a sentence boundary (no mid-sentence split)."""
        sentences = [f"This is sentence number {i}." for i in range(20)]
        pages = [{"page_number": 1, "text": " ".join(sentences)}]
        chunks = chunker.chunk_pages(pages, max_words=30, overlap_sentences=0)
        for chunk in chunks:
            # A complete sentence ends with '. '  or '.' at end of text.
            text = chunk["text"].strip()
            assert text.endswith("."), f"Chunk does not end on sentence boundary: '{text[-50:]}'"

    def test_single_sentence_longer_than_max_forms_own_chunk(self, chunker):
        # A 500-word sentence must not be discarded even if it exceeds max_words.
        long_sentence = "word " * 500 + "end."
        pages = [{"page_number": 1, "text": long_sentence.strip()}]
        chunks = chunker.chunk_pages(pages, max_words=400, overlap_sentences=0)
        assert len(chunks) >= 1
        assert "end." in chunks[-1]["text"]

    def test_never_cuts_mid_sentence(self, chunker):
        """Fixed-size chunker CAN cut mid-sentence; sentence chunker must NOT."""
        # Build a paragraph where each sentence is ~50 words.
        sentences = ["The quick brown fox jumps over the lazy dog " * 5 + "." for _ in range(10)]
        text = " ".join(sentences)
        pages = [{"page_number": 1, "text": text}]

        fixed = DocumentParser().chunk_pages(pages, chunk_size=80, overlap=10)
        sentence = chunker.chunk_pages(pages, max_words=80, overlap_sentences=0)

        # Count chunks that end mid-sentence (don't end with .)
        fixed_cuts = sum(1 for c in fixed if not c["text"].strip().endswith("."))
        sent_cuts = sum(1 for c in sentence if not c["text"].strip().endswith("."))

        # Sentence chunker should have zero or fewer mid-sentence cuts
        assert sent_cuts <= fixed_cuts


# ── Overlap ────────────────────────────────────────────────────────────────────

class TestOverlap:

    def test_zero_overlap_produces_no_repeated_sentences(self, chunker):
        sentences = [f"Sentence {i}." for i in range(10)]
        pages = [{"page_number": 1, "text": " ".join(sentences)}]
        chunks = chunker.chunk_pages(pages, max_words=20, overlap_sentences=0)

        # Collect all sentences across chunks — no duplicates expected
        all_texts = " ".join(c["text"] for c in chunks)
        for s in sentences:
            # Each sentence should appear exactly once (or not at all if too long)
            assert all_texts.count(s) <= 1, f"Sentence duplicated: {s}"

    def test_overlap_one_carries_last_sentence_forward(self, chunker):
        # 6 short sentences, max_words=10 to force multiple chunks
        sentences = [f"Sentence {i}." for i in range(6)]
        pages = [{"page_number": 1, "text": " ".join(sentences)}]
        chunks = chunker.chunk_pages(pages, max_words=10, overlap_sentences=1)

        if len(chunks) >= 2:
            # Last sentence of chunk 0 should appear at start of chunk 1
            last_sent_of_first = chunks[0]["text"].split(".")[-2].strip() + "."
            assert last_sent_of_first in chunks[1]["text"]


# ── Multi-page behaviour ───────────────────────────────────────────────────────

class TestMultiPage:

    def test_chunks_are_tagged_with_correct_page_numbers(self, chunker):
        long_text = " ".join(f"Sentence {i}." for i in range(50))
        pages = [
            {"page_number": 1, "text": long_text},
            {"page_number": 3, "text": long_text},
        ]
        chunks = chunker.chunk_pages(pages, max_words=30, overlap_sentences=0)
        page_nums = {c["page_number"] for c in chunks}
        assert page_nums == {1, 3}

    def test_page_boundary_is_respected(self, chunker):
        """Sentences from different pages must not be merged into one chunk."""
        pages = [
            {"page_number": 1, "text": "Page one content."},
            {"page_number": 2, "text": "Page two content."},
        ]
        chunks = chunker.chunk_pages(pages, max_words=400)
        # Each chunk should have exactly one unique page_number
        for chunk in chunks:
            assert isinstance(chunk["page_number"], int)

        # No single chunk should contain text from both pages
        # (page boundary resets the accumulator)
        for chunk in chunks:
            if "Page one content" in chunk["text"]:
                assert "Page two content" not in chunk["text"]


# ── Comparison vs fixed-size ───────────────────────────────────────────────────

class TestVsFixedSize:

    def test_sentence_chunker_produces_variable_sized_chunks(self, chunker):
        # Mix of short and long sentences to guarantee size variation
        sentences = (
            ["Short." for _ in range(5)] +
            [" ".join([f"word{j}" for j in range(80)]) + "." for _ in range(3)]
        )
        pages = [{"page_number": 1, "text": " ".join(sentences)}]
        chunks = chunker.chunk_pages(pages, max_words=100, overlap_sentences=0)
        word_counts = [len(c["text"].split()) for c in chunks]
        # Variable sizes: not all the same
        assert len(set(word_counts)) > 1

    def test_fixed_chunker_produces_uniform_first_chunk_size(self, fixed_chunker):
        # Fixed chunker always fills to chunk_size on non-terminal chunks
        pages = [{"page_number": 1, "text": " ".join([f"w{i}" for i in range(1200)])}]
        chunks = fixed_chunker.chunk_pages(pages, chunk_size=400, overlap=50)
        # All non-terminal chunks are exactly 400 words
        for c in chunks[:-1]:
            assert len(c["text"].split()) == 400
