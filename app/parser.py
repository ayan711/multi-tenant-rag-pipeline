# ─────────────────────────────────────────────────────────────────────────────
# parser.py — PDF Document Parser & Text Chunker
# ─────────────────────────────────────────────────────────────────────────────
#
# Responsible for two sequential operations:
#
#   1. EXTRACTION — Open a PDF file and pull out raw text, page by page,
#      using PyMuPDF (imported as `fitz`). PyMuPDF is chosen over alternatives
#      like pdfminer because it's faster and handles complex layouts better.
#
#   2. CHUNKING — Split the extracted text into overlapping windows of fixed
#      word count. Why overlapping? If a sentence spans two chunk boundaries,
#      a pure fixed-split would cut it in half and neither chunk would contain
#      the full thought. Overlap ensures continuity at boundaries.
#
# Output of this module feeds directly into embedder.py — each chunk becomes
# one vector in the database.
#
# Implemented in: Task 2.1 (extraction) and Task 2.2 (chunking).

import re

import fitz  # PyMuPDF — the import name is `fitz` even though the package is `pymupdf`
import nltk
from nltk.tokenize import sent_tokenize

# punkt_tab is the model backing sent_tokenize in nltk >= 3.9
try:
    sent_tokenize("warm up.")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


class DocumentParser:
    """
    Extracts clean, structured text from PDF files page by page.

    Designed to be instantiated once and reused for multiple files — there is
    no per-instance state, so a single parser object is safe to use throughout
    the app's lifetime.
    """

    def extract_text_per_page(self, file_path: str) -> list[dict]:
        """
        Open a PDF and return its text as a list of per-page dictionaries.
            {
                "page_number": int,   # 1-indexed (human-friendly, matches PDF viewers)
                "text": str           # cleaned text for that page
            }

        Pages that are blank or contain only whitespace are skipped so
        downstream chunking doesn't process empty chunks.

        Args:
            file_path: Absolute or relative path to a .pdf file on disk.
        Returns:
            A list of page dicts, in document order.
        Raises:
            fitz.FileNotFoundError: If the path points to a non-existent file.
            fitz.FileDataError:     If the file is not a valid PDF.
        """
        pages = []

        # fitz.open() returns a Document object. Using it as a context manager
        # guarantees the file handle is closed even if an exception is raised
        # mid-loop — important for long-running server processes that process
        # many PDFs without restarting.
        with fitz.open(file_path) as doc:

            # `doc` is iterable; each iteration yields one `Page` object.
            for page_index, page in enumerate(doc):

                # `get_text("text")` extracts the text layer of a PDF page as
                # a plain string, preserving newlines that separate lines.
                raw_text = page.get_text("text")

                # Clean up the raw extracted text so the downstream embedder
                # doesn't waste dimensions on formatting noise.
                cleaned = self._clean_text(raw_text)

                # Skip pages that are blank or decoration-only (cover images,
                # divider pages, etc.). An empty chunk sent to the embedder
                # would produce a near-zero vector with no semantic meaning.
                if not cleaned:
                    continue

                pages.append({
                    "page_number": page_index + 1,  # convert 0-based → 1-based
                    "text": cleaned,
                })

        return pages

    def chunk_pages(
        self,
        pages: list[dict],
        chunk_size: int = 400,
        overlap: int = 50,
    ) -> list[dict]:
        """
        Split a list of page dicts (from extract_text_per_page) into overlapping word windows.

        Each output dict contains:
            {
                "text":         str,  # the chunk's words joined by spaces
                "page_number":  int,  # source page this chunk came from
                "chunk_index":  int,  # globally sequential index across all pages (0-based)
            }

        The overlap ensures that a sentence crossing a chunk boundary appears in both
        the trailing chunk and the leading chunk — neither chunk loses the full thought.

        Args:
            pages:       Output of extract_text_per_page(). Empty pages are tolerated (skipped).
            chunk_size:  Maximum words per chunk.
            overlap:     Words shared between consecutive chunks on the same page.
        Returns:
            Flat list of chunk dicts in document order.
        """
        chunks = []
        chunk_index = 0
        # step is how many words we advance the window start each iteration.
        # A step smaller than chunk_size is what creates the overlap.
        step = chunk_size - overlap

        for page in pages:
            words = page["text"].split()
            if not words:
                continue

            start = 0
            while start < len(words):
                end = start + chunk_size
                chunks.append({
                    "text": " ".join(words[start:end]),
                    "page_number": page["page_number"],
                    "chunk_index": chunk_index,
                })
                chunk_index += 1

                # Stop once the window reaches or passes the last word.
                if end >= len(words):
                    break

                start += step

        return chunks

    # ── Private helpers ────────────────────────────────────────────────────────

    def _clean_text(self, raw: str) -> str:
        """
        Normalize whitespace and remove low-signal noise from extracted text.

        Steps applied in order:
          1. Rejoin hyphenated line-breaks (e.g. "impor-\\ntant" → "important").
             PDFs hyphenate words that wrap at the right margin; without this
             fix the embedder sees "impor" and "tant" as separate tokens.
          2. Collapse runs of whitespace (spaces, tabs, multiple newlines) into
             a single space so the downstream word-count logic is predictable.
          3. Strip leading and trailing whitespace from the final string.

        Returns an empty string if the page had no meaningful content.
        """
        # Step 1 — rejoin hyphenated line breaks.
        # The regex matches a hyphen immediately followed by a newline,
        # optionally surrounded by spaces, and replaces the whole thing with
        # nothing (merging the two word-halves together).
        text = re.sub(r"-\n\s*", "", raw)

        # Step 2 — collapse all whitespace runs (including newlines) into a
        # single space. This converts multi-line page text into one continuous
        # string, which is what the word-based chunker in Task 2.2 expects.
        text = re.sub(r"\s+", " ", text)

        # Step 3 — remove leading/trailing whitespace from the final result.
        return text.strip()


class SentenceChunker:
    """
    Groups sentences into chunks without cutting mid-sentence.

    Uses nltk.sent_tokenize to split each page into discrete sentences, then
    accumulates them greedily until adding the next sentence would exceed
    max_words. The last overlap_sentences of each chunk are prepended to the
    next chunk so context carries across boundaries.

    Contrast with DocumentParser.chunk_pages():
      - Fixed-size: always exact word counts, may split mid-sentence.
      - Sentence-aware: variable word counts, never splits mid-sentence.
    """

    def chunk_pages(
        self,
        pages: list[dict],
        max_words: int = 400,
        overlap_sentences: int = 1,
    ) -> list[dict]:
        """
        Split page dicts into sentence-aligned chunks.

        Args:
            pages:              Output of DocumentParser.extract_text_per_page().
            max_words:          Soft upper bound on words per chunk.
                                A single sentence longer than this will still form
                                its own chunk rather than being discarded.
            overlap_sentences:  Number of trailing sentences from each chunk to
                                prepend to the next chunk for context continuity.
        Returns:
            Flat list of chunk dicts: {text, page_number, chunk_index}.
        """
        chunks = []
        chunk_index = 0

        for page in pages:
            sentences = sent_tokenize(page["text"])
            if not sentences:
                continue

            carry: list[str] = []  # overlap sentences carried from previous chunk
            current: list[str] = []
            current_words = 0

            for sentence in sentences:
                sentence_words = len(sentence.split())

                # Flush the current accumulator when adding this sentence would
                # exceed max_words — but only if we already have content.
                if current_words + sentence_words > max_words and current:
                    chunks.append({
                        "text": " ".join(current),
                        "page_number": page["page_number"],
                        "chunk_index": chunk_index,
                    })
                    chunk_index += 1
                    # Carry the tail sentences forward for overlap
                    carry = current[-overlap_sentences:] if overlap_sentences else []
                    current = carry + [sentence]
                    current_words = sum(len(s.split()) for s in current)
                else:
                    current.append(sentence)
                    current_words += sentence_words

            # Emit whatever remains after the last sentence on this page.
            if current:
                chunks.append({
                    "text": " ".join(current),
                    "page_number": page["page_number"],
                    "chunk_index": chunk_index,
                })
                chunk_index += 1

        return chunks
