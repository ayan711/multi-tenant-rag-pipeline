"""
compare_chunking.py — Task 2.6: Chunking Strategy Comparison

Runs the same PDF through two chunking strategies:
  1. Fixed-size (word-count sliding window, DocumentParser.chunk_pages)
  2. Sentence-boundary (NLTK sent_tokenize, SentenceChunker.chunk_pages)

For each strategy, embeds all chunks with all-MiniLM-L6-v2 and retrieves
the top-3 by cosine similarity for 5 test questions. Prints a side-by-side
table so you can see where each strategy wins or loses.

Usage:
    python scripts/compare_chunking.py sample_data/iso27001.pdf
    python scripts/compare_chunking.py path/to/any.pdf --top-k 5
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.embedder import LocalEmbedder
from app.parser import DocumentParser, SentenceChunker

# ── Test questions ─────────────────────────────────────────────────────────────
#
# These are chosen to cover different retrieval difficulties:
#   Q1 — broad definitional question (answer likely repeated in several places)
#   Q2 — procedural question (answer in specific numbered clauses)
#   Q3 — risk-specific question (specialised term, less repeated)
#   Q4 — relationship/scope question (cross-cutting concept)
#   Q5 — control-specific question (technical detail in an annex)

TEST_QUESTIONS = [
    "What is the purpose and scope of Constituency?",
    "Which specific Article guarantees equality before the law?",
    "Which specific Article guarantees equality before the law?",
    "Can the Supreme Court's original jurisdiction be invoked for a dispute between two individual citizens? Why or why not?",
    "According to the Constitution, who is the current serving Chief Justice of India?"
    ]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def retrieve_top_k(
    query_vec: list[float],
    chunk_vecs: list[list[float]],
    chunks: list[dict],
    k: int,
) -> list[tuple[float, dict]]:
    """Return the top-k (score, chunk) pairs sorted by cosine similarity."""
    scored = [(cosine_similarity(query_vec, cv), c) for cv, c in zip(chunk_vecs, chunks)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


def wrap(text: str, width: int = 90) -> str:
    """Hard-wrap text for display without breaking words mid-token."""
    words = text.split()
    lines, line = [], []
    length = 0
    for w in words:
        if length + len(w) + 1 > width and line:
            lines.append(" ".join(line))
            line, length = [w], len(w)
        else:
            line.append(w)
            length += len(w) + 1
    if line:
        lines.append(" ".join(line))
    return "\n    ".join(lines)


def run_strategy(
    name: str,
    chunks: list[dict],
    embedder: LocalEmbedder,
    questions: list[str],
    top_k: int,
) -> dict:
    """Embed chunks and retrieve top-k for each question. Returns timing + results."""
    t0 = time.perf_counter()
    texts = [c["text"] for c in chunks]
    chunk_vecs = embedder.embed_batch(texts)
    embed_time = time.perf_counter() - t0

    results = []
    for q in questions:
        q_vec = embedder.embed(q)
        hits = retrieve_top_k(q_vec, chunk_vecs, chunks, top_k)
        results.append(hits)

    word_counts = [len(c["text"].split()) for c in chunks]
    return {
        "name": name,
        "n_chunks": len(chunks),
        "avg_words": sum(word_counts) / len(word_counts) if word_counts else 0,
        "min_words": min(word_counts) if word_counts else 0,
        "max_words": max(word_counts) if word_counts else 0,
        "embed_time": embed_time,
        "results": results,  # list[list[(score, chunk)]]
    }


def print_stats(stat_a: dict, stat_b: dict) -> None:
    print(f"\n{'═' * 100}")
    print(f"  CHUNKING STRATEGY COMPARISON")
    print(f"{'═' * 100}")
    header = f"  {'Metric':<25} {'Fixed-size':>18}  {'Sentence-boundary':>20}"
    print(header)
    print(f"  {'─' * 65}")
    rows = [
        ("Chunks produced", f"{stat_a['n_chunks']}", f"{stat_b['n_chunks']}"),
        ("Avg words / chunk", f"{stat_a['avg_words']:.0f}", f"{stat_b['avg_words']:.0f}"),
        ("Min words / chunk", f"{stat_a['min_words']}", f"{stat_b['min_words']}"),
        ("Max words / chunk", f"{stat_a['max_words']}", f"{stat_b['max_words']}"),
        ("Embed time (s)", f"{stat_a['embed_time']:.2f}", f"{stat_b['embed_time']:.2f}"),
    ]
    for label, va, vb in rows:
        print(f"  {label:<25} {va:>18}  {vb:>20}")


def print_retrieval(questions: list[str], stat_a: dict, stat_b: dict, top_k: int) -> None:
    print(f"\n{'═' * 100}")
    print(f"  RETRIEVAL RESULTS  (top {top_k} per strategy)")
    print(f"{'═' * 100}")

    for i, (q, hits_a, hits_b) in enumerate(zip(questions, stat_a["results"], stat_b["results"]), 1):
        print(f"\n  Q{i}: {q}")
        print(f"  {'─' * 96}")

        for rank in range(top_k):
            score_a, chunk_a = hits_a[rank]
            score_b, chunk_b = hits_b[rank]

            preview_a = wrap(chunk_a["text"][:300] + ("…" if len(chunk_a["text"]) > 300 else ""))
            preview_b = wrap(chunk_b["text"][:300] + ("…" if len(chunk_b["text"]) > 300 else ""))

            print(f"\n  Rank {rank + 1}")
            print(f"    [Fixed-size   score={score_a:.4f}  pg={chunk_a['page_number']:>3}  "
                  f"words={len(chunk_a['text'].split()):>4}]")
            print(f"    {preview_a}")
            print()
            print(f"    [Sentence-bnd score={score_b:.4f}  pg={chunk_b['page_number']:>3}  "
                  f"words={len(chunk_b['text'].split()):>4}]")
            print(f"    {preview_b}")

        print()


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare fixed-size vs sentence-boundary chunking.")
    ap.add_argument("pdf", help="Path to a PDF file.")
    ap.add_argument("--top-k", type=int, default=3, help="Hits to retrieve per question (default: 3).")
    ap.add_argument("--chunk-size", type=int, default=400, help="Words per fixed chunk (default: 400).")
    ap.add_argument("--overlap", type=int, default=50, help="Overlap words for fixed chunker (default: 50).")
    ap.add_argument("--max-words", type=int, default=400, help="Max words for sentence chunker (default: 400).")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"[error] File not found: {pdf_path}")
        sys.exit(1)

    print(f"\n{'─' * 60}")
    print(f"  File : {pdf_path.name}")
    print(f"  Fixed-size — chunk_size={args.chunk_size}, overlap={args.overlap}")
    print(f"  Sentence   — max_words={args.max_words}, overlap_sentences=1")
    print(f"{'─' * 60}\n")

    # ── Extract ────────────────────────────────────────────────────────────────
    print("[ 1/4 ] Extracting text...")
    doc_parser = DocumentParser()
    pages = doc_parser.extract_text_per_page(str(pdf_path))
    print(f"        → {len(pages)} pages\n")

    # ── Chunk ─────────────────────────────────────────────────────────────────
    print("[ 2/4 ] Chunking with both strategies...")
    fixed_chunks = doc_parser.chunk_pages(pages, chunk_size=args.chunk_size, overlap=args.overlap)
    sent_chunks = SentenceChunker().chunk_pages(pages, max_words=args.max_words, overlap_sentences=1)
    print(f"        → fixed-size: {len(fixed_chunks)} chunks")
    print(f"        → sentence  : {len(sent_chunks)} chunks\n")

    # ── Load embedder ─────────────────────────────────────────────────────────
    print("[ 3/4 ] Loading embedder (may download model weights on first run)...")
    embedder = LocalEmbedder()
    print(f"        → {embedder.MODEL_NAME}  dim={embedder.embedding_dim}\n")

    # ── Embed + retrieve ──────────────────────────────────────────────────────
    print("[ 4/4 ] Embedding and retrieving...")
    stat_fixed = run_strategy("Fixed-size", fixed_chunks, embedder, TEST_QUESTIONS, args.top_k)
    stat_sent = run_strategy("Sentence", sent_chunks, embedder, TEST_QUESTIONS, args.top_k)
    print()

    # ── Report ────────────────────────────────────────────────────────────────
    print_stats(stat_fixed, stat_sent)
    print_retrieval(TEST_QUESTIONS, stat_fixed, stat_sent, args.top_k)

    print(f"{'═' * 100}")
    print("  Run complete. Review the retrieved snippets above to judge semantic coherence.")
    print(f"{'═' * 100}\n")


if __name__ == "__main__":
    main()
