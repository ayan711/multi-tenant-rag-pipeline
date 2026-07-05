"""
verify_pipeline.py — Task 2.5: In-Memory Ingestion Verification

Pipes a local PDF through the full extraction → embedding pipeline and
prints shape/content stats so you can confirm everything wired up correctly
before hooking up ChromaDB and Celery.

Usage:
    python scripts/verify_pipeline.py path/to/file.pdf
    python scripts/verify_pipeline.py path/to/file.pdf --chunk-size 200 --overlap 25
"""

import argparse
import sys
import time
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.embedder import LocalEmbedder
from app.parser import DocumentParser


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the parser → embedder pipeline.")
    parser.add_argument("pdf", help="Path to a PDF file to ingest.")
    parser.add_argument("--chunk-size", type=int, default=400, help="Words per chunk (default: 400).")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap words between chunks (default: 50).")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"[error] File not found: {pdf_path}")
        sys.exit(1)
    if pdf_path.suffix.lower() != ".pdf":
        print(f"[error] Expected a .pdf file, got: {pdf_path.suffix}")
        sys.exit(1)

    print(f"\n{'─' * 60}")
    print(f"  File        : {pdf_path.name}")
    print(f"  Chunk size  : {args.chunk_size} words  |  Overlap: {args.overlap} words")
    print(f"{'─' * 60}\n")

    # ── Step 1: Extraction ────────────────────────────────────────────────────
    print("[ 1/3 ] Extracting text from PDF...")
    t0 = time.perf_counter()
    doc_parser = DocumentParser()
    pages = doc_parser.extract_text_per_page(str(pdf_path))
    print(f"        → {len(pages)} non-blank page(s) extracted  ({time.perf_counter() - t0:.2f}s)\n")

    if not pages:
        print("[warn] No text found — the PDF may be image-only or encrypted. Exiting.")
        sys.exit(0)

    # ── Step 2: Chunking ──────────────────────────────────────────────────────
    print("[ 2/3 ] Chunking pages...")
    t0 = time.perf_counter()
    chunks = doc_parser.chunk_pages(pages, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"        → {len(chunks)} chunk(s) produced  ({time.perf_counter() - t0:.2f}s)")
    print(f"        → chunk_index range: 0 – {chunks[-1]['chunk_index']}\n")

    # ── Step 3: Embedding ─────────────────────────────────────────────────────
    print("[ 3/3 ] Loading embedder and vectorising chunks...")
    print("        (first run downloads model weights ~80 MB — cached after that)")
    t0 = time.perf_counter()
    embedder = LocalEmbedder()
    texts = [c["text"] for c in chunks]
    vectors = embedder.embed_batch(texts)
    elapsed = time.perf_counter() - t0

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("  PIPELINE RESULTS")
    print(f"{'─' * 60}")
    print(f"  Embedding dim   : {embedder.embedding_dim}")
    print(f"  Vectors produced: {len(vectors)}")
    print(f"  Each vector len : {len(vectors[0])}  (expected {embedder.embedding_dim})")
    print(f"  Embed time      : {elapsed:.2f}s  ({elapsed / len(vectors) * 1000:.1f} ms/chunk)")
    print(f"\n  First chunk preview  : \"{texts[0][:80]}...\"")
    print(f"  First vector ([:5])  : {[round(v, 5) for v in vectors[0][:5]]}")
    print(f"  Last vector  ([:5])  : {[round(v, 5) for v in vectors[-1][:5]]}")
    print(f"{'─' * 60}\n")

    # Basic sanity assertions — fail loudly if something is wrong
    assert len(vectors) == len(chunks), "vector count != chunk count"
    assert all(len(v) == embedder.embedding_dim for v in vectors), "unexpected vector dimension"
    print("[ok] All assertions passed — pipeline is working correctly.\n")


if __name__ == "__main__":
    main()
