"""
eval_retrieval_quality.py — Task 6.5: Retrieval Quality Evaluation

Runs 5 known-answer questions against the already-ingested ChromaDB data
for one tenant, comparing search_tenant_context() at n_results=3 vs
n_results=8. For each question we check whether a keyword we know appears
in the source document shows up anywhere in the retrieved chunks — a cheap
proxy for "did retrieval actually surface the answer", without needing an
LLM call to grade correctness.

Prerequisite: the target document must already be ingested for the given
tenant (see Task 6.2's upload smoke test).

Usage:
    python scripts/eval_retrieval_quality.py --tenant acme
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.embedder import LocalEmbedder
from app.vector_db import VectorStoreManager

# ── Test questions ───────────────────────────────────────────────────────────
#
# Each question is paired with a keyword/phrase known to appear verbatim in
# the ingested ISO/IEC 27001:2022 text near its answer. If the keyword shows
# up in a retrieved chunk, we count that question as "hit" at that n_results.
TEST_CASES = [
    {
        "question": "What is the purpose of an information security management system?",
        "expect_keyword": "confidentiality, integrity and availability",
    },
    {
        "question": "What must top management's information security policy include?",
        "expect_keyword": "information security objectives",
    },
    {
        "question": "What does Annex A of the standard contain?",
        "expect_keyword": "Information security controls",
    },
    {
        "question": "What must organizations establish for internal audits?",
        "expect_keyword": "audit programme",
    },
    {
        "question": "What must top management do during a management review?",
        "expect_keyword": "management review",
    },
]


def run_eval(vector_store: VectorStoreManager, embedder: LocalEmbedder, tenant_id: str, n_results: int) -> list[dict]:
    results = []
    for case in TEST_CASES:
        query_vec = embedder.embed(case["question"])
        t0 = time.perf_counter()
        chunks = vector_store.search_tenant_context(query_vec, tenant_id, n_results=n_results)
        elapsed = time.perf_counter() - t0

        hit = any(case["expect_keyword"].lower() in c["text"].lower() for c in chunks)
        distances = [c["distance"] for c in chunks]
        results.append(
            {
                "question": case["question"],
                "expect_keyword": case["expect_keyword"],
                "hit": hit,
                "n_returned": len(chunks),
                "min_distance": min(distances) if distances else None,
                "max_distance": max(distances) if distances else None,
                "elapsed": elapsed,
            }
        )
    return results


def print_report(results_low: list[dict], results_high: list[dict], n_low: int, n_high: int) -> None:
    print(f"\n{'═' * 100}")
    print(f"  RETRIEVAL QUALITY — n_results={n_low} vs n_results={n_high}")
    print(f"{'═' * 100}")

    hits_low = sum(r["hit"] for r in results_low)
    hits_high = sum(r["hit"] for r in results_high)
    avg_time_low = sum(r["elapsed"] for r in results_low) / len(results_low)
    avg_time_high = sum(r["elapsed"] for r in results_high) / len(results_high)

    print(f"  {'Metric':<28} {f'n={n_low}':>12} {f'n={n_high}':>12}")
    print(f"  {'-' * 54}")
    print(f"  {'Questions answered (hit)':<28} {f'{hits_low}/{len(results_low)}':>12} {f'{hits_high}/{len(results_high)}':>12}")
    print(f"  {'Avg query latency (s)':<28} {avg_time_low:>12.4f} {avg_time_high:>12.4f}")

    print(f"\n  {'Per-question breakdown':}")
    print(f"  {'-' * 96}")
    for i, (q, r_low, r_high) in enumerate(zip(TEST_CASES, results_low, results_high), 1):
        mark_low = "✓" if r_low["hit"] else "✗"
        mark_high = "✓" if r_high["hit"] else "✗"
        print(f"  Q{i}: {q['question']}")
        print(f"       expects: \"{q['expect_keyword']}\"")
        print(f"       n={n_low}: {mark_low}  (dist {r_low['min_distance']:.3f}-{r_low['max_distance']:.3f})   "
              f"n={n_high}: {mark_high}  (dist {r_high['min_distance']:.3f}-{r_high['max_distance']:.3f})")
        print()

    print(f"{'═' * 100}")
    print("  Precision/recall tradeoff: a smaller n_results is cheaper and less likely to add")
    print("  irrelevant context, but a larger n_results catches answers ranked lower — visible")
    print("  above whenever a '✗' at n_low flips to '✓' at n_high.")
    print(f"{'═' * 100}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare retrieval quality at different n_results.")
    ap.add_argument("--tenant", default="acme", help="Tenant ID whose ingested docs to query (default: acme).")
    ap.add_argument("--n-low", type=int, default=3, help="Smaller n_results value (default: 3).")
    ap.add_argument("--n-high", type=int, default=8, help="Larger n_results value (default: 8).")
    args = ap.parse_args()

    print("[ 1/2 ] Loading embedder and connecting to ChromaDB...")
    embedder = LocalEmbedder()
    vector_store = VectorStoreManager()

    print(f"[ 2/2 ] Running {len(TEST_CASES)} questions at n_results={args.n_low} and n_results={args.n_high}...")
    results_low = run_eval(vector_store, embedder, args.tenant, args.n_low)
    results_high = run_eval(vector_store, embedder, args.tenant, args.n_high)

    print_report(results_low, results_high, args.n_low, args.n_high)


if __name__ == "__main__":
    main()
