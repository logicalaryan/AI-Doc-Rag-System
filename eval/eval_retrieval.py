"""
eval/eval_retrieval.py — Retrieval Quality Metrics.

Metrics computed:
  - Hit Rate   : fraction of questions where the correct chunk appears in top-k
  - MRR        : Mean Reciprocal Rank — average of 1/rank across questions
  - Precision@k: fraction of returned chunks that are relevant, averaged across questions

Usage:
    python eval/eval_retrieval.py
    python eval/eval_retrieval.py --k 5 --cases eval/test_cases.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retriever import load_vectorstore, retrieve


def _chunk_is_relevant(chunk_text: str, expected_contains: str) -> bool:
    """Case-insensitive substring match — simple but effective for demos."""
    return expected_contains.lower() in chunk_text.lower()


def compute_hit_rate(results: List[Dict[str, Any]]) -> float:
    """
    Hit Rate = (questions where correct chunk is in top-k) / total questions.
    """
    if not results:
        return 0.0
    hits = sum(1 for r in results if r["hit"])
    return hits / len(results)


def compute_mrr(results: List[Dict[str, Any]]) -> float:
    """
    MRR = average of 1/rank for the first relevant result per question.
    If no relevant result found, reciprocal rank = 0.
    """
    if not results:
        return 0.0
    reciprocal_ranks = []
    for r in results:
        rank = r.get("rank")  # 1-based rank of first relevant chunk, or None
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def compute_precision_at_k(results: List[Dict[str, Any]], k: int) -> float:
    """
    Precision@k = (relevant chunks in top-k) / k, averaged across questions.
    """
    if not results:
        return 0.0
    precisions = []
    for r in results:
        relevant_in_top_k = r.get("relevant_in_top_k", 0)
        precisions.append(relevant_in_top_k / k)
    return sum(precisions) / len(precisions)


def evaluate_retrieval(
    test_cases_path: str,
    vectorstore=None,
    k: int = 3,
    persist_dir: str | None = None,
) -> Dict[str, Any]:
    """
    Run retrieval evaluation over all test cases.

    Args:
        test_cases_path: Path to the JSON file with ground truth cases.
        vectorstore:     Pre-loaded Chroma instance (optional).
        k:               Number of chunks to retrieve per question.
        persist_dir:     ChromaDB directory (used if vectorstore is None).

    Returns:
        Dict with per-question results and aggregate metrics.
    """
    from app.ingest import CHROMA_PERSIST_DIR
    persist_dir = persist_dir or CHROMA_PERSIST_DIR

    with open(test_cases_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    if vectorstore is None:
        vectorstore = load_vectorstore(persist_dir)

    per_question = []

    for case in test_cases:
        question = case["question"]
        expected = case["expected_chunk_contains"]

        retrieved_docs = retrieve(question, vectorstore=vectorstore, k=k)
        chunks = [doc.page_content for doc in retrieved_docs]

        # Find rank of first relevant chunk (1-based)
        rank = None
        relevant_count = 0
        for i, chunk in enumerate(chunks, 1):
            if _chunk_is_relevant(chunk, expected):
                relevant_count += 1
                if rank is None:
                    rank = i

        per_question.append({
            "question": question,
            "expected_chunk_contains": expected,
            "hit": rank is not None,
            "rank": rank,
            "relevant_in_top_k": relevant_count,
            "retrieved_chunks": chunks,
        })

    hit_rate = compute_hit_rate(per_question)
    mrr = compute_mrr(per_question)
    precision = compute_precision_at_k(per_question, k)

    return {
        "k": k,
        "total": len(per_question),
        "hit_rate": hit_rate,
        "mrr": mrr,
        f"precision_at_{k}": precision,
        "per_question": per_question,
    }


def print_retrieval_report(report: Dict[str, Any]) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    k = report["k"]
    print(f"\nRetrieval Evaluation (k={k}, n={report['total']}):")
    print(f"  Hit Rate:       {report['hit_rate']:.2f}")
    print(f"  MRR:            {report['mrr']:.2f}")
    print(f"  Precision@{k}:   {report[f'precision_at_{k}']:.2f}")
    print()
    for r in report["per_question"]:
        status = "✅" if r["hit"] else "❌"
        rank_str = f"rank={r['rank']}" if r["rank"] else "not found"
        print(f"  {status} Q: {r['question'][:60]!r}  ({rank_str})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument("--k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument(
        "--cases",
        default="eval/test_cases.json",
        help="Path to test_cases.json",
    )
    parser.add_argument(
        "--persist-dir",
        default=None,
        help="Path to ChromaDB directory (overrides .env)",
    )
    args = parser.parse_args()

    report = evaluate_retrieval(
        test_cases_path=args.cases,
        k=args.k,
        persist_dir=args.persist_dir,
    )
    print_retrieval_report(report)
