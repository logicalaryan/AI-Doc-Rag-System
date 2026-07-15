"""
eval/eval_generation.py — Answer Quality Metrics.

Metrics computed:
  - Faithfulness    : fraction of answers that contain only information
                      present in the expected answer keywords (no hallucination)
  - Answer Relevancy: fraction of answers that actually relate to the question

Both use simple keyword/substring matching — no paid APIs required.
Good enough to catch obvious hallucinations and off-topic answers in a demo.

Usage:
    python eval/eval_generation.py
    python eval/eval_generation.py --cases eval/test_cases.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.chain import ask
from app.retriever import load_vectorstore


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def is_faithful(answer: str, expected_contains: str) -> bool:
    """
    A simple faithfulness check:
    The expected_answer_contains string must appear in the answer.
    If it doesn't, the answer either hallucinated or missed the fact entirely.

    This is a binary pass/fail per question. For portfolio purposes, this is
    sufficient. RAGAS / G-Eval would give a graded score but require an LLM judge.
    """
    return expected_contains.lower() in answer.lower()


def is_relevant(answer: str, question: str) -> bool:
    """
    A simple relevancy check:
    At least one meaningful word from the question must appear in the answer.
    Filters out common stop words to avoid trivial matches.
    """
    STOP_WORDS = {
        "what", "when", "where", "who", "how", "is", "was", "the",
        "a", "an", "are", "were", "did", "do", "does", "has", "have",
        "of", "in", "on", "at", "to", "for", "and", "or", "by",
    }
    question_words = [
        w.lower().strip("?.,") for w in question.split()
        if w.lower().strip("?.,") not in STOP_WORDS and len(w) > 2
    ]
    answer_lower = answer.lower()
    return any(word in answer_lower for word in question_words)


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_generation(
    test_cases_path: str,
    vectorstore=None,
    llm=None,
    k: int = 3,
    persist_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run generation evaluation over all test cases.

    For each test case:
      - Calls ask() to generate a real answer
      - Checks Faithfulness and Answer Relevancy

    Args:
        test_cases_path: Path to the JSON ground truth file.
        vectorstore:     Pre-loaded Chroma instance (optional).
        llm:             Pre-built LLM (optional — useful for passing mocks).
        k:               Retrieval top-k.
        persist_dir:     ChromaDB directory.

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
        expected_answer = case["expected_answer_contains"]

        result = ask(question, vectorstore=vectorstore, llm=llm, k=k)
        answer = result.answer

        faithful = is_faithful(answer, expected_answer)
        relevant = is_relevant(answer, question)

        per_question.append({
            "question": question,
            "expected_answer_contains": expected_answer,
            "answer": answer,
            "faithful": faithful,
            "relevant": relevant,
        })

    total = len(per_question)
    faithfulness_score = sum(r["faithful"] for r in per_question) / total if total else 0.0
    relevancy_score = sum(r["relevant"] for r in per_question) / total if total else 0.0

    return {
        "total": total,
        "faithfulness": faithfulness_score,
        "answer_relevancy": relevancy_score,
        "per_question": per_question,
    }


def print_generation_report(report: Dict[str, Any]) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print(f"\nGeneration Evaluation (n={report['total']}):")
    print(f"  Faithfulness:     {report['faithfulness']:.2f}")
    print(f"  Answer Relevancy: {report['answer_relevancy']:.2f}")
    print()
    for r in report["per_question"]:
        f_icon = "✅" if r["faithful"] else "❌"
        rel_icon = "✅" if r["relevant"] else "❌"
        print(f"  Q: {r['question'][:55]!r}")
        print(f"     Faithful: {f_icon}  Relevant: {rel_icon}")
        print(f"     Answer: {r['answer'][:80]!r}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG generation quality")
    parser.add_argument(
        "--cases",
        default="eval/test_cases.json",
        help="Path to test_cases.json",
    )
    parser.add_argument("--k", type=int, default=3, help="Retrieval top-k")
    parser.add_argument(
        "--persist-dir",
        default=None,
        help="Path to ChromaDB directory (overrides .env)",
    )
    args = parser.parse_args()

    report = evaluate_generation(
        test_cases_path=args.cases,
        k=args.k,
        persist_dir=args.persist_dir,
    )
    print_generation_report(report)
