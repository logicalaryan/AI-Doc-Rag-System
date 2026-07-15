"""
eval/eval_report.py — Full Evaluation Runner.

Runs both retrieval and generation evaluations, then prints a combined
scored report. Also saves the report to eval/last_report.json so you
can track progress over time in DECISIONS.md.

Usage:
    python eval/eval_report.py
    python eval/eval_report.py --k 5 --cases eval/test_cases.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.eval_retrieval import evaluate_retrieval, print_retrieval_report
from eval.eval_generation import evaluate_generation, print_generation_report


def compute_overall_score(retrieval_report: dict, generation_report: dict, k: int) -> float:
    """
    Overall score = equal-weighted average of all five metrics:
      Hit Rate, MRR, Precision@k, Faithfulness, Answer Relevancy
    """
    scores = [
        retrieval_report["hit_rate"],
        retrieval_report["mrr"],
        retrieval_report.get(f"precision_at_{k}", 0.0),
        generation_report["faithfulness"],
        generation_report["answer_relevancy"],
    ]
    return sum(scores) / len(scores)


def run_full_evaluation(
    test_cases_path: str = "eval/test_cases.json",
    k: int = 3,
    persist_dir: str | None = None,
    vectorstore=None,
    llm=None,
) -> dict:
    """
    Run retrieval + generation evals and return a combined report dict.
    """
    print("Running retrieval evaluation...")
    retrieval = evaluate_retrieval(
        test_cases_path=test_cases_path,
        vectorstore=vectorstore,
        k=k,
        persist_dir=persist_dir,
    )

    print("Running generation evaluation...")
    generation = evaluate_generation(
        test_cases_path=test_cases_path,
        vectorstore=vectorstore,
        llm=llm,
        k=k,
        persist_dir=persist_dir,
    )

    overall = compute_overall_score(retrieval, generation, k)

    return {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "k": k,
        "overall_score": overall,
        "retrieval": retrieval,
        "generation": generation,
    }


def print_full_report(report: dict) -> None:
    k = report["k"]
    print("\n" + "=" * 44)
    print("      RAG EVALUATION REPORT")
    print("=" * 44)
    print(f"  Date:         {report['date']}")
    print(f"  Test cases:   {report['retrieval']['total']}")
    print(f"  Retrieval k:  {k}")
    print()

    r = report["retrieval"]
    g = report["generation"]

    print("RETRIEVAL:")
    print(f"  Hit Rate:       {r['hit_rate']:.2f}")
    print(f"  MRR:            {r['mrr']:.2f}")
    print(f"  Precision@{k}:   {r.get(f'precision_at_{k}', 0.0):.2f}")
    print()
    print("GENERATION:")
    print(f"  Faithfulness:     {g['faithfulness']:.2f}")
    print(f"  Answer Relevancy: {g['answer_relevancy']:.2f}")
    print()
    print(f"  OVERALL SCORE:  {report['overall_score']:.2f} / 1.00")
    print("=" * 44)


def save_report(report: dict, output_path: str = "eval/last_report.json") -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[eval] Report saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full RAG evaluation")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--cases", default="eval/test_cases.json")
    parser.add_argument("--persist-dir", default=None)
    parser.add_argument(
        "--output", default="eval/last_report.json",
        help="Where to save the JSON report",
    )
    args = parser.parse_args()

    report = run_full_evaluation(
        test_cases_path=args.cases,
        k=args.k,
        persist_dir=args.persist_dir,
    )
    print_full_report(report)
    save_report(report, args.output)
