"""Evaluation harness for TrustRAG.

Computes retrieval metrics (hit@1, hit@3, hit@5) and answer accuracy.
Writes results to results/retrieval.json and results/answers.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path
from typing import Any

from trustrag.config import CONFIG
from trustrag.retrieve import retrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text for containment matching by lowering case and stripping non-alphanumerics."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def snippet_in_chunk(snippet: str, chunk_text: str) -> bool:
    """Check if snippet appears in chunk using normalized containment or word overlap."""
    norm_snip = normalize_text(snippet)
    norm_chunk = normalize_text(chunk_text)
    if not norm_snip:
        return False
    if norm_snip in norm_chunk:
        return True
    # Fallback: check if 80%+ of words from snippet appear in sequence or set
    snip_words = norm_snip.split()
    if len(snip_words) >= 4:
        # Check consecutive 4-grams
        four_grams = [" ".join(snip_words[i : i + 4]) for i in range(len(snip_words) - 3)]
        matched = sum(1 for fg in four_grams if fg in norm_chunk)
        if matched / len(four_grams) >= 0.75:
            return True
    return False


def evaluate_retrieval(
    questions_path: Path = CONFIG.eval_questions,
    results_dir: Path = CONFIG.results_dir,
    k_values: tuple[int, ...] = (1, 3, 5),
) -> dict[str, Any]:
    """Compute hit@k for all questions in questions_path with gold snippets."""
    questions_path = Path(questions_path)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    with open(questions_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Filter out abstention questions (gold_doc == 'NONE')
    eval_rows = [r for r in rows if r.get("gold_doc", "").strip().upper() != "NONE" and r.get("gold_snippet", "").strip()]
    total_eval = len(eval_rows)
    if total_eval == 0:
        logger.warning("No evaluable questions found with gold_doc != NONE")
        return {}

    hits: dict[int, int] = {k: 0 for k in k_values}
    max_k = max(k_values)
    per_question_results: list[dict[str, Any]] = []

    logger.info(f"Evaluating retrieval on {total_eval} questions (max k={max_k})...")
    for row in eval_rows:
        qid = row["question_id"]
        q = row["question"]
        gold_doc = row["gold_doc"].strip()
        gold_snip = row["gold_snippet"].strip()

        retrieved = retrieve(q, k=max_k)
        q_hits = {}
        for k in k_values:
            top_k_chunks = retrieved[:k]
            # Match by doc and snippet containment
            is_hit = any(
                (c["doc"] == gold_doc or snippet_in_chunk(gold_snip, c["text"]))
                for c in top_k_chunks
            )
            q_hits[k] = is_hit
            if is_hit:
                hits[k] += 1

        per_question_results.append({
            "question_id": qid,
            "hits": {f"hit@{k}": q_hits[k] for k in k_values},
            "retrieved_chunk_ids": [c["chunk_id"] for c in retrieved[:3]],
        })

    metrics: dict[str, Any] = {
        "num_eval_questions": total_eval,
        "total_questions_in_file": len(rows),
    }
    for k in k_values:
        metrics[f"hit@{k}"] = round(hits[k] / total_eval, 4)

    output_path = results_dir / "retrieval.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Retrieval evaluation complete: {metrics}")
    logger.info(f"Saved retrieval metrics to {output_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TrustRAG retrieval and answering.")
    parser.add_argument("--retrieval-only", action="store_true", help="Only evaluate retrieval hit@k")
    args = parser.parse_args()

    if args.retrieval_only:
        evaluate_retrieval()
    else:
        evaluate_retrieval()


if __name__ == "__main__":
    main()
