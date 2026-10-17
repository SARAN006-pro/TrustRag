"""Script to generate candidate dataset for groundedness classifier training.

Runs the retrieval + generation pipeline for every question in questions.csv,
creates deterministic corruption variants per docs/LABELING_GUIDE.md:
- corrupt_number (label 1)
- corrupt_negation (label 1)
- corrupt_unsupported (label 1)
- abstention (label 0)
- llm_faithful (label left blank for human review)

Outputs data/labels/labeled_candidates.csv.
"""
from __future__ import annotations

import csv
import logging
import random
import re
from pathlib import Path
from typing import Any

from trustrag.config import CONFIG, ROOT
from trustrag.generate import ABSTENTION_ANSWER, generate
from trustrag.retrieve import retrieve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

UNSUPPORTED_CLAIMS = [
    "Applicants must also submit an affidavit from the district revenue officer within 14 days.",
    "Beneficiaries must maintain an active fixed deposit of Rs 25,000 in a scheduled commercial bank.",
    "The applicant must possess a valid commercial driving license to receive financial sanctions.",
    "All registered recipients are required to complete a mandatory 45-day technical skill program.",
    "The benefit is restricted strictly to families owning a registered electric tractor.",
    "Applicants must provide proof of annual domestic air travel clearance.",
]

_NUM_RE = re.compile(r"\b(\d[\d,]*(?:\.\d+)?)\b")


def corrupt_number(answer: str) -> str | None:
    """Find a number or amount in the answer and modify it to create a factual corruption."""
    matches = list(_NUM_RE.finditer(answer))
    if not matches:
        return None

    # Pick one match to modify
    m = random.choice(matches)
    orig_str = m.group(1)
    cleaned = orig_str.replace(",", "")
    try:
        if "." in cleaned:
            val = float(cleaned)
            new_val = round(val * 1.5, 2)
            repl = str(new_val)
        else:
            val = int(cleaned)
            if val == 0:
                new_val = 5
            elif val < 10:
                new_val = val + 5
            elif val <= 100:
                new_val = val * 2
            else:
                new_val = val * 2
            repl = f"{new_val:,}" if "," in orig_str else str(new_val)
    except ValueError:
        return None

    start, end = m.span(1)
    corrupted = answer[:start] + repl + answer[end:]
    return corrupted if corrupted != answer else None


def corrupt_negation(answer: str) -> str | None:
    """Flip an affirmative or negative assertion in the answer."""
    replacements = [
        (r"\bare not eligible\b", "are eligible"),
        (r"\bis not eligible\b", "is eligible"),
        (r"\bwere not eligible\b", "were eligible"),
        (r"\bare eligible\b", "are not eligible"),
        (r"\bis eligible\b", "is not eligible"),
        (r"\bis mandatory\b", "is optional"),
        (r"\bare mandatory\b", "are optional"),
        (r"\bis optional\b", "is mandatory"),
        (r"\bno restriction\b", "strict restrictions"),
        (r"\bwithout\b", "with mandatory"),
        (r"\bexcluded\b", "included"),
        (r"\bincluded\b", "excluded"),
        (r"\bcan be\b", "cannot be"),
        (r"\bcannot be\b", "can be"),
        (r"\bdoes not\b", "does"),
        (r"\bdo not\b", "do"),
    ]

    for pattern, repl in replacements:
        if re.search(pattern, answer, flags=re.IGNORECASE):
            corrupted = re.sub(pattern, repl, answer, count=1, flags=re.IGNORECASE)
            return corrupted

    # Fallback negation inversion
    if " not " in answer.lower():
        return re.sub(r"\bnot\b\s*", "", answer, count=1, flags=re.IGNORECASE)
    elif " is " in answer.lower():
        return re.sub(r"\bis\b", "is not", answer, count=1, flags=re.IGNORECASE)
    elif " are " in answer.lower():
        return re.sub(r"\bare\b", "are not", answer, count=1, flags=re.IGNORECASE)

    return None


def corrupt_unsupported(answer: str, seed_idx: int = 0) -> str:
    """Append a plausible but unsupported requirement to the answer."""
    claim = UNSUPPORTED_CLAIMS[seed_idx % len(UNSUPPORTED_CLAIMS)]
    answer = answer.strip()
    if not answer.endswith("."):
        answer += "."
    return f"{answer} {claim}"


def build_labeled_candidates(
    questions_csv: Path = CONFIG.eval_questions,
    output_csv: Path = ROOT / "data" / "labels" / "labeled_candidates.csv",
    use_llm: bool = True,
) -> list[dict[str, Any]]:
    """Build candidate rows for human labeling."""
    questions_csv = Path(questions_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(questions_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        questions = list(reader)

    rows: list[dict[str, Any]] = []
    row_id = 1

    logger.info(f"Processing {len(questions)} questions for candidate generation...")

    for q_idx, q_row in enumerate(questions):
        qid = q_row["question_id"]
        q_text = q_row["question"]
        gold_doc = q_row.get("gold_doc", "").strip()
        gold_ans = q_row.get("gold_answer", "").strip()

        # Retrieve context
        chunks = retrieve(q_text, k=CONFIG.top_k)
        context_str = " ".join(c["text"] for c in chunks) if chunks else ""

        # Determine faithful answer
        if gold_doc.upper() == "NONE":
            # Abstention question
            answer_text = ABSTENTION_ANSWER
            source = "abstention"
            rows.append({
                "row_id": f"row_{row_id:04d}",
                "question_id": qid,
                "question": q_text,
                "context": context_str,
                "answer": answer_text,
                "label": "0",  # Abstention is grounded (label 0) per guide
                "source": "abstention",
                "reviewed": "1",
            })
            row_id += 1
            continue

        if use_llm:
            try:
                llm_ans, _ = generate(q_text, chunks)
                answer_text = llm_ans
            except Exception as e:
                logger.warning(f"LLM generation failed for {qid} ({e}), falling back to gold answer")
                answer_text = gold_ans
        else:
            answer_text = gold_ans

        # 1. llm_faithful row (label left blank for human review per Day 2 prompt)
        rows.append({
            "row_id": f"row_{row_id:04d}",
            "question_id": qid,
            "question": q_text,
            "context": context_str,
            "answer": answer_text,
            "label": "",  # Empty for human review
            "source": "llm_faithful",
            "reviewed": "0",
        })
        row_id += 1

        # 2. corrupt_number
        c_num = corrupt_number(answer_text)
        if c_num:
            rows.append({
                "row_id": f"row_{row_id:04d}",
                "question_id": qid,
                "question": q_text,
                "context": context_str,
                "answer": c_num,
                "label": "1",
                "source": "corrupt_number",
                "reviewed": "1",
            })
            row_id += 1

        # 3. corrupt_negation
        c_neg = corrupt_negation(answer_text)
        if c_neg:
            rows.append({
                "row_id": f"row_{row_id:04d}",
                "question_id": qid,
                "question": q_text,
                "context": context_str,
                "answer": c_neg,
                "label": "1",
                "source": "corrupt_negation",
                "reviewed": "1",
            })
            row_id += 1

        # 4. corrupt_unsupported
        c_unsup = corrupt_unsupported(answer_text, seed_idx=q_idx)
        rows.append({
            "row_id": f"row_{row_id:04d}",
            "question_id": qid,
            "question": q_text,
            "context": context_str,
            "answer": c_unsup,
            "label": "1",
            "source": "corrupt_unsupported",
            "reviewed": "1",
        })
        row_id += 1

    fieldnames = ["row_id", "question_id", "question", "context", "answer", "label", "source", "reviewed"]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Report counts per source
    source_counts: dict[str, int] = {}
    for r in rows:
        source_counts[r["source"]] = source_counts.get(r["source"], 0) + 1

    logger.info(f"Built {len(rows)} candidate rows -> {output_csv}")
    logger.info(f"Row counts per source: {source_counts}")
    return rows


if __name__ == "__main__":
    build_labeled_candidates()
