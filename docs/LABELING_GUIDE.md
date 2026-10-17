# Labeling guide (CLAUDE-OWNED)

Goal: about 300 rows of `(question, context, answer, label)` for the groundedness classifier.
Be honest in your README: part of this data is synthetic, built by deliberately corrupting correct answers.

## Label meaning
- **0 = grounded:** every claim in the answer (numbers, conditions, eligibility, documents) can be found in the context.
- **1 = hallucinated:** at least one claim is missing from the context or contradicts it.
- Abstention ("Not found in the provided documents.") is `label 0`, `source = abstention`, and counts as grounded, but only include it
  when the context truly lacks the answer.

## How to build rows (use `scripts/build_labeled_candidates.py`, then review by hand)
For about 100 questions, create from each one:
1. `llm_faithful`: the pipeline's real answer. Read it against the context and set label 0 only if every claim is supported. If it contains an unsupported claim, set label 1 (a natural hallucination is the most valuable row).
2. `corrupt_number`: copy the faithful answer and change one number or amount. Label 1.
3. `corrupt_negation`: flip an eligibility statement (eligible to not eligible, or the reverse). Label 1.
4. `corrupt_unsupported`: add one plausible-sounding sentence that is not in the context. Label 1.
Add about 20 `abstention` rows.

## Borderline cases
- A correct answer that rewords the context: 0.
- A correct but incomplete answer: 0 (incomplete is not hallucinated).
- A true real-world fact that is NOT in the context: 1 (the system must stay grounded in its documents).
- Rounding ("about 6000" vs "6,000"): 0. A different value ("12,000"): 1.

## Quality rules
- Review at least 50 rows personally and set `reviewed = 1`. Review all rows if time allows.
- Keep class balance within 40/60.
- Never let rows from the same `question_id` appear in both train and test.
- Record in the README: total rows, rows per `source`, rows reviewed.
