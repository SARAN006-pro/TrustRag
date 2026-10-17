# TrustRAG: rules for coding agents (Antigravity)

TrustRAG answers questions about Indian government scheme documents with a local LLM, cites its sources, and
checks its own answer with a trained groundedness classifier. Read `docs/SPEC.md` before writing code.
It is the source of truth for architecture, data formats, API and metrics.

## Ownership
These files are CLAUDE-OWNED. Do not edit them without asking the user first:
`docs/SPEC.md`, `docs/LABELING_GUIDE.md`, `src/trustrag/config.py`, `src/trustrag/features.py`, `tests/test_features.py`.
You may add new modules, tests and scripts. Import configuration from `trustrag.config.CONFIG`; do not hardcode paths or model names.

## Hard rules
1. **No invented numbers.** Every metric in the README or docs must come from a command you actually ran, saved to `results/*.json`. If something was not run, say "not run".
2. **Never auto-label ground truth with an LLM.** An LLM may draft answers. Labels follow `docs/LABELING_GUIDE.md`, and a human reviews them.
3. **Split data by `question_id`**, never by row (`GroupShuffleSplit` or `GroupKFold`). Row-level splits leak and inflate scores.
4. **Local only.** Ollama for generation, sentence-transformers for embeddings. No paid APIs, no secrets in code or git.
5. **Dependencies:** only those in `requirements.txt`. Ask before adding any.
6. **Errors must be honest.** If Ollama is down, the API returns HTTP 503 with a clear message. Never return a fake answer.
7. **Every module ships with a test or a runnable smoke command.** Do not weaken or delete a failing test to make it pass; fix the cause.
8. **Small commits**, message format `type: summary` (feat, fix, test, docs, chore).
9. If blocked (missing model, missing data), write it to `BLOCKERS.md`, continue with independent work, and report it.

## Report format after every task
- Files created or changed
- Exact commands run
- Actual output (pasted, not summarized)
- What is still not working
- Next action
