# Workflow and prompts

## Who does what
| Owner | Work |
|---|---|
| Claude (done) | AGENTS.md, SPEC.md, LABELING_GUIDE.md, config.py, features.py + tests, requirements |
| You (human) | Collect 10-15 scheme PDFs, write `questions.csv`, review and label rows, understand and demo everything |
| Antigravity | ingest, index, retrieve, generate, verify, train, evaluate, api, db, tests, README |
| Claude (checkpoints) | Review diffs and results, fix design problems, write the interview guide |

## One-time setup (you)
```
git init && git add . && git commit -m "chore: scaffold"
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
pytest                         # must show 8 passed before anything else
ollama pull llama3.2:3b        # or another model that fits 16 GB RAM
```
Put the documents in `data/raw/`. Use official PDFs or pages for 10-15 schemes (for example PM-KISAN, PMAY, Ayushman Bharat,
NSP scholarships, Tamil Nadu schemes). Write `data/eval/questions.csv` from `questions_template.csv` (50+ rows, 5+ with answer NONE).

---
## DAY 1 prompt: retrieval
```
Read AGENTS.md and docs/SPEC.md fully. Do not edit CLAUDE-OWNED files.
Task: implement ingest.py, index.py, retrieve.py and their tests (test_ingest.py, test_retrieve.py).
1. ingest.py: read PDFs/txt from CONFIG.raw_dir (pypdf), clean whitespace, chunk per SPEC section 6, write chunks.jsonl per SPEC section 5.
2. index.py: embed chunks with CONFIG.embed_model, store in ChromaDB at CONFIG.chroma_dir. Rebuilding must be idempotent.
3. retrieve.py: function retrieve(question, k) returning chunks with scores.
4. evaluate.py (retrieval part only): compute hit@1/3/5 from data/eval/questions.csv, write results/retrieval.json.
Run: pytest, python -m trustrag.ingest, python -m trustrag.index, python -m trustrag.evaluate --retrieval-only.
Report in the format required by AGENTS.md and paste the real hit@k numbers. Do not tune anything yet.
```
**Checkpoint 1 (come to Claude):** paste the report and `results/retrieval.json`. If hit@3 is below about 0.6, we debug chunking before moving on.

---
## DAY 2 prompt A: generation and candidate data
```
Read AGENTS.md and docs/SPEC.md. Implement generate.py (Ollama via httpx, prompt rules in SPEC section 6, temperature 0, timeout and clear errors)
and scripts/build_labeled_candidates.py. The script runs the pipeline on every question in questions.csv, then writes data/labels/labeled_candidates.csv
with the faithful answer (source=llm_faithful, label empty) and the three corruption variants from docs/LABELING_GUIDE.md
(number change, negation flip, unsupported sentence), plus abstention rows. Corruptions are done by deterministic code, not by an LLM.
Do not fill in labels for llm_faithful rows. Add tests for the corruption functions. Run it and report row counts per source.
```
**You:** open `labeled_candidates.csv`, set the labels for `llm_faithful` rows per the guide, review at least 50 rows, set `reviewed=1`, save as `data/labels/labeled.csv`.

## DAY 2 prompt B: classifier
```
Read AGENTS.md and docs/SPEC.md section 7. Implement verify.py and train.py.
train.py: load data/labels/labeled.csv, compute features with trustrag.features.extract_features (embed_sim from the embedding model),
compare the four models in SPEC section 7 using GroupShuffleSplit and GroupKFold by question_id, log each run to MLflow, save the best
model to CONFIG.model_path and write results/classifier.json and results/feature_importance.json.
verify.py: load the model, expose verify(answer, chunks) -> (grounded: bool, risk_score: float), with the abstention rule.
Add tests/test_verify.py. Run python -m trustrag.train and report the real metrics for all four models, including the baseline.
If the best model is barely above baseline, say so plainly and do not tune to hide it.
```
**Checkpoint 2 (come to Claude):** paste `classifier.json`. We check for leakage and read the feature importances together.

---
## DAY 3 prompt: API, experiments, docs
```
Read AGENTS.md and docs/SPEC.md sections 8, 9, 10. Implement db.py and api.py exactly per SPEC section 9 (including 503 and 422 behavior),
tests/test_api.py (use FastAPI TestClient with a stubbed generator for unit tests, and one clearly marked integration test that needs Ollama).
Finish evaluate.py (answer accuracy, abstention accuracy, latency, chunk-size x top_k experiments logged to MLflow).
Then write README.md: overview, architecture diagram, install, commands, real results copied from results/*.json, and a Limitations section.
Run: pytest, python -m trustrag.evaluate, uvicorn trustrag.api:app, then call /health and /ask with curl against the real model and paste the actual responses.
```
**Checkpoint 3 (come to Claude):** paste README, results and a screenshot or response from `/ask`. I write `docs/INTERVIEW_GUIDE.md` and run a mock interview on your actual numbers.

---
## Demo (rehearse twice)
1. `uvicorn trustrag.api:app` then `curl /health`
2. Ask a question that is in the documents: show the answer, sources and `grounded: true`
3. Ask one that is not: show the abstention
4. Show a hallucination caught: use an example from your test set where `risk_score` is high
5. Open MLflow (`mlflow ui`) and show the model comparison and the chunk-size experiment
6. Explain the three layers: deterministic code, LLM inference, trained classifier
