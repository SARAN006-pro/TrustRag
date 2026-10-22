# TrustRAG: specification (CLAUDE-OWNED)

## 1. Problem
People miss government benefits because eligibility rules are buried in long PDFs, and plain LLMs invent criteria
with confidence. TrustRAG answers only from the documents, cites the source passage, and flags answers
that are probably not supported by that passage.

## 2. Scope
In: PDF/text ingestion, chunking, embeddings, retrieval, local-LLM answering with citations, a trained groundedness
classifier, an evaluation harness, a FastAPI service, SQLite query log, MLflow experiment tracking.
Out (do not build): auth, multi-language, fine-tuning an LLM, deployment automation, agent frameworks, voice.
Optional only if everything else is done: a small Streamlit page.

## 3. Architecture
```mermaid
flowchart LR
  A[data/raw PDFs] --> B[ingest.py: extract + chunk]
  B --> C[(chunks.jsonl)]
  C --> D[index.py: embed + ChromaDB]
  Q[Question] --> E[retrieve.py: top-k chunks]
  D --> E
  E --> F[generate.py: Ollama answer + citations]
  F --> G[verify.py: features + classifier]
  G --> H[API response: answer, sources, grounded flag, risk score]
  H --> I[(SQLite log)]
  J[labeled.csv] --> K[train.py: scikit-learn + MLflow]
  K --> L[(groundedness.joblib)]
  L --> G
  M[questions.csv] --> N[evaluate.py: hit@k, accuracy, classifier metrics]
```
Deterministic code: ingest, chunking, features, API. LLM inference: only `generate.py`. Trained ML: the classifier in `train.py`.

## 4. Repository layout (create exactly this)
```
src/trustrag/  config.py* features.py* ingest.py index.py retrieve.py generate.py verify.py train.py evaluate.py api.py db.py
tests/         test_features.py* test_ingest.py test_retrieve.py test_verify.py test_api.py
scripts/       build_labeled_candidates.py
data/raw/  data/eval/questions.csv  data/labels/labeled.csv  results/  models/  docs/
(* = already exists, CLAUDE-OWNED)
```

## 5. Data contracts
**chunks.jsonl** (one JSON per line): `chunk_id` (`<doc>#<n>`), `doc`, `page` (int or null), `text`, `char_start`.
**questions.csv**: `question_id, question, gold_answer, gold_doc, gold_snippet`. Target at least 50 rows over at least 10 documents,
including 5 or more questions whose answer is NOT in any document (`gold_doc` = `NONE`).
**labeled.csv**: `row_id, question_id, question, context, answer, label, source, reviewed`
- `label`: 0 = grounded, 1 = hallucinated (see LABELING_GUIDE)
- `source`: `llm_faithful | corrupt_number | corrupt_negation | corrupt_unsupported | abstention`
- `reviewed`: 1 once a human checked the row. Target about 300 rows, balanced to within 40/60.

## 6. Retrieval and generation
- Chunk by characters (`CONFIG.chunk_chars`, `CONFIG.chunk_overlap`), splitting on sentence boundaries where possible.
- Embeddings: `CONFIG.embed_model`; vector store: ChromaDB persisted at `CONFIG.chroma_dir`; cosine distance.
- Prompt rules for the LLM: answer only from the numbered context; cite as `[chunk_id]`; if the context lacks the answer,
  reply exactly `Not found in the provided documents.`; temperature 0.
- Ollama call via `httpx` to `/api/chat`, with timeout `CONFIG.llm_timeout_s`.

## 7. Verification (the ML part)
- Features come from `features.extract_features(answer, context, embed_sim)` with `context` = the concatenated retrieved chunks.
- Models to compare, with identical splits: majority-class baseline, LogisticRegression (scaled), RandomForest, GradientBoosting.
- Split: `GroupShuffleSplit` by `question_id`, 80/20, seed `CONFIG.random_seed`; also report 5-fold `GroupKFold` mean and std.
- Report for the **hallucinated class (1)**: precision, recall, F1, plus ROC-AUC and a confusion matrix. Save to `results/classifier.json`.
- Explain the winner: save coefficients or feature importances to `results/feature_importance.json`.
- Risk score = predicted probability of class 1. `grounded = risk < CONFIG.hallucination_threshold`.
- Abstention answers (`is_abstention` = 1) are always `grounded = true` and are not scored as hallucinations.
- Log every training run to MLflow (params, metrics, model). Experiment name `trustrag`.

## 8. Evaluation harness (`evaluate.py`)
Writes `results/retrieval.json` and `results/answers.json`:
- **hit@k**: the chunk containing `gold_snippet` appears in the top-k (k = 1, 3, 5). Match by normalized text containment.
- **answer accuracy**: record `gold_answer` and the generated answer for each question and mark correct or incorrect with a simple rule
  (key numbers and key terms from the gold answer are present). Spot-check 20 by hand and report the agreement.
- **abstention accuracy** on the NONE questions.
- **latency**: p50 and p95 seconds for retrieval and for generation.
- **Experiments:** chunk size {500, 800, 1200} x top_k {3, 5}, each logged to MLflow with hit@k.

## 9. API (FastAPI)
- `GET /health` returns `{status, ollama_reachable, index_chunks, model_loaded}`.
- `POST /ask` with body `{question: str (1..500 chars)}` returns
  `{answer, sources: [{chunk_id, doc, page, snippet}], grounded: bool, risk_score: float, latency_ms: int}`.
- Ollama unreachable returns 503 `{detail: "LLM backend unavailable"}`. Empty or overlong question returns 422. Every request is logged to SQLite
  (`ts, question, answer, grounded, risk_score, latency_ms`).

## 10. Definition of done
- `pytest` passes; `python -m trustrag.evaluate` and `python -m trustrag.train` produce the files in `results/`.
- `uvicorn trustrag.api:app` answers a real question end-to-end using real documents and a real Ollama model.
- README lists exact commands, real metrics copied from `results/`, and a Limitations section.
- Known limits are documented honestly (small self-built dataset, synthetic corruptions, 3B model quality).

## 11. V2 Architecture: Workspaces, Interfaces, and Privacy (V2 Extension)

### 11.1 Workspaces and Multi-Tenant Isolation
- **Storage Isolation**: Every workspace maintains its own isolated ChromaDB collection named `ws_<workspace_id>` (sanitized).
- **No Global Default**: All retrieval, indexing, and deletion operations strictly REQUIRE an explicit `workspace_id`. Calling any retrieval or index operation without a `workspace_id` raises a `ValueError`.
- **Workspaces Registry**: SQLite `workspaces` table tracks configured workspaces:
  `id TEXT PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL, privacy TEXT NOT NULL CHECK(privacy IN ('private', 'shared'))`
- **V1 Backward Compatibility**: Existing V1 scheme documents are migrated non-destructively into workspace `'default'`.

### 11.2 VectorStore and EmbeddingProvider Interfaces
- `VectorStore` (ABC):
  - `add(workspace_id, chunks, embeddings) -> int`
  - `query(workspace_id, query_embedding, k) -> list[chunk_dict]`
  - `delete_by_source(workspace_id, source_doc) -> int`
  - `count(workspace_id) -> int`
  - Default implementation: `ChromaVectorStore` persisting to `CONFIG.chroma_dir`.
- `EmbeddingProvider` (ABC):
  - `embed_texts(texts: list[str]) -> list[list[float]]`
  - `embed_query(query: str) -> list[float]`
  - Default implementation: `SentenceTransformerEmbedding` using `CONFIG.embed_model`.

### 11.3 LLMProvider Interface
- `LLMProvider` (ABC):
  - `name: str`
  - `is_hosted: bool`
  - `generate(messages: list[dict], **opts) -> {"text": str, "usage": dict}`
  - `stream(messages: list[dict], **opts) -> Iterator[str]`
  - `astream(messages: list[dict], **opts) -> AsyncIterator[str]`
- Implementations:
  - `OllamaProvider`: Local, non-hosted (`is_hosted = False`). Default model from `CONFIG.llm_model`.
  - `OpenAICompatibleProvider`: Hosted (`is_hosted = True`). Base URL and API key loaded strictly from environment variables (`OPENAI_BASE_URL`, `OPENAI_API_KEY`).

### 11.4 Privacy Enforcement
- **Rule**: A workspace configured with `privacy="private"` strictly refuses hosted providers (`provider.is_hosted == True`) and raises `PrivacyViolationError` unless the environment variable `ALLOW_HOSTED_FOR_PRIVATE=true` is explicitly set.
- **Secret Isolation**: Provider credentials and API keys are never written to SQLite logs, disk files, error strings, or API responses. Any potential secret patterns in exceptions are sanitized and masked as `[REDACTED]`.
