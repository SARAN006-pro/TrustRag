# TrustRAG V2 Architectural Audit & Readiness Report

**Date:** 2026-10-22  
**Auditor:** Antigravity AI  
**Scope:** Repository-wide audit of V1 architecture against V2 expansion requirements.

---

## 1. What Exists and Actually Works

TrustRAG is an operational, local-first RAG system with multi-provider LLM support, dense vector retrieval over Indian welfare schemes, trained ML hallucination detection, and a React + TypeScript frontend.

### Component Evidence
- **Document Ingestion & Sentence Chunking** ([src/trustrag/ingest.py](file:///c:/Users/saran/Desktop/trustrag/src/trustrag/ingest.py)):
  - Parses text files and extracts chunks bounded by `CONFIG.chunk_chars` (800) and `CONFIG.chunk_overlap` (100) on sentence boundaries.
  - Successfully produced 33 chunks across 12 schemes saved in [data/chunks.jsonl](file:///c:/Users/saran/Desktop/trustrag/data/chunks.jsonl).
- **Persistent Vector Index** ([src/trustrag/index.py](file:///c:/Users/saran/Desktop/trustrag/src/trustrag/index.py)):
  - Generates dense embeddings using `sentence-transformers/all-MiniLM-L6-v2` with `HF_HUB_OFFLINE=1`.
  - Persisted in ChromaDB at [chroma_db/](file:///c:/Users/saran/Desktop/trustrag/chroma_db) with collection `scheme_chunks` (33 active chunks).
- **Dense Vector Retrieval** ([src/trustrag/retrieve.py](file:///c:/Users/saran/Desktop/trustrag/src/trustrag/retrieve.py)):
  - Encodes queries and performs cosine distance queries against ChromaDB, returning top-k chunks with metadata and similarity scores in ~40ms (warm).
- **ML Groundedness Verification** ([src/trustrag/features.py](file:///c:/Users/saran/Desktop/trustrag/src/trustrag/features.py), [src/trustrag/verify.py](file:///c:/Users/saran/Desktop/trustrag/src/trustrag/verify.py)):
  - Extracts 11 deterministic features (lexical overlap, token containment, negation mismatch, number mismatch, citation format).
  - Evaluates risk with trained Gradient Boosting classifier [models/groundedness.joblib](file:///c:/Users/saran/Desktop/trustrag/models/groundedness.joblib).
- **Multi-Provider LLM & Streaming Engine** ([src/trustrag/llm.py](file:///c:/Users/saran/Desktop/trustrag/src/trustrag/llm.py)):
  - Unified interface `BaseLLMProvider` supporting Ollama, OpenAI, Groq, OpenRouter, Anthropic, Gemini, and custom OpenAI-compatible endpoints with Server-Sent Events (SSE) streaming and sync generation.
- **FastAPI Service & Query Auditing** ([src/trustrag/api.py](file:///c:/Users/saran/Desktop/trustrag/src/trustrag/api.py), [src/trustrag/db.py](file:///c:/Users/saran/Desktop/trustrag/src/trustrag/db.py)):
  - Endpoints: `GET /health`, `GET /providers`, `POST /ask`, `POST /ask/stream`.
  - WAL-mode SQLite database [trustrag.sqlite](file:///c:/Users/saran/Desktop/trustrag/trustrag.sqlite) logging every answered query.
- **Frontend Interface** ([frontend/](file:///c:/Users/saran/Desktop/trustrag/frontend)):
  - React 18, TypeScript, Vite, Tailwind CSS application with Scheme Navigator, dynamic chat stream, structured cards, evidence inspector, and settings modal with session-isolated credentials.

### Observed Test & API Outputs

#### Pytest Test Suite Execution (All 42 Tests Passed)
Command run:
```powershell
$env:PYTHONPATH="src;. "; python -m pytest
```
Output:
```text
============================= test session starts =============================
platform win32 -- Python 3.13.0, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\saran\Desktop\trustrag
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.6.2.post1, hypothesis-6.92.1, langsmith-0.9.5
collected 42 items

tests\test_api.py .......                                                [ 16%]
tests\test_corruptions.py .....                                          [ 28%]
tests\test_features.py ........                                          [ 47%]
tests\test_ingest.py .....                                               [ 59%]
tests\test_llm.py .....                                                  [ 71%]
tests\test_retrieve.py ...                                               [ 78%]
tests\test_stream.py .....                                               [ 90%]
tests\test_verify.py ....                                                [100%]

======================= 42 passed, 2 warnings in 39.69s =======================
```

#### Real API Health and Live Query Execution (`/health`, `/providers`, `/ask`)
Command run:
```powershell
$env:PYTHONPATH="src;. "; python -c "from fastapi.testclient import TestClient; from trustrag.api import app; client = TestClient(app); print(client.get('/health').json()); print(client.post('/ask', json={'question': 'What is the annual benefit in PM-KISAN?'}).json())"
```
Output:
```text
{'status': 'ok', 'ollama_reachable': True, 'index_chunks': 33, 'model_loaded': True}

{'answer': 'The annual benefit in PM-KISAN is Rs 6,000.',
 'sources': [{'chunk_id': 'pm_kisan.txt#0', 'doc': 'pm_kisan.txt', 'page': None, 'snippet': "Pradhan Mantri Kisan Samman Nidhi (PM-KISAN) 1. Scheme Overview: PM-KISAN is a Central Sector Scheme launched by the Government of India to provide income support to all landholding farmers' families ", 'score': None},
             {'chunk_id': 'ayushman_bharat.txt#0', 'doc': 'ayushman_bharat.txt', 'page': None, 'snippet': "Ayushman Bharat - Pradhan Mantri Jan Arogya Yojana (AB-PMJAY) 1. Overview and Coverage: Ayushman Bharat PM-JAY is the world's largest health assurance scheme fully financed by the Government. It aims ", 'score': None},
             {'chunk_id': 'pm_kisan.txt#3', 'doc': 'pm_kisan.txt', 'page': None, 'snippet': '- All superannuated or retired pensioners whose monthly pension is Rs 10,000 or more (excluding Multi Tasking Staff / Class IV / Group D employees). - All persons who paid Income Tax in the last asses', 'score': None},
             {'chunk_id': 'pm_svanidhi.txt#0', 'doc': 'pm_svanidhi.txt', 'page': None, 'snippet': "PM Street Vendor's AtmaNirbhar Nidhi (PM SVANidhi) 1. Background and Objective: PM SVANidhi is a Central Sector Scheme launched by the Ministry of Housing and Urban Affairs to facilitate working capit", 'score': None}],
 'grounded': True,
 'risk_score': 0.1682,
 'latency_ms': 5839}
```

---

## 2. What is a Stub, Untested, or Incomplete

1. **PDF Page Extraction Incomplete** ([src/trustrag/ingest.py:84-106](file:///c:/Users/saran/Desktop/trustrag/src/trustrag/ingest.py#L84-L106)):
   - While `ingest_pdf()` exists using `pypdf`, all raw data files currently in [data/raw/](file:///c:/Users/saran/Desktop/trustrag/data/raw) are `.txt` files. As a result, `page` metadata is universally `None`. No integration test verifies multi-page PDF coordinate extraction or page numbering in retrieval.
2. **Score Field Omission in Sync `/ask` Response** ([src/trustrag/api.py:168](file:///c:/Users/saran/Desktop/trustrag/src/trustrag/api.py#L168)):
   - In `generate()`, raw source dicts do not preserve the `score` computed by `retrieve()`. Consequently, `sources[i].score` is returned as `null`/`None` in `POST /ask`, whereas `POST /ask/stream` correctly includes `score`.
3. **No Automated Integration Tests for Cloud Providers** ([tests/test_stream.py](file:///c:/Users/saran/Desktop/trustrag/tests/test_stream.py)):
   - Cloud providers (OpenAI, Groq, Anthropic, Gemini, OpenRouter) are tested exclusively with mocks and stubbed responses. There are no opt-in live network integration tests for cloud endpoints.
4. **Streamlit File Missing** ([docs/SPEC.md:12](file:///c:/Users/saran/Desktop/trustrag/docs/SPEC.md#L12)):
   - The specification mentioned an optional Streamlit page (`requirements.txt` includes `streamlit`), but no Streamlit script was created (superseded by the React + Vite frontend).
5. **Small Evaluation Dataset** ([data/eval/questions.csv](file:///c:/Users/saran/Desktop/trustrag/data/eval/questions.csv)):
   - Contains 60 evaluation questions with only 6 abstention (`NONE`) questions. Edge cases such as ambiguous or contradictory scheme policies are not represented in the evaluation set.

---

## 3. Where V2 Requirements Conflict with Current Architecture

| V2 Requirement Area | Current V1 Implementation | Architectural Conflict & Root Cause |
| :--- | :--- | :--- |
| **Multi-Tenancy & Workspaces** | Single global singleton `CONFIG` in [src/trustrag/config.py:37](file:///c:/Users/saran/Desktop/trustrag/src/trustrag/config.py#L37). Paths like `chroma_dir` and `sqlite_path` are static. | V1 assumes a single machine and one shared filesystem. There is no concept of `workspace_id` or `tenant_id` to separate data, document uploads, or query histories. |
| **Shared Vector Database** | `COLLECTION_NAME = "scheme_chunks"` hardcoded in [src/trustrag/index.py:16](file:///c:/Users/saran/Desktop/trustrag/src/trustrag/index.py#L16). | Every document is stored in a single Chroma collection. Multiple workspaces or organizations would suffer data leakage without collection partitioning or `where={"workspace_id": ...}` metadata filters. |
| **LLM Inference Coupling** | Dual implementation: direct Ollama payload in [src/trustrag/generate.py:71-91](file:///c:/Users/saran/Desktop/trustrag/src/trustrag/generate.py#L71-L91) vs. `BaseLLMProvider` in [src/trustrag/llm.py](file:///c:/Users/saran/Desktop/trustrag/src/trustrag/llm.py). | `generate.py` duplicates Ollama HTTP handling instead of delegating to `llm.py`. Any parameter or prompt update made to `llm.py` fails to apply to `generate.py`. |
| **Authentication & Authorization** | No authentication middleware or security layer in [src/trustrag/api.py](file:///c:/Users/saran/Desktop/trustrag/src/trustrag/api.py). | All endpoints are public. There is no bearer token, API key validation, rate-limiting per user/IP, or RBAC mechanism. |
| **Answer Schema & Contract** | `AskResponse.answer` is an unstructured string ([src/trustrag/api.py:59](file:///c:/Users/saran/Desktop/trustrag/src/trustrag/api.py#L59)). | The frontend is forced to parse Markdown headers ("Eligibility", "Benefits", etc.) using regex heuristics in [frontend/src/components/StructuredAnswer.tsx:18](file:///c:/Users/saran/Desktop/trustrag/frontend/src/components/StructuredAnswer.tsx#L18). There is no guaranteed JSON schema. |
| **Citation Attribution** | LLM is prompted to include `[chunk_id]` tags in prose ([src/trustrag/generate.py:27](file:///c:/Users/saran/Desktop/trustrag/src/trustrag/generate.py#L27)). | Citations are purely best-effort text output from the model. Verification only checks whether any `[doc#n]` regex exists in the answer, not whether individual claims are entailed by the specific cited chunks. |
| **SQLite Auditing Concurrency** | Single local file `trustrag.sqlite` with WAL mode ([src/trustrag/db.py:20](file:///c:/Users/saran/Desktop/trustrag/src/trustrag/db.py#L20)). | SQLite does not support horizontal scaling across multi-instance API servers or cloud deployments. |

---

## 4. Concrete List of Smallest Refactors (Ordered by Dependency)

### Step 1: Unify LLM Dispatch into `generate.py`
- **Objective:** Eliminate code duplication between `generate.py` and `llm.py`.
- **Action:** Refactor [src/trustrag/generate.py](file:///c:/Users/saran/Desktop/trustrag/src/trustrag/generate.py) to delegate to `get_llm_provider("ollama")` while retaining the signature `generate(question, chunks, ...)` to preserve 100% backward compatibility with existing tests.
- **Risk:** Minimal.

### Step 2: Unify Retrieval Score Serialization in Sync `/ask`
- **Objective:** Fix `'score': None` inconsistency in sync API responses.
- **Action:** Pass chunk cosine similarity score through `generate()` into `raw_sources` so `POST /ask` populates `SourceItem.score`.
- **Risk:** None. Fixes existing API response gap.

### Step 3: Workspace-Scoped Vector Indexing & Retrieval
- **Objective:** Prevent cross-workspace data contamination.
- **Action:**
  - Add optional `collection_name: str` or `workspace_id: str` parameter to `retrieve()`, `index_chunks()`, and `get_chroma_client()`.
  - Add metadata tag `{"workspace_id": workspace_id}` to ChromaDB records.
  - Default to `"scheme_chunks"` when no workspace is specified to preserve V1 behavior.
- **Risk:** Low. Fully backward compatible when defaulting to `"scheme_chunks"`.

### Step 4: Structured Output Contract (`AnswerSchema`)
- **Objective:** Replace heuristic frontend regex parsing with a typed schema.
- **Action:**
  - Introduce `StructuredAnswerPayload` in `src/trustrag/api.py`:
    ```python
    class StructuredAnswerPayload(BaseModel):
        summary: str
        eligibility: list[str] = []
        benefits: list[str] = []
        required_documents: list[str] = []
        application_steps: list[str] = []
        missing_information: list[str] = []
        citations: list[dict[str, str]] = []
    ```
  - Support content negotiation: return structured JSON when `Accept: application/json` or requested via parameter, while keeping `answer: str` populated for V1 clients.
- **Risk:** Low. Backward compatible.

### Step 5: Claim-Level Citation Verification
- **Objective:** Ground specific statements rather than verifying document-level bag-of-words overlap.
- **Action:**
  - Enhance [src/trustrag/features.py](file:///c:/Users/saran/Desktop/trustrag/src/trustrag/features.py) to extract sentences containing `[chunk_id]` tags and compute per-claim similarity against the referenced passage.
  - Flag answers where cited passages do not contain key entities from the associated claim sentence.
- **Risk:** Medium. Must preserve existing features for `groundedness.joblib` classifier compatibility.

### Step 6: Authentication & Multi-Tenant Database Storage
- **Objective:** Add tenant isolation and security.
- **Action:**
  - Introduce optional `X-API-Key` / Bearer token authentication dependency in FastAPI.
  - Extend `query_log` table in [src/trustrag/db.py](file:///c:/Users/saran/Desktop/trustrag/src/trustrag/db.py) with `workspace_id` column (`DEFAULT 'default'`).
  - Add environment switch `TRUSTRAG_DB_URL` allowing PostgreSQL connection strings for production multi-instance deployments alongside local SQLite.
- **Risk:** Low. Default behavior remains single-user SQLite when no auth is configured.
