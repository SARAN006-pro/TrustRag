# TrustRAG Blockers and Environment Notes

## Resolved Blockers
1. **Ollama Registry Download Connectivity**:
   - `ollama pull llama3.2:3b` encountered DNS lookup failure (`lookup dd20bb891979d25aebc8bec07b2b3bbc.r2.cloudflarestorage.com: no such host`) on the machine's network.
   - **Resolution**: Utilized the pre-installed local Ollama model (`gpt-oss:20b`) via `TRUSTRAG_LLM=gpt-oss:20b` in `.env`.
   - Pipeline also gracefully supports offline deterministic fallbacks and mock generators in unit tests.

2. **Hugging Face Hub Online Lookup Latency**:
   - Initial `SentenceTransformer` loads attempted remote HEAD requests to Hugging Face Hub, causing 20-40s delays per script invocation.
   - **Resolution**: Explicitly configured `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` once `sentence-transformers/all-MiniLM-L6-v2` weights were downloaded to local disk cache. Retrieval latency dropped to ~0.07s (p50).

## Current Status
- All components operational: Ingestion, indexing (ChromaDB), retrieval, generation, ML verification, SQLite logging, and FastAPI endpoints.
- Total pytest suite passing: 31 tests.
