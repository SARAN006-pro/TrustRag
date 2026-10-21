# TrustRAG Blockers and Environment Notes

## Resolved Blockers
1. **Ollama Registry Download Connectivity**:
   - `ollama pull llama3.2:3b` initially encountered a temporary DNS resolution issue with Cloudflare R2 on this machine.
   - **Resolution**: Verified DNS resolution with `Resolve-DnsName`, cleared incomplete layer via `ollama rm llama3.2:3b`, and completed `ollama pull llama3.2:3b`. `llama3.2:3b` is now successfully installed and running locally as `TRUSTRAG_LLM=llama3.2:3b`.

2. **Hugging Face Hub Online Lookup Latency**:
   - Initial `SentenceTransformer` loads attempted remote HEAD requests to Hugging Face Hub, causing 20-40s delays per script invocation.
   - **Resolution**: Explicitly configured `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` once `sentence-transformers/all-MiniLM-L6-v2` weights were downloaded to local disk cache. Retrieval latency dropped to ~0.07s (p50).

## Current Status
- All components operational: Ingestion, indexing (ChromaDB), retrieval, generation with `llama3.2:3b`, ML verification, SQLite logging, and FastAPI endpoints.
- Total pytest suite passing: 32 tests.
