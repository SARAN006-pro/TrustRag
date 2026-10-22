"""Retrieval module for TrustRAG with workspace isolation.

V2 features:
- workspace_id is strictly required (no global default).
- Queries the workspace-partitioned VectorStore.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from trustrag.config import CONFIG
from trustrag.vectorstore import (
    EmbeddingProvider,
    VectorStore,
    get_default_embedding_provider,
    get_default_vector_store,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def get_embedding_model(model_name: str = CONFIG.embed_model):
    """Backward compatibility helper returning SentenceTransformer model instance."""
    provider = get_default_embedding_provider()
    return getattr(provider, "model", None)


def retrieve(
    question: str,
    workspace_id: str | None = None,
    k: int = CONFIG.top_k,
    vector_store: VectorStore | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    chroma_dir: Path = CONFIG.chroma_dir,
    embed_model_name: str = CONFIG.embed_model,
) -> list[dict[str, Any]]:
    """Retrieve top-k chunks from ChromaDB for the given question in a required workspace.

    Args:
        question: Query text.
        workspace_id: Required workspace identifier (e.g. 'default', 'ws1').
        k: Number of chunks to retrieve.
        vector_store: Optional VectorStore implementation.
        embedding_provider: Optional EmbeddingProvider implementation.

    Returns:
        List of chunk dicts with chunk_id, doc, page, text, char_start, score.
    """
    if workspace_id is None or not str(workspace_id).strip():
        raise ValueError("workspace_id is required; there is no global default")

    ws_id = str(workspace_id).strip()
    if not question or not question.strip():
        return []

    store = vector_store or get_default_vector_store()
    emb_provider = embedding_provider or get_default_embedding_provider()

    query_emb = emb_provider.embed_query(question.strip())
    return store.query(workspace_id=ws_id, query_embedding=query_emb, k=k)


def delete_by_source(
    source_doc: str,
    workspace_id: str | None = None,
    vector_store: VectorStore | None = None,
) -> int:
    """Delete all chunks originating from source_doc in the specified workspace."""
    if workspace_id is None or not str(workspace_id).strip():
        raise ValueError("workspace_id is required; there is no global default")

    store = vector_store or get_default_vector_store()
    return store.delete_by_source(workspace_id=str(workspace_id).strip(), source_doc=source_doc)


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "How much is the benefit in PM-KISAN?"
    print(f"Querying 'default' workspace: {q}")
    for res in retrieve(q, workspace_id="default", k=3):
        print(f"[{res['chunk_id']}] (score: {res['score']}) {res['text'][:120]}...")
