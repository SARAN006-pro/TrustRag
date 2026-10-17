"""Retrieval module for TrustRAG.

Retrieves top-k relevant chunks from ChromaDB for a given user query.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

# Use local cached model directly without querying remote HF hub
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from sentence_transformers import SentenceTransformer

from trustrag.config import CONFIG
from trustrag.index import COLLECTION_NAME, get_chroma_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def get_embedding_model(model_name: str = CONFIG.embed_model) -> SentenceTransformer:
    """Get or cache the SentenceTransformer embedding model."""
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def retrieve(
    question: str,
    k: int = CONFIG.top_k,
    chroma_dir: Path = CONFIG.chroma_dir,
    embed_model_name: str = CONFIG.embed_model,
) -> list[dict[str, Any]]:
    """Retrieve top-k chunks from ChromaDB collection for the given question.

    Returns:
        List of dicts:
        [{
            "chunk_id": str,
            "doc": str,
            "page": int | None,
            "text": str,
            "char_start": int,
            "score": float  # Cosine similarity (1.0 - cosine_distance)
        }]
    """
    if not question or not question.strip():
        return []

    client = get_chroma_client(chroma_dir)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as exc:
        logger.error(f"Failed to access ChromaDB collection '{COLLECTION_NAME}': {exc}")
        return []

    model = get_embedding_model(embed_model_name)
    query_emb = model.encode(question, normalize_embeddings=True).tolist()

    total_chunks = collection.count()
    if total_chunks == 0:
        return []

    actual_k = min(k, total_chunks)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=actual_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[dict[str, Any]] = []
    if not results or not results["ids"] or not results["ids"][0]:
        return []

    ids = results["ids"][0]
    docs = results["documents"][0] if results.get("documents") else []
    metadatas = results["metadatas"][0] if results.get("metadatas") else []
    distances = results["distances"][0] if results.get("distances") else []

    for idx, chunk_id in enumerate(ids):
        doc_text = docs[idx] if idx < len(docs) else ""
        meta = metadatas[idx] if idx < len(metadatas) else {}
        dist = distances[idx] if idx < len(distances) else 1.0
        # Cosine distance in Chroma is 1 - cosine_similarity (range 0 to 2 for normalized embeddings)
        score = max(0.0, 1.0 - dist)

        page_val = meta.get("page")
        page = None if page_val is None or page_val == -1 else int(page_val)

        chunks.append({
            "chunk_id": chunk_id,
            "doc": str(meta.get("doc", "")),
            "page": page,
            "text": doc_text,
            "char_start": int(meta.get("char_start", 0)),
            "score": round(score, 4),
        })

    return chunks


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "How much is the benefit in PM-KISAN?"
    print(f"Query: {q}")
    for res in retrieve(q, k=3):
        print(f"[{res['chunk_id']}] (score: {res['score']}) {res['text'][:120]}...")
