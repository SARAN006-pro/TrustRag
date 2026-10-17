"""Index chunks into ChromaDB vector store.

Embeds chunks from CONFIG.chunks_path using CONFIG.embed_model
and persists the index in CONFIG.chroma_dir with cosine distance.
Rebuilding is idempotent.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from trustrag.config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COLLECTION_NAME = "chunks"


def get_chroma_client(chroma_dir: Path = CONFIG.chroma_dir) -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client."""
    chroma_dir = Path(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_dir))


def build_index(
    chunks_path: Path = CONFIG.chunks_path,
    chroma_dir: Path = CONFIG.chroma_dir,
    embed_model_name: str = CONFIG.embed_model,
) -> int:
    """Read chunks from chunks.jsonl, embed them, and index into ChromaDB idempotently."""
    chunks_path = Path(chunks_path)
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found at {chunks_path}. Run ingest first.")

    chunks: list[dict[str, Any]] = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    if not chunks:
        logger.warning(f"No chunks found in {chunks_path}")
        return 0

    logger.info(f"Loading embedding model: {embed_model_name}")
    model = SentenceTransformer(embed_model_name)

    logger.info(f"Connecting to ChromaDB at: {chroma_dir}")
    client = get_chroma_client(chroma_dir)

    # Idempotent rebuild: delete collection if exists
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        logger.info(f"Deleting existing collection '{COLLECTION_NAME}' for clean rebuild")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]
    metadatas = [
        {
            "chunk_id": c["chunk_id"],
            "doc": c["doc"],
            "page": c["page"] if c["page"] is not None else -1,
            "char_start": c["char_start"],
        }
        for c in chunks
    ]

    logger.info(f"Computing embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True).tolist()

    logger.info(f"Inserting {len(ids)} chunks into ChromaDB collection '{COLLECTION_NAME}'...")
    collection.add(
        ids=ids,
        documents=texts,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    count = collection.count()
    logger.info(f"Successfully indexed {count} chunks in ChromaDB")
    return count


if __name__ == "__main__":
    build_index()
