"""Index chunks into workspace-partitioned ChromaDB vector store.

V2 features:
- workspace_id is strictly required (no global default).
- Uses VectorStore and EmbeddingProvider interfaces.
- Non-destructive migration of v1 data to 'default' workspace.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import chromadb

from trustrag.config import CONFIG
from trustrag.db import init_db
from trustrag.vectorstore import (
    EmbeddingProvider,
    VectorStore,
    get_default_embedding_provider,
    get_default_vector_store,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

COLLECTION_NAME = "chunks"


def get_chroma_client(chroma_dir: Path = CONFIG.chroma_dir) -> chromadb.PersistentClient:
    """Return a persistent ChromaDB client."""
    chroma_dir = Path(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(chroma_dir))


def index_chunks(
    chunks: list[dict[str, Any]],
    workspace_id: str | None = None,
    vector_store: VectorStore | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> int:
    """Index a list of chunks into the required workspace."""
    if workspace_id is None or not str(workspace_id).strip():
        raise ValueError("workspace_id is required; there is no global default")

    ws_id = str(workspace_id).strip()
    if not chunks:
        return 0

    store = vector_store or get_default_vector_store()
    emb_provider = embedding_provider or get_default_embedding_provider()

    texts = [c["text"] for c in chunks]
    embeddings = emb_provider.embed_texts(texts)
    return store.add(workspace_id=ws_id, chunks=chunks, embeddings=embeddings)


def build_index(
    workspace_id: str | None = None,
    chunks_path: Path = CONFIG.chunks_path,
    vector_store: VectorStore | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> int:
    """Read chunks from chunks.jsonl and index into the specified workspace."""
    if workspace_id is None or not str(workspace_id).strip():
        raise ValueError("workspace_id is required; there is no global default")

    ws_id = str(workspace_id).strip()
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

    return index_chunks(
        chunks=chunks,
        workspace_id=ws_id,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )


def migrate_v1_to_default() -> int:
    """Migrate existing v1 chunks into the 'default' workspace without deleting v1 data."""
    init_db()
    store = get_default_vector_store()
    existing_count = store.count("default")
    if existing_count > 0:
        logger.info(f"Workspace 'default' already contains {existing_count} chunks. Migration skipped.")
        return existing_count

    logger.info("Migrating v1 chunks into workspace 'default'...")
    count = build_index(workspace_id="default")
    logger.info(f"Migration complete: {count} chunks indexed into 'default'.")
    return count


if __name__ == "__main__":
    count = migrate_v1_to_default()
    print(f"Default workspace chunk count: {count}")
