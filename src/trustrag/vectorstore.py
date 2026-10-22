"""VectorStore and EmbeddingProvider interfaces with ChromaDB and SentenceTransformer implementations.

V2 features:
- Abstract VectorStore interface (add, query, delete_by_source, count)
- Abstract EmbeddingProvider interface (embed_texts, embed_query)
- ChromaVectorStore: requires workspace_id, isolates one collection per workspace
- SentenceTransformerEmbedding: local dense embedding provider
"""
from __future__ import annotations

import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import chromadb
from sentence_transformers import SentenceTransformer

from trustrag.config import CONFIG

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract interface for generating embeddings."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text passages."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query."""
        pass


class SentenceTransformerEmbedding(EmbeddingProvider):
    """Local SentenceTransformer embedding implementation."""

    def __init__(self, model_name: str = CONFIG.embed_model):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        embedding = self.model.encode(query, normalize_embeddings=True, show_progress_bar=False)
        return embedding.tolist()


class VectorStore(ABC):
    """Abstract interface for workspace-partitioned vector databases."""

    @abstractmethod
    def add(
        self,
        workspace_id: str,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> int:
        """Add chunks and embeddings to the specified workspace collection."""
        pass

    @abstractmethod
    def query(
        self,
        workspace_id: str,
        query_embedding: list[float],
        k: int,
    ) -> list[dict[str, Any]]:
        """Query top-k chunks from the specified workspace collection."""
        pass

    @abstractmethod
    def delete_by_source(self, workspace_id: str, source_doc: str) -> int:
        """Delete all chunks originating from source_doc in the specified workspace."""
        pass

    @abstractmethod
    def count(self, workspace_id: str) -> int:
        """Count total chunks stored in the specified workspace collection."""
        pass


def sanitize_collection_name(workspace_id: str) -> str:
    """Derive a safe ChromaDB collection name from workspace_id."""
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", workspace_id.strip())
    # Chroma requires: 3-63 chars, start and end with alnum
    name = f"ws_{clean}"[:60]
    while len(name) < 3:
        name += "_0"
    if not name[-1].isalnum():
        name = name[:-1] + "1"
    return name


class ChromaVectorStore(VectorStore):
    """ChromaDB implementation with strict per-workspace collection isolation."""

    def __init__(self, chroma_dir: Path = CONFIG.chroma_dir):
        self.chroma_dir = Path(chroma_dir)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client: chromadb.ClientAPI | None = None

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(self.chroma_dir))
        return self._client

    def _validate_workspace(self, workspace_id: str | None) -> str:
        if workspace_id is None or not str(workspace_id).strip():
            raise ValueError("workspace_id is required; there is no global default")
        return str(workspace_id).strip()

    def _get_collection(self, workspace_id: str, create: bool = True):
        ws_id = self._validate_workspace(workspace_id)
        coll_name = sanitize_collection_name(ws_id)
        if create:
            return self.client.get_or_create_collection(
                name=coll_name,
                metadata={"hnsw:space": "cosine", "workspace_id": ws_id},
            )
        else:
            return self.client.get_collection(coll_name)

    def add(
        self,
        workspace_id: str,
        chunks: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> int:
        ws_id = self._validate_workspace(workspace_id)
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError("Length of chunks and embeddings must match.")

        collection = self._get_collection(ws_id, create=True)

        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id") or f"{chunk.get('doc', 'doc')}#{idx}"
            ids.append(chunk_id)
            texts.append(chunk.get("text", ""))
            metadatas.append({
                "doc": str(chunk.get("doc", "")),
                "page": -1 if chunk.get("page") is None else int(chunk.get("page")),
                "char_start": int(chunk.get("char_start", 0)),
                "workspace_id": ws_id,
            })

        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(ids)

    def query(
        self,
        workspace_id: str,
        query_embedding: list[float],
        k: int,
    ) -> list[dict[str, Any]]:
        ws_id = self._validate_workspace(workspace_id)
        try:
            collection = self._get_collection(ws_id, create=False)
        except Exception:
            return []

        total = collection.count()
        if total == 0:
            return []

        actual_k = min(max(1, k), total)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        docs = results["documents"][0] if results.get("documents") else []
        metas = results["metadatas"][0] if results.get("metadatas") else []
        dists = results["distances"][0] if results.get("distances") else []

        output: list[dict[str, Any]] = []
        for i, chunk_id in enumerate(ids):
            text = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) else {}
            dist = dists[i] if i < len(dists) else 1.0
            score = max(0.0, 1.0 - dist)

            page_val = meta.get("page")
            page = None if page_val is None or page_val == -1 else int(page_val)

            output.append({
                "chunk_id": chunk_id,
                "doc": str(meta.get("doc", "")),
                "page": page,
                "text": text,
                "char_start": int(meta.get("char_start", 0)),
                "score": round(score, 4),
            })
        return output

    def delete_by_source(self, workspace_id: str, source_doc: str) -> int:
        ws_id = self._validate_workspace(workspace_id)
        try:
            collection = self._get_collection(ws_id, create=False)
        except Exception:
            return 0

        initial_count = collection.count()
        collection.delete(where={"doc": source_doc})
        new_count = collection.count()
        return max(0, initial_count - new_count)

    def count(self, workspace_id: str) -> int:
        ws_id = self._validate_workspace(workspace_id)
        try:
            collection = self._get_collection(ws_id, create=False)
            return collection.count()
        except Exception:
            return 0


# Default global instances
_EMBEDDING_PROVIDER: EmbeddingProvider | None = None
_VECTOR_STORE: VectorStore | None = None


def get_default_embedding_provider() -> EmbeddingProvider:
    global _EMBEDDING_PROVIDER
    if _EMBEDDING_PROVIDER is None:
        _EMBEDDING_PROVIDER = SentenceTransformerEmbedding()
    return _EMBEDDING_PROVIDER


def get_default_vector_store() -> VectorStore:
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        _VECTOR_STORE = ChromaVectorStore()
    return _VECTOR_STORE
