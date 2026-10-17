"""FastAPI service for TrustRAG.

Endpoints per SPEC section 9:
- GET /health -> {status, ollama_reachable, index_chunks, model_loaded}
- POST /ask -> {answer, sources, grounded, risk_score, latency_ms}
- Honest errors: 503 for LLM unavailable, 422 for invalid question length.
- SQLite query logging for every request.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import httpx
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from trustrag.config import CONFIG
from trustrag.db import init_db, log_query
from trustrag.generate import LLMBackendError, generate
from trustrag.index import COLLECTION_NAME, get_chroma_client
from trustrag.retrieve import retrieve
from trustrag.verify import load_classifier, verify

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="TrustRAG API", description="Grounded Q&A over Indian Government Schemes with Verification", version="1.0.0")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="User question between 1 and 500 characters")

    @field_validator("question")
    @classmethod
    def validate_not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Question cannot be blank or whitespace-only")
        return stripped


class SourceItem(BaseModel):
    chunk_id: str
    doc: str
    page: int | None = None
    snippet: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    grounded: bool
    risk_score: float
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    index_chunks: int
    model_loaded: bool


@app.on_event("startup")
def on_startup() -> None:
    """Initialize SQLite database and verify components on startup."""
    init_db()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Check system health, Ollama availability, ChromaDB index status, and model readiness."""
    # 1. Ollama reachable
    ollama_ok = False
    try:
        with httpx.Client(timeout=2.0) as client:
            resp = client.get(f"{CONFIG.ollama_url.rstrip('/')}/api/tags")
            if resp.status_code == 200:
                ollama_ok = True
    except Exception:
        ollama_ok = False

    # 2. ChromaDB index chunks count
    index_count = 0
    try:
        client = get_chroma_client(CONFIG.chroma_dir)
        coll = client.get_collection(COLLECTION_NAME)
        index_count = coll.count()
    except Exception:
        index_count = 0

    # 3. Model loaded
    model_ok = False
    try:
        _ = load_classifier(CONFIG.model_path)
        model_ok = True
    except Exception:
        model_ok = False

    overall_status = "ok" if (ollama_ok and index_count > 0 and model_ok) else "degraded"

    return HealthResponse(
        status=overall_status,
        ollama_reachable=ollama_ok,
        index_chunks=index_count,
        model_loaded=model_ok,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Process a user query, retrieve relevant scheme chunks, generate answers with citations,
    and verify groundedness with the trained classifier.
    """
    start_time = time.perf_counter()
    question = req.question

    # 1. Retrieve top-k chunks
    chunks = retrieve(question, k=CONFIG.top_k)

    # 2. Generate answer with Ollama
    try:
        answer, raw_sources = generate(question, chunks)
    except LLMBackendError as exc:
        logger.error(f"LLM backend unavailable: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM backend unavailable",
        ) from exc
    except Exception as exc:
        logger.error(f"Unexpected generation failure: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM backend unavailable",
        ) from exc

    # 3. Groundedness verification
    grounded, risk_score = verify(answer, chunks)

    # 4. Compute latency
    latency_ms = int((time.perf_counter() - start_time) * 1000)

    # 5. Log to SQLite
    log_query(
        question=question,
        answer=answer,
        grounded=grounded,
        risk_score=risk_score,
        latency_ms=latency_ms,
    )

    sources = [
        SourceItem(
            chunk_id=s["chunk_id"],
            doc=s["doc"],
            page=s.get("page"),
            snippet=s.get("snippet", ""),
        )
        for s in raw_sources
    ]

    return AskResponse(
        answer=answer,
        sources=sources,
        grounded=grounded,
        risk_score=risk_score,
        latency_ms=latency_ms,
    )
