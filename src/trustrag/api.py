"""FastAPI service for TrustRAG with workspace isolation and privacy enforcement.

Endpoints per SPEC section 9 & extensions:
- GET /health -> {status, ollama_reachable, index_chunks, model_loaded}
- GET /providers -> List of supported LLM backends & config options
- POST /ask -> {answer, sources, grounded, risk_score, latency_ms}
- POST /ask/stream -> Server-Sent Events (SSE) streaming tokens & verification
- Honest errors: 503 for LLM unavailable, 403 for privacy violation, 422 for invalid question length.
- SQLite query logging with workspace_id tracking.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, AsyncGenerator

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import httpx
from fastapi import FastAPI, HTTPException, Request, Header, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from trustrag.config import CONFIG
from trustrag.db import init_db, log_query, get_workspace
from trustrag.generate import LLMBackendError, generate, format_context
from trustrag.vectorstore import get_default_vector_store
from trustrag.llm import (
    ABSTENTION_ANSWER,
    LLMAuthError,
    LLMError,
    LLMRateLimitError,
    PrivacyViolationError,
    check_workspace_privacy,
    get_available_providers_info,
    get_llm_provider,
)
from trustrag.retrieve import retrieve
from trustrag.verify import load_classifier, verify

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TrustRAG API",
    description="Grounded Q&A over Indian Government Schemes with Verification, Workspaces & Privacy Enforcement",
    version="2.0.0",
)

# Enable CORS for local Vite development frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="User question between 1 and 500 characters")
    workspace_id: str = Field(default="default", description="Workspace identifier (default: 'default')")
    provider: str = Field(default="ollama", description="LLM provider: 'ollama' or 'openai_compatible'")
    model: str | None = Field(default=None, description="Optional custom model name")
    api_key: str | None = Field(default=None, description="Optional ephemeral API key (never stored or logged)")
    base_url: str | None = Field(default=None, description="Optional custom endpoint base URL")

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
    score: float | None = None


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

    # 2. ChromaDB default workspace chunks count
    index_count = 0
    try:
        store = get_default_vector_store()
        index_count = store.count("default")
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


@app.get("/providers")
def list_providers() -> list[dict[str, Any]]:
    """Return available LLM backends and runtime metadata."""
    return get_available_providers_info()


@app.post("/ask", response_model=AskResponse)
def ask(
    req: AskRequest,
    x_llm_key: str | None = Header(default=None, alias="X-LLM-Key"),
    x_llm_provider: str | None = Header(default=None, alias="X-LLM-Provider"),
) -> AskResponse:
    """Process a user query in a workspace, retrieve relevant chunks, generate answers with citations,
    and verify groundedness with the trained classifier.
    """
    start_time = time.perf_counter()
    question = req.question
    workspace_id = req.workspace_id or "default"
    provider_name = req.provider or x_llm_provider or "ollama"
    effective_api_key = req.api_key or x_llm_key

    # Initialize provider
    try:
        llm = get_llm_provider(
            provider_name=provider_name,
            model=req.model,
            api_key=effective_api_key,
            base_url=req.base_url,
        )
        # Check workspace privacy rule
        check_workspace_privacy(workspace_id, llm)
    except PrivacyViolationError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except LLMAuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc

    # 1. Retrieve top-k chunks from the specific workspace
    try:
        chunks = retrieve(question, workspace_id=workspace_id, k=CONFIG.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    # 2. Generate answer
    if provider_name == "ollama" and req.model is None and req.base_url is None and effective_api_key is None:
        # Default backward-compatible path
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
    else:
        if not chunks:
            answer = ABSTENTION_ANSWER
            raw_sources = []
        else:
            try:
                context_str = format_context(chunks)
                messages = [
                    {"role": "system", "content": "You are TrustRAG, an assistant for Indian government welfare schemes. Answer directly and concisely based strictly on the context below. Cite sources as [chunk_id]."},
                    {"role": "user", "content": f"Context passages:\n{context_str}\n\nQuestion: {question.strip()}"},
                ]
                res = llm.generate(messages)
                answer = res.get("text", "")
                raw_sources = [
                    {
                        "chunk_id": c["chunk_id"],
                        "doc": c["doc"],
                        "page": c.get("page"),
                        "snippet": c.get("text", "")[:200],
                        "score": c.get("score"),
                    }
                    for c in chunks
                ]
            except LLMAuthError as exc:
                raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc
            except LLMRateLimitError as exc:
                raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)) from exc
            except LLMError as exc:
                raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
            except Exception as exc:
                raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"LLM failure: {exc}") from exc

    # 3. Groundedness verification
    grounded, risk_score = verify(answer, chunks)

    # 4. Compute latency
    latency_ms = int((time.perf_counter() - start_time) * 1000)

    # 5. Log to SQLite with workspace_id (never stores user API keys)
    log_query(
        question=question,
        answer=answer,
        grounded=grounded,
        risk_score=risk_score,
        latency_ms=latency_ms,
        workspace_id=workspace_id,
    )

    sources = [
        SourceItem(
            chunk_id=s["chunk_id"],
            doc=s["doc"],
            page=s.get("page"),
            snippet=s.get("snippet", ""),
            score=s.get("score"),
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


@app.post("/ask/stream")
async def ask_stream(
    req: AskRequest,
    raw_request: Request,
    x_llm_key: str | None = Header(default=None, alias="X-LLM-Key"),
    x_llm_provider: str | None = Header(default=None, alias="X-LLM-Provider"),
) -> StreamingResponse:
    """Stream answers as tokens are generated using Server-Sent Events (SSE)."""
    question = req.question
    workspace_id = req.workspace_id or "default"
    provider_name = req.provider or x_llm_provider or "ollama"
    effective_api_key = req.api_key or x_llm_key

    async def event_generator() -> AsyncGenerator[str, None]:
        start_time = time.perf_counter()
        accumulated_answer_parts: list[str] = []

        def sse_message(event: str, data: dict[str, Any]) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        try:
            # Check privacy rule
            llm = get_llm_provider(
                provider_name=provider_name,
                model=req.model,
                api_key=effective_api_key,
                base_url=req.base_url,
            )
            check_workspace_privacy(workspace_id, llm)

            # 1. Retrieval Stage
            yield sse_message("status", {"stage": "retrieving", "message": f"Searching workspace '{workspace_id}'..."})
            chunks = retrieve(question, workspace_id=workspace_id, k=CONFIG.top_k)

            sources_payload = [
                {
                    "chunk_id": c["chunk_id"],
                    "doc": c["doc"],
                    "page": c.get("page"),
                    "snippet": c.get("text", "")[:250],
                    "score": c.get("score"),
                }
                for c in chunks
            ]
            yield sse_message("sources", {"sources": sources_payload})

            # Check if context was found
            if not chunks:
                answer = ABSTENTION_ANSWER
                yield sse_message("token", {"token": answer})
                latency_ms = int((time.perf_counter() - start_time) * 1000)
                log_query(question=question, answer=answer, grounded=True, risk_score=0.0, latency_ms=latency_ms, workspace_id=workspace_id)
                yield sse_message(
                    "done",
                    {
                        "answer": answer,
                        "sources": [],
                        "grounded": True,
                        "risk_score": 0.0,
                        "latency_ms": latency_ms,
                    },
                )
                return

            # 2. Generation Stage
            yield sse_message("status", {"stage": "generating", "message": f"Generating response with {provider_name}..."})
            context_str = format_context(chunks)
            messages = [
                {"role": "system", "content": "You are TrustRAG, an assistant for Indian government welfare schemes. Answer directly and concisely based strictly on the context below. Cite sources as [chunk_id]."},
                {"role": "user", "content": f"Context passages:\n{context_str}\n\nQuestion: {question.strip()}"},
            ]

            async for token in llm.astream(messages):
                if await raw_request.is_disconnected():
                    logger.info("Client disconnected from /ask/stream, stopping generation.")
                    return
                accumulated_answer_parts.append(token)
                yield sse_message("token", {"token": token})

            final_answer = "".join(accumulated_answer_parts).strip()
            if not final_answer:
                final_answer = ABSTENTION_ANSWER

            # 3. Groundedness Verification Stage
            yield sse_message("status", {"stage": "verifying", "message": "Analyzing groundedness & citations..."})
            grounded, risk_score = verify(final_answer, chunks)

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Log to SQLite
            log_query(
                question=question,
                answer=final_answer,
                grounded=grounded,
                risk_score=risk_score,
                latency_ms=latency_ms,
                workspace_id=workspace_id,
            )

            # 4. Completion Event
            yield sse_message(
                "done",
                {
                    "answer": final_answer,
                    "sources": sources_payload,
                    "grounded": grounded,
                    "risk_score": risk_score,
                    "latency_ms": latency_ms,
                },
            )

        except PrivacyViolationError as exc:
            yield sse_message("error", {"error": str(exc), "code": 403, "provider": provider_name})
        except LLMAuthError as exc:
            yield sse_message("error", {"error": str(exc), "code": 401, "provider": provider_name})
        except LLMRateLimitError as exc:
            yield sse_message("error", {"error": str(exc), "code": 429, "provider": provider_name})
        except LLMError as exc:
            yield sse_message("error", {"error": exc.message, "code": exc.status_code, "provider": provider_name})
        except Exception as exc:
            logger.error(f"Stream generation error: {exc}")
            yield sse_message("error", {"error": f"Internal generation failure: {exc}", "code": 500, "provider": provider_name})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
