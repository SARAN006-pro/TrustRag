"""Generation module using local Ollama.

Answers questions based strictly on retrieved chunks with citations.
Adheres to SPEC section 6 prompt rules:
- Answer only from the provided numbered context.
- Cite sources as [chunk_id].
- If context lacks the answer, reply exactly: 'Not found in the provided documents.'
- Temperature 0.
- Honest error reporting if Ollama is unreachable.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from trustrag.config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are TrustRAG, an assistant for Indian government welfare schemes.\n"
    "Answer the user's question strictly and only using the provided numbered context passages below.\n"
    "Every statement you make must be directly supported by the context.\n"
    "Cite source passages using their exact [chunk_id] tag (for example: [pm_kisan.txt#0]).\n"
    "If the context does not contain enough information to answer the question, reply with exactly:\n"
    "Not found in the provided documents.\n"
    "Do not extrapolate, assume, or invent facts or eligibility rules."
)

ABSTENTION_ANSWER = "Not found in the provided documents."


class LLMBackendError(Exception):
    """Raised when the Ollama LLM backend is unreachable or fails."""
    pass


def format_context(chunks: list[dict[str, Any]]) -> str:
    """Format chunks into numbered context blocks with chunk_id tags."""
    lines: list[str] = []
    for idx, c in enumerate(chunks, start=1):
        chunk_id = c.get("chunk_id", f"chunk_{idx}")
        doc = c.get("doc", "")
        page_info = f" (page {c['page']})" if c.get("page") is not None else ""
        text = c.get("text", "").strip()
        lines.append(f"[{chunk_id}] {doc}{page_info}:\n{text}")
    return "\n\n".join(lines)


def generate(
    question: str,
    chunks: list[dict[str, Any]],
    ollama_url: str = CONFIG.ollama_url,
    model_name: str = CONFIG.llm_model,
    timeout_s: float = CONFIG.llm_timeout_s,
) -> tuple[str, list[dict[str, Any]]]:
    """Generate an answer with citations for the question given context chunks.

    Returns:
        (answer_text, sources_list)
    """
    if not chunks:
        return ABSTENTION_ANSWER, []

    context_str = format_context(chunks)
    user_content = f"Context passages:\n{context_str}\n\nQuestion: {question.strip()}"

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "options": {
            "temperature": 0.0,
            "num_predict": 128,
        },
        "stream": False,
    }

    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(f"{ollama_url.rstrip('/')}/api/chat", json=payload)
            if resp.status_code != 200:
                raise LLMBackendError(f"Ollama returned HTTP {resp.status_code}: {resp.text}")
            data = resp.json()
            answer = data.get("message", {}).get("content", "").strip()
    except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
        logger.error(f"Cannot connect to Ollama at {ollama_url}: {exc}")
        raise LLMBackendError("LLM backend unavailable") from exc
    except httpx.ReadTimeout as exc:
        logger.error(f"Ollama inference timed out after {timeout_s}s: {exc}")
        raise LLMBackendError("LLM inference timed out") from exc
    except Exception as exc:
        logger.error(f"Ollama request error: {exc}")
        raise LLMBackendError(f"Ollama backend error: {exc}") from exc

    # Clean up sources
    sources = [
        {
            "chunk_id": c["chunk_id"],
            "doc": c["doc"],
            "page": c.get("page"),
            "snippet": c.get("text", "")[:200],
        }
        for c in chunks
    ]

    return answer, sources


if __name__ == "__main__":
    from trustrag.retrieve import retrieve
    q = "What is the annual benefit amount in PM-KISAN?"
    retrieved_chunks = retrieve(q, k=3)
    ans, srcs = generate(q, retrieved_chunks)
    print("Question:", q)
    print("Answer:\n", ans)
    print("Sources:", [s["chunk_id"] for s in srcs])
