"""Unit tests for /ask/stream SSE streaming endpoint."""
import json
import pytest
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from trustrag.api import app

client = TestClient(app)


def test_get_providers_endpoint():
    resp = client.get("/providers")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    assert any(p["id"] == "ollama" for p in data)
    assert any(p["id"] == "openai" for p in data)


def test_ask_stream_empty_question_validation():
    resp = client.post("/ask/stream", json={"question": "   "})
    assert resp.status_code == 422


def test_ask_stream_success_flow():
    """Verify full SSE stream lifecycle: status -> sources -> token -> done."""
    mock_chunks = [
        {
            "chunk_id": "pm_kisan.txt#0",
            "doc": "pm_kisan.txt",
            "page": None,
            "text": "Eligible farmers receive Rs 6,000 per year.",
            "char_start": 0,
            "score": 0.88,
        }
    ]

    async def fake_tokens(*args, **kwargs):
        for token in ["Under ", "PM-KISAN, ", "farmers receive ", "Rs 6,000. ", "[pm_kisan.txt#0]"]:
            yield token

    mock_provider = AsyncMock()
    mock_provider.astream = fake_tokens

    with patch("trustrag.api.retrieve", return_value=mock_chunks), \
         patch("trustrag.api.get_llm_provider", return_value=mock_provider), \
         patch("trustrag.api.verify", return_value=(True, 0.05)):

        resp = client.post("/ask/stream", json={"question": "How much does PM-KISAN pay?"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        body = resp.text
        events = [line for line in body.split("\n") if line.startswith("event: ")]
        assert "event: status" in events
        assert "event: sources" in events
        assert "event: token" in events
        assert "event: done" in events


def test_ask_stream_missing_key_emits_error_event():
    """Verify that provider auth failure emits an SSE error event with status 401."""
    mock_chunks = [
        {
            "chunk_id": "pm_kisan.txt#0",
            "doc": "pm_kisan.txt",
            "page": None,
            "text": "Eligible farmers receive Rs 6,000 per year.",
            "char_start": 0,
            "score": 0.88,
        }
    ]

    with patch("trustrag.api.retrieve", return_value=mock_chunks):
        resp = client.post("/ask/stream", json={
            "question": "What is PM-KISAN?",
            "provider": "openai",
            "api_key": None,
        })
        assert resp.status_code == 200
        assert "event: error" in resp.text
        assert "401" in resp.text or "API key missing" in resp.text


def test_ask_stream_abstention_when_no_context():
    """Verify abstention when retrieval finds no chunks."""
    with patch("trustrag.api.retrieve", return_value=[]):
        resp = client.post("/ask/stream", json={"question": "Who won the 2026 cricket world cup?"})
        assert resp.status_code == 200
        assert "Not found in the provided documents." in resp.text
