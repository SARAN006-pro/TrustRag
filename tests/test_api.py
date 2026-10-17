"""Unit and integration tests for FastAPI service."""
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from trustrag.api import app
from trustrag.config import CONFIG
from trustrag.db import get_recent_logs
from trustrag.generate import LLMBackendError

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "ollama_reachable" in data
    assert "index_chunks" in data
    assert "model_loaded" in data
    assert data["index_chunks"] > 0
    assert data["model_loaded"] is True


def test_ask_valid_question_with_stubbed_generator():
    mock_answer = "Under PM-KISAN, eligible farmers receive Rs 6,000 per year."
    mock_sources = [
        {
            "chunk_id": "pm_kisan.txt#0",
            "doc": "pm_kisan.txt",
            "page": None,
            "snippet": "PM-KISAN income support of Rs 6,000 per year...",
        }
    ]

    with patch("trustrag.api.generate", return_value=(mock_answer, mock_sources)):
        response = client.post("/ask", json={"question": "How much does PM-KISAN pay?"})
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == mock_answer
        assert len(data["sources"]) > 0
        assert data["sources"][0]["chunk_id"] == "pm_kisan.txt#0"
        assert "grounded" in data
        assert "risk_score" in data
        assert "latency_ms" in data
        assert data["latency_ms"] >= 0


def test_ask_empty_question_returns_422():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422

    response_blank = client.post("/ask", json={"question": "    "})
    assert response_blank.status_code == 422


def test_ask_overlong_question_returns_422():
    long_question = "What is the scheme? " * 35  # > 500 chars
    assert len(long_question) > 500
    response = client.post("/ask", json={"question": long_question})
    assert response.status_code == 422


def test_ask_ollama_unavailable_returns_503():
    with patch("trustrag.api.generate", side_effect=LLMBackendError("Connection refused")):
        response = client.post("/ask", json={"question": "Who is eligible for PM-KISAN?"})
        assert response.status_code == 503
        data = response.json()
        assert data["detail"] == "LLM backend unavailable"


def test_query_is_logged_to_sqlite():
    initial_logs = get_recent_logs(limit=10)
    initial_count = len(initial_logs)

    mock_answer = "Eligible farmers receive Rs 6,000 annually."
    mock_sources = [{"chunk_id": "pm_kisan.txt#0", "doc": "pm_kisan.txt", "page": None, "snippet": "..."}]

    with patch("trustrag.api.generate", return_value=(mock_answer, mock_sources)):
        test_q = "Testing SQLite query logging function"
        resp = client.post("/ask", json={"question": test_q})
        assert resp.status_code == 200

    new_logs = get_recent_logs(limit=10)
    assert len(new_logs) >= initial_count + 1
    latest = new_logs[0]
    assert latest["question"] == test_q
    assert latest["answer"] == mock_answer
    assert "ts" in latest


@pytest.mark.integration
def test_ask_real_ollama_integration():
    """Integration test that requires running local Ollama."""
    health_resp = client.get("/health")
    if not health_resp.json().get("ollama_reachable"):
        pytest.skip("Ollama is not reachable, skipping live integration test.")

    response = client.post("/ask", json={"question": "What is the annual benefit in PM-KISAN?"})
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        data = response.json()
        assert len(data["answer"]) > 0
        assert len(data["sources"]) > 0
        assert isinstance(data["grounded"], bool)
        assert 0.0 <= data["risk_score"] <= 1.0
