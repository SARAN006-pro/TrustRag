"""Tests for retrieval module with workspace isolation."""
import pytest
from trustrag.retrieve import retrieve
from trustrag.config import CONFIG


def test_retrieve_returns_chunks_for_known_scheme():
    results = retrieve("How much financial benefit is given under PM-KISAN?", workspace_id="default", k=3)
    assert len(results) > 0
    top = results[0]
    assert "chunk_id" in top
    assert "doc" in top
    assert "text" in top
    assert "score" in top
    assert top["doc"] == "pm_kisan.txt"
    assert top["score"] > 0.4


def test_retrieve_empty_query():
    assert retrieve("", workspace_id="default") == []
    assert retrieve("   ", workspace_id="default") == []


def test_retrieve_respects_k():
    results = retrieve("scheme", workspace_id="default", k=2)
    assert len(results) <= 2


def test_retrieve_without_workspace_id_raises():
    with pytest.raises(ValueError) as exc:
        retrieve("What is PM-KISAN?")
    assert "workspace_id is required" in str(exc.value)
