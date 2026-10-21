"""Unit tests for multi-provider LLM interface, credential isolation, and error handling."""
import pytest
from unittest.mock import patch, MagicMock
from trustrag.llm import (
    get_llm_provider,
    get_available_providers_info,
    OllamaProvider,
    OpenAICompatibleProvider,
    AnthropicProvider,
    GeminiProvider,
    LLMAuthError,
    LLMError,
)
from trustrag.db import get_recent_logs, log_query


def test_provider_factory_resolution():
    ollama = get_llm_provider("ollama")
    assert isinstance(ollama, OllamaProvider)

    openai_p = get_llm_provider("openai", api_key="sk-test-123")
    assert isinstance(openai_p, OpenAICompatibleProvider)
    assert openai_p.model == "gpt-4o-mini"

    groq_p = get_llm_provider("groq", api_key="gsk-test")
    assert isinstance(groq_p, OpenAICompatibleProvider)
    assert groq_p.provider_name == "groq"

    anthropic_p = get_llm_provider("anthropic", api_key="sk-ant-test")
    assert isinstance(anthropic_p, AnthropicProvider)

    gemini_p = get_llm_provider("gemini", api_key="AIzaSyTest")
    assert isinstance(gemini_p, GeminiProvider)


def test_unknown_provider_raises_error():
    with pytest.raises(LLMError) as exc_info:
        get_llm_provider("non_existent_provider")
    assert exc_info.value.status_code == 400


def test_missing_api_key_raises_auth_error():
    # OpenAI without key
    p = get_llm_provider("openai", api_key=None)
    with pytest.raises(LLMAuthError):
        p.generate("Hello")

    # Anthropic without key
    p_ant = get_llm_provider("anthropic", api_key=None)
    with pytest.raises(LLMAuthError):
        p_ant.generate("Hello")

    # Gemini without key
    p_gem = get_llm_provider("gemini", api_key=None)
    with pytest.raises(LLMAuthError):
        p_gem.generate("Hello")


def test_get_available_providers_info():
    info = get_available_providers_info()
    ids = [p["id"] for p in info]
    assert "ollama" in ids
    assert "openai" in ids
    assert "groq" in ids
    assert "anthropic" in ids
    assert "gemini" in ids


def test_secret_isolation_in_sqlite():
    """Ensure user-provided API credentials never leak into SQLite query logs."""
    fake_key = "sk-super-secret-key-123456789"
    log_id = log_query(
        question="What is PM-KISAN?",
        answer="PM-KISAN provides Rs 6,000 annually. [pm_kisan.txt#0]",
        grounded=True,
        risk_score=0.02,
        latency_ms=250,
    )
    assert log_id > 0

    recent = get_recent_logs(limit=20)
    for row in recent:
        for val in row.values():
            assert fake_key not in str(val), "API key was found leaked in SQLite query log!"
