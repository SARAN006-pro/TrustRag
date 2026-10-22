"""Tests for V2 workspace isolation, vector store partitioning, privacy enforcement, and credential security.

Acceptance criteria:
(a) two workspaces with different documents queried with the same question never return each other's chunks, including after a delete;
(b) a retrieve call without workspace_id raises;
(c) the private-workspace rule;
(d) API keys never appear in logs or error messages.
"""
import os
import pytest
from unittest.mock import patch, MagicMock

from trustrag.db import (
    create_workspace,
    get_workspace,
    init_db,
    log_query,
    get_recent_logs,
)
from trustrag.index import index_chunks
from trustrag.retrieve import retrieve, delete_by_source
from trustrag.vectorstore import get_default_vector_store
from trustrag.llm import (
    OllamaProvider,
    OpenAICompatibleProvider,
    PrivacyViolationError,
    check_workspace_privacy,
    get_llm_provider,
    LLMAuthError,
    sanitize_secret,
)


@pytest.fixture(autouse=True)
def setup_db():
    init_db()


def test_workspaces_never_leak_chunks_and_delete_isolation():
    """(a) Two workspaces with different documents queried with the same question
    never return each other's chunks, including after a delete.
    """
    ws1 = "test_ws_alpha"
    ws2 = "test_ws_beta"

    create_workspace(ws1, "Alpha Scheme Org", privacy="shared")
    create_workspace(ws2, "Beta Scheme Org", privacy="shared")

    # Document A in Workspace Alpha
    doc_alpha_chunks = [
        {
            "chunk_id": "alpha_doc#0",
            "doc": "alpha_doc.txt",
            "page": 1,
            "text": "Alpha scheme grants Rs 50,000 subsidy exclusively for organic farmers.",
            "char_start": 0,
        }
    ]

    # Document B in Workspace Beta
    doc_beta_chunks = [
        {
            "chunk_id": "beta_doc#0",
            "doc": "beta_doc.txt",
            "page": 1,
            "text": "Beta scheme grants Rs 100,000 credit exclusively for dairy farmers.",
            "char_start": 0,
        }
    ]

    index_chunks(doc_alpha_chunks, workspace_id=ws1)
    index_chunks(doc_beta_chunks, workspace_id=ws2)

    query = "What grant is provided for farmers?"

    # Query Workspace Alpha
    res_alpha = retrieve(query, workspace_id=ws1, k=5)
    assert len(res_alpha) > 0
    alpha_chunk_ids = [c["chunk_id"] for c in res_alpha]
    assert "alpha_doc#0" in alpha_chunk_ids
    assert "beta_doc#0" not in alpha_chunk_ids, "Cross-workspace leak: Beta chunk found in Alpha!"

    # Query Workspace Beta
    res_beta = retrieve(query, workspace_id=ws2, k=5)
    assert len(res_beta) > 0
    beta_chunk_ids = [c["chunk_id"] for c in res_beta]
    assert "beta_doc#0" in beta_chunk_ids
    assert "alpha_doc#0" not in beta_chunk_ids, "Cross-workspace leak: Alpha chunk found in Beta!"

    # Delete Document A from Workspace Alpha
    deleted = delete_by_source(source_doc="alpha_doc.txt", workspace_id=ws1)
    assert deleted > 0

    # Verify Alpha is empty for that document
    after_del_alpha = retrieve(query, workspace_id=ws1, k=5)
    assert not any(c["doc"] == "alpha_doc.txt" for c in after_del_alpha)

    # Verify Beta is completely unaffected
    after_del_beta = retrieve(query, workspace_id=ws2, k=5)
    assert any(c["doc"] == "beta_doc.txt" for c in after_del_beta)


def test_retrieve_call_without_workspace_id_raises():
    """(b) A retrieve call without workspace_id raises."""
    with pytest.raises(ValueError) as exc:
        retrieve("What is the pension amount under PM-SYM?")
    assert "workspace_id is required" in str(exc.value)

    with pytest.raises(ValueError) as exc2:
        retrieve("What is the pension amount under PM-SYM?", workspace_id="")
    assert "workspace_id is required" in str(exc2.value)

    with pytest.raises(ValueError) as exc3:
        retrieve("What is the pension amount under PM-SYM?", workspace_id=None)
    assert "workspace_id is required" in str(exc3.value)


def test_private_workspace_refuses_hosted_providers():
    """(c) Privacy rule: a workspace with privacy='private' must refuse hosted providers
    unless ALLOW_HOSTED_FOR_PRIVATE=true is set; refuse with a clear error, not silently.
    """
    private_ws_id = "test_ws_confidential"
    create_workspace(private_ws_id, "Confidential Government Data", privacy="private")

    ollama_provider = get_llm_provider("ollama")
    hosted_provider = get_llm_provider("openai_compatible")

    # Local Ollama must be allowed for private workspace
    check_workspace_privacy(private_ws_id, ollama_provider)  # Must not raise

    # Hosted provider must be rejected by default
    with patch.dict(os.environ, {"ALLOW_HOSTED_FOR_PRIVATE": "false"}):
        with pytest.raises(PrivacyViolationError) as exc_info:
            check_workspace_privacy(private_ws_id, hosted_provider)
        assert "refuses hosted provider" in str(exc_info.value)
        assert private_ws_id in str(exc_info.value)

    # Hosted provider allowed when ALLOW_HOSTED_FOR_PRIVATE=true
    with patch.dict(os.environ, {"ALLOW_HOSTED_FOR_PRIVATE": "true"}):
        check_workspace_privacy(private_ws_id, hosted_provider)  # Must not raise


def test_api_keys_never_appear_in_logs_or_error_messages():
    """(d) API keys never appear in logs or error messages."""
    secret_key = "sk-live-super-secret-production-token-987654"

    # 1. Error sanitization check
    raw_error_message = f"Failed connecting to server with Authorization: Bearer {secret_key}"
    sanitized = sanitize_secret(raw_error_message)
    assert secret_key not in sanitized
    assert "[REDACTED]" in sanitized

    # 2. Provider exception message check
    provider = OpenAICompatibleProvider(api_key=secret_key, base_url="http://invalid.invalid:9999")
    with pytest.raises(Exception) as exc_info:
        provider.generate([{"role": "user", "content": "test"}])
    err_str = str(exc_info.value)
    assert secret_key not in err_str, f"Secret key leaked in exception message: {err_str}"

    # 3. SQLite query logs check
    log_id = log_query(
        question="Testing secret isolation",
        answer="Safe answer [doc#0]",
        grounded=True,
        risk_score=0.01,
        latency_ms=100,
        workspace_id="test_security_ws",
    )
    assert log_id > 0

    logs = get_recent_logs(limit=50)
    for row in logs:
        for key, val in row.items():
            assert secret_key not in str(val), f"Secret key leaked in SQLite column '{key}'!"
