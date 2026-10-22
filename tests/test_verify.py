"""Tests for the groundedness verification module."""
import pytest
from trustrag.verify import verify
from trustrag.generate import ABSTENTION_ANSWER


from trustrag.retrieve import retrieve
from trustrag.config import CONFIG


@pytest.fixture
def sample_chunks():
    chunks = retrieve("How much financial benefit is given under PM-KISAN?", workspace_id="default", k=CONFIG.top_k)
    if not chunks:
        # Fallback if DB empty
        return [
            {
                "chunk_id": "pm_kisan.txt#0",
                "doc": "pm_kisan.txt",
                "page": 1,
                "text": "Pradhan Mantri Kisan Samman Nidhi (PM-KISAN) provides income support of Rs 6,000 per year in three equal four-monthly instalments of Rs 2,000 each. " * 5,
                "char_start": 0,
                "score": 0.85,
            }
        ]
    return chunks


def test_faithful_answer_is_grounded(sample_chunks):
    answer = "Under PM-KISAN, eligible farmers receive Rs 6,000 per year in three equal instalments of Rs 2,000."
    grounded, risk = verify(answer, sample_chunks)
    assert grounded is True
    assert risk < 0.5


def test_corrupted_number_is_flagged(sample_chunks):
    answer = "The scheme provides Rs 18,000 per year in three instalments of Rs 6,000."
    grounded, risk = verify(answer, sample_chunks)
    assert grounded is False
    assert risk >= 0.5


def test_unsupported_claim_is_flagged(sample_chunks):
    answer = "The scheme gives Rs 6,000 per year. Farmers must also purchase a tractor and install solar panels."
    grounded, risk = verify(answer, sample_chunks)
    assert grounded is False
    assert risk >= 0.5


def test_abstention_rule_always_grounded(sample_chunks):
    grounded, risk = verify(ABSTENTION_ANSWER, sample_chunks)
    assert grounded is True
    assert risk == 0.0

    # Also test case insensitivity
    grounded_lower, risk_lower = verify("not found in the provided documents.", sample_chunks)
    assert grounded_lower is True
    assert risk_lower == 0.0
