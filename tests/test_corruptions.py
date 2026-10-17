"""Unit tests for candidate corruption functions."""
from scripts.build_labeled_candidates import corrupt_number, corrupt_negation, corrupt_unsupported


def test_corrupt_number_modifies_digits():
    ans = "The scheme pays Rs 6,000 per year in three instalments."
    corrupted = corrupt_number(ans)
    assert corrupted is not None
    assert corrupted != ans
    # Assert at least one number changed
    assert ("6,000" not in corrupted) or ("three" in corrupted)


def test_corrupt_number_returns_none_when_no_numbers():
    ans = "No numerical values in this sentence."
    assert corrupt_number(ans) is None


def test_corrupt_negation_flips_eligibility():
    ans = "Government employees are not eligible for this scheme."
    corrupted = corrupt_negation(ans)
    assert corrupted is not None
    assert "are eligible" in corrupted
    assert "are not eligible" not in corrupted


def test_corrupt_negation_flips_mandatory():
    ans = "Aadhaar authentication is mandatory for enrollment."
    corrupted = corrupt_negation(ans)
    assert corrupted is not None
    assert "is optional" in corrupted


def test_corrupt_unsupported_appends_unsupported_clause():
    ans = "The monthly pension is Rs 3,000."
    corrupted = corrupt_unsupported(ans, seed_idx=0)
    assert corrupted.startswith(ans)
    assert len(corrupted) > len(ans)
    assert "Applicants must also submit" in corrupted
