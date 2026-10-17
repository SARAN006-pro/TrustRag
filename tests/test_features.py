from trustrag.features import FEATURE_NAMES, extract_features, extract_numbers

CTX = ("Farmers owning up to 2 hectares of land are eligible. The scheme pays Rs 6,000 per year "
       "in three instalments. Government employees are not eligible.")


def test_vector_matches_feature_names():
    f = extract_features("Eligible farmers receive Rs 6,000 per year.", CTX, embed_sim=0.8)
    assert len(f.to_vector()) == len(FEATURE_NAMES)


def test_faithful_answer_scores_high_support():
    f = extract_features("The scheme pays Rs 6,000 per year in three instalments.", CTX)
    assert f.token_precision > 0.9
    assert f.unsupported_number_ratio == 0.0
    assert f.negation_mismatch == 0.0


def test_wrong_number_is_flagged():
    f = extract_features("The scheme pays Rs 12,000 per year in three instalments.", CTX)
    assert f.unsupported_number_ratio > 0.0


def test_flipped_eligibility_is_flagged():
    f = extract_features("Government employees are eligible.", CTX)
    assert f.negation_mismatch == 1.0


def test_unsupported_claim_lowers_min_support():
    f = extract_features("The scheme pays Rs 6,000 per year. Applicants must also own a tractor.", CTX)
    assert f.sentence_support_min < 0.7


def test_abstention_detected():
    f = extract_features("Not found in the provided documents.", CTX)
    assert f.is_abstention == 1.0


def test_empty_answer_does_not_crash():
    f = extract_features("", CTX)
    assert f.token_precision == 0.0 and f.sentence_support_min == 0.0


def test_number_extraction_normalizes_commas():
    assert extract_numbers("Rs 6,000 and 2.5 hectares") == {"6000", "2.5"}
