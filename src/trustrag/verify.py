"""Verification module for TrustRAG.

Loads the trained groundedness model, extracts features from (answer, chunks),
and predicts whether the answer is grounded along with a risk score.
Per SPEC section 7:
- Abstention answers (is_abstention = 1) are always grounded = True, risk_score = 0.0
- risk_score = predicted probability of class 1 (hallucinated)
- grounded = risk_score < CONFIG.hallucination_threshold
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from trustrag.config import CONFIG
from trustrag.features import extract_features
from trustrag.generate import ABSTENTION_ANSWER
from trustrag.retrieve import get_embedding_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, Any] = {}


def load_classifier(model_path: Path = CONFIG.model_path) -> Any:
    """Load or cache the trained groundedness classifier model."""
    path_str = str(model_path)
    if path_str not in _MODEL_CACHE:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Classifier model not found at {model_path}. Run 'python -m trustrag.train' first."
            )
        _MODEL_CACHE[path_str] = joblib.load(model_path)
    return _MODEL_CACHE[path_str]


def verify(
    answer: str,
    chunks: list[dict[str, Any]],
    model_path: Path = CONFIG.model_path,
    hallucination_threshold: float = CONFIG.hallucination_threshold,
) -> tuple[bool, float]:
    """Verify whether answer is grounded in retrieved chunks.

    Returns:
        (grounded: bool, risk_score: float)
        where risk_score is in [0.0, 1.0] representing probability of hallucination.
    """
    clean_answer = answer.strip()

    # Concatenate context chunks
    context = " ".join(c.get("text", "") for c in chunks) if chunks else ""

    # Check for immediate abstention
    if clean_answer.lower() == ABSTENTION_ANSWER.lower():
        return True, 0.0

    # Compute embedding similarity between answer and context
    embed_sim = 0.0
    if clean_answer and context:
        embed_model = get_embedding_model(CONFIG.embed_model)
        ans_emb = embed_model.encode(clean_answer, normalize_embeddings=True)
        ctx_emb = embed_model.encode(context, normalize_embeddings=True)
        embed_sim = float(np.dot(ans_emb, ctx_emb))

    # Extract features
    feat = extract_features(clean_answer, context, embed_sim=embed_sim)

    # Abstention rule (SPEC section 7)
    if feat.is_abstention == 1.0:
        return True, 0.0

    # Load trained model
    model = load_classifier(model_path)
    feature_vec = np.array([feat.to_vector()], dtype=np.float32)

    # Predict risk score (probability of class 1)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feature_vec)[0]
        # class 1 probability
        risk_score = float(probs[1]) if len(probs) > 1 else float(probs[0])
    elif hasattr(model, "decision_function"):
        score = model.decision_function(feature_vec)[0]
        risk_score = float(1.0 / (1.0 + np.exp(-score)))
    else:
        pred = model.predict(feature_vec)[0]
        risk_score = 1.0 if pred == 1 else 0.0

    risk_score = round(max(0.0, min(1.0, risk_score)), 4)
    grounded = bool(risk_score < hallucination_threshold)

    return grounded, risk_score


if __name__ == "__main__":
    from trustrag.retrieve import retrieve
    q = "What is the annual benefit amount in PM-KISAN?"
    test_chunks = retrieve(q, k=3)
    good_ans = "Under PM-KISAN, an income support of Rs 6,000 per year is provided in three instalments."
    bad_ans = "The scheme gives Rs 25,000 per month and requires an electric yacht."
    abst_ans = "Not found in the provided documents."

    print("Good answer verification:", verify(good_ans, test_chunks))
    print("Bad answer verification:", verify(bad_ans, test_chunks))
    print("Abstention verification:", verify(abst_ans, test_chunks))
