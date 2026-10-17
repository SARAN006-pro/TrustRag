"""Groundedness features. CLAUDE-OWNED. Pure Python (no model downloads) so it is fast, testable and explainable.

Each feature answers one question: "is the answer supported by the retrieved context?"
The classifier (trained later with scikit-learn) learns how to weigh these signals.
`embed_sim` (cosine similarity of answer vs context embeddings) is computed elsewhere and passed in,
because it needs the embedding model. Training and inference MUST both pass it.
"""
import re
from dataclasses import dataclass, asdict

_TOKEN = re.compile(r"[a-z0-9]+")
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_NEGATIONS = {"not", "no", "never", "cannot", "without", "neither", "nor", "ineligible", "excluded"}
_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "of", "to", "in", "on", "for", "and", "or",
    "with", "by", "as", "at", "it", "this", "that", "these", "those", "from", "under", "per", "can", "will",
    "you", "your", "they", "their", "has", "have", "had", "do", "does", "if", "any", "all",
}
_ABSTAIN = ("not found in", "cannot find", "can't find", "no information", "not mentioned", "do not have enough")

FEATURE_NAMES = [
    "token_precision",          # share of answer content words that appear in the context
    "sentence_support_mean",    # average per-sentence support
    "sentence_support_min",     # weakest sentence (one unsupported claim is enough to be wrong)
    "numbers_in_answer",        # how many numbers the answer states
    "unsupported_number_ratio", # share of those numbers that never appear in the context
    "negation_mismatch",        # 1 if answer and its best-matching context sentence disagree on negation
    "length_ratio",             # answer length / context length (capped at 1)
    "is_abstention",            # 1 if the answer says the info was not found
    "embed_sim",                # cosine similarity answer vs context (supplied by caller)
]


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in _STOP]


def extract_numbers(text: str) -> set[str]:
    return {m.replace(",", "").rstrip(".") for m in _NUMBER.findall(text)}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]


def _support(tokens: list[str], context_set: set[str]) -> float:
    return sum(t in context_set for t in tokens) / len(tokens) if tokens else 0.0


def _has_negation(text: str) -> bool:
    return any(t in _NEGATIONS for t in tokenize(text))


@dataclass
class GroundednessFeatures:
    token_precision: float
    sentence_support_mean: float
    sentence_support_min: float
    numbers_in_answer: float
    unsupported_number_ratio: float
    negation_mismatch: float
    length_ratio: float
    is_abstention: float
    embed_sim: float

    def to_vector(self) -> list[float]:
        d = asdict(self)
        return [float(d[name]) for name in FEATURE_NAMES]

    def to_dict(self) -> dict:
        return asdict(self)


def extract_features(answer: str, context: str, embed_sim: float = 0.0) -> GroundednessFeatures:
    answer_l = answer.lower()
    abstain = float(any(p in answer_l for p in _ABSTAIN))

    ctx_tokens = set(content_tokens(context))
    ans_tokens = content_tokens(answer)
    token_precision = _support(ans_tokens, ctx_tokens)

    ans_sents = split_sentences(answer)
    ctx_sents = split_sentences(context)
    supports = [_support(content_tokens(s), ctx_tokens) for s in ans_sents] or [0.0]

    ans_nums = extract_numbers(answer)
    ctx_nums = extract_numbers(context)
    unsupported = len(ans_nums - ctx_nums) / len(ans_nums) if ans_nums else 0.0

    # Negation check: compare each answer sentence with its most similar context sentence.
    mismatch = 0.0
    for s in ans_sents:
        s_tokens = set(content_tokens(s))
        if not s_tokens:
            continue
        best, best_overlap = None, 0.0
        for c in ctx_sents:
            overlap = len(s_tokens & set(content_tokens(c))) / len(s_tokens)
            if overlap > best_overlap:
                best, best_overlap = c, overlap
        if best is not None and best_overlap >= 0.3 and _has_negation(s) != _has_negation(best):
            mismatch = 1.0
            break

    ctx_len = len(tokenize(context))
    length_ratio = min(1.0, len(tokenize(answer)) / ctx_len) if ctx_len else 0.0

    return GroundednessFeatures(
        token_precision=token_precision,
        sentence_support_mean=sum(supports) / len(supports),
        sentence_support_min=min(supports),
        numbers_in_answer=float(len(ans_nums)),
        unsupported_number_ratio=unsupported,
        negation_mismatch=mismatch,
        length_ratio=length_ratio,
        is_abstention=abstain,
        embed_sim=float(embed_sim),
    )
