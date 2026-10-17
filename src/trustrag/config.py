"""Central configuration. CLAUDE-OWNED: change values via environment variables, not by editing logic."""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    # Paths
    raw_dir: Path = ROOT / "data" / "raw"
    chunks_path: Path = ROOT / "data" / "chunks.jsonl"
    chroma_dir: Path = ROOT / "chroma_db"
    eval_questions: Path = ROOT / "data" / "eval" / "questions.csv"
    labeled_path: Path = ROOT / "data" / "labels" / "labeled.csv"
    model_path: Path = ROOT / "models" / "groundedness.joblib"
    results_dir: Path = ROOT / "results"
    sqlite_path: Path = ROOT / "trustrag.sqlite"

    # Retrieval
    embed_model: str = os.getenv("TRUSTRAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chunk_chars: int = int(os.getenv("TRUSTRAG_CHUNK_CHARS", "800"))
    chunk_overlap: int = int(os.getenv("TRUSTRAG_CHUNK_OVERLAP", "100"))
    top_k: int = int(os.getenv("TRUSTRAG_TOP_K", "4"))

    # Generation (local Ollama only, no paid APIs). Pick a model that fits 16 GB RAM.
    ollama_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model: str = os.getenv("TRUSTRAG_LLM", "llama3.2:3b")
    llm_timeout_s: float = float(os.getenv("TRUSTRAG_LLM_TIMEOUT", "120"))

    # Verification
    hallucination_threshold: float = float(os.getenv("TRUSTRAG_HALLU_THRESHOLD", "0.5"))
    random_seed: int = 42


CONFIG = Config()
