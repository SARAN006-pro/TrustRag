"""Tests for document ingestion and chunking."""
from pathlib import Path
import json
import pytest

from trustrag.ingest import clean_text, chunk_text, ingest_documents
from trustrag.config import CONFIG


def test_clean_text_normalizes_whitespace():
    raw = "Hello   world!\r\n\r\nThis  is a   test.\n\n\nNew paragraph."
    cleaned = clean_text(raw)
    assert "   " not in cleaned
    assert "\r" not in cleaned
    assert "Hello world!" in cleaned
    assert "\n\n\n" not in cleaned


def test_chunk_text_splits_sentences():
    text = (
        "Sentence one is here. Sentence two follows it. "
        "Sentence three provides more context. Sentence four concludes."
    )
    chunks = chunk_text(text, chunk_chars=50, chunk_overlap=10)
    assert len(chunks) >= 2
    for chunk_str, char_start in chunks:
        assert len(chunk_str) <= 70
        assert char_start >= 0


def test_chunk_text_handles_empty():
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_chunk_text_handles_oversized_sentence():
    long_sent = "Word " * 200
    chunks = chunk_text(long_sent, chunk_chars=100, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk_str, _ in chunks:
        assert len(chunk_str) <= 120


def test_ingest_documents_creates_valid_jsonl(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    doc_path = raw_dir / "scheme_test.txt"
    doc_path.write_text(
        "Scheme A provides Rs 5000 grant. Beneficiaries must be over 18 years old. "
        "Application is done online.",
        encoding="utf-8",
    )
    chunks_file = tmp_path / "chunks.jsonl"

    chunks = ingest_documents(raw_dir=raw_dir, output_path=chunks_file)
    assert len(chunks) >= 1
    assert chunks_file.exists()

    with open(chunks_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == len(chunks)
    first = lines[0]
    assert "chunk_id" in first
    assert "doc" in first
    assert "text" in first
    assert "char_start" in first
    assert first["doc"] == "scheme_test.txt"
