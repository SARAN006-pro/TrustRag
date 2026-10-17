"""Document ingestion and chunking module.

Reads documents (.pdf and .txt) from CONFIG.raw_dir, cleans whitespace,
chunks text respecting sentence boundaries, and writes chunks to CONFIG.chunks_path.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterator

from pypdf import PdfReader

from trustrag.config import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def clean_text(text: str) -> str:
    """Normalize whitespace and line breaks while preserving sentence punctuation."""
    # Replace carriage returns and excessive whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Replace tabs and multiple spaces with a single space
    text = re.sub(r"[ \t]+", " ", text)
    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences_with_spans(text: str) -> list[tuple[str, int, int]]:
    """Split text into sentences with start and end character positions."""
    spans: list[tuple[str, int, int]] = []
    last_end = 0
    for match in _SENT_SPLIT.finditer(text):
        start = match.start()
        sent = text[last_end:start].strip()
        if sent:
            # find actual start of stripped sentence
            actual_start = text.find(sent, last_end)
            spans.append((sent, actual_start, actual_start + len(sent)))
        last_end = match.end()
    trailing = text[last_end:].strip()
    if trailing:
        actual_start = text.find(trailing, last_end)
        spans.append((trailing, actual_start, actual_start + len(trailing)))
    return spans


def chunk_text(
    text: str,
    chunk_chars: int = CONFIG.chunk_chars,
    chunk_overlap: int = CONFIG.chunk_overlap,
) -> list[tuple[str, int]]:
    """Chunk text into segments of approximately chunk_chars with chunk_overlap,
    splitting on sentence boundaries where possible. Returns list of (chunk_text, char_start).
    """
    clean = clean_text(text)
    if not clean:
        return []

    sentence_spans = split_sentences_with_spans(clean)
    if not sentence_spans:
        return [(clean, 0)]

    chunks: list[tuple[str, int]] = []
    sent_idx = 0
    n_sents = len(sentence_spans)

    while sent_idx < n_sents:
        current_chunk_sents: list[str] = []
        chunk_char_start = sentence_spans[sent_idx][1]
        current_length = 0
        end_idx = sent_idx

        while end_idx < n_sents:
            sent_text = sentence_spans[end_idx][0]
            candidate_length = current_length + (1 if current_length > 0 else 0) + len(sent_text)

            # If the very first sentence is longer than chunk_chars, we must slice it
            if current_length == 0 and len(sent_text) > chunk_chars:
                # Slices for overly long sentences
                sub_start = 0
                while sub_start < len(sent_text):
                    sub_end = min(sub_start + chunk_chars, len(sent_text))
                    sub_text = sent_text[sub_start:sub_end].strip()
                    if sub_text:
                        chunks.append((sub_text, sentence_spans[end_idx][1] + sub_start))
                    if sub_end == len(sent_text):
                        break
                    sub_start = max(sub_start + 1, sub_end - chunk_overlap)
                end_idx += 1
                sent_idx = end_idx
                break

            if candidate_length <= chunk_chars or current_length == 0:
                current_chunk_sents.append(sent_text)
                current_length = candidate_length
                end_idx += 1
            else:
                break

        if current_chunk_sents:
            chunk_str = " ".join(current_chunk_sents)
            chunks.append((chunk_str, chunk_char_start))

            # Determine next start index using overlap
            if end_idx >= n_sents:
                break

            # Calculate how many sentences to rewind for overlap
            overlap_accum = 0
            rewind_idx = end_idx - 1
            while rewind_idx > sent_idx:
                overlap_accum += len(sentence_spans[rewind_idx][0]) + 1
                if overlap_accum >= chunk_overlap:
                    break
                rewind_idx -= 1

            if rewind_idx <= sent_idx:
                # Guarantee progress
                sent_idx = max(sent_idx + 1, end_idx)
            else:
                sent_idx = rewind_idx

    return chunks


def read_document(file_path: Path) -> Iterator[tuple[str | None, str]]:
    """Yields (page_number_or_None, text) for each page or file."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(file_path))
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            yield str(page_num), page_text
    elif suffix in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            yield None, f.read()
    else:
        logger.warning(f"Skipping unsupported file format: {file_path}")


def ingest_documents(raw_dir: Path = CONFIG.raw_dir, output_path: Path = CONFIG.chunks_path) -> list[dict[str, Any]]:
    """Process all raw documents in raw_dir and write chunks.jsonl."""
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks: list[dict[str, Any]] = []
    files = sorted([p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in (".pdf", ".txt", ".md")])

    for file_path in files:
        doc_name = file_path.name
        chunk_idx = 0
        for page_num, raw_content in read_document(file_path):
            page_chunks = chunk_text(raw_content)
            for text, char_start in page_chunks:
                page_int = int(page_num) if page_num is not None else None
                chunk_record = {
                    "chunk_id": f"{doc_name}#{chunk_idx}",
                    "doc": doc_name,
                    "page": page_int,
                    "text": text,
                    "char_start": char_start,
                }
                chunks.append(chunk_record)
                chunk_idx += 1

    with open(output_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    logger.info(f"Ingested {len(files)} files into {len(chunks)} chunks -> {output_path}")
    return chunks


if __name__ == "__main__":
    ingest_documents()
