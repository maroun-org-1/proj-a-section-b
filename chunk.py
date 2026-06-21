"""Preprocessing and chunking of corpus pages into retrieval units."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

# Sliding sentence window. SIZE sentences per chunk, advancing STEP each time
# (so consecutive chunks overlap by SIZE - STEP). Sized to stay under MiniLM's
# 256-token limit so no chunk is silently truncated at embedding time.
# Finer chunks (3/2) isolate individual fact-sentences for the dense signal and
# beat 6/4 on the public set (0.4294 -> 0.4419), improving 4/5 CV folds.
CHUNK_SIZE = 3
CHUNK_STEP = 2


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """Split one corpus entry into overlapping sentence-window chunks."""
    page_id = int(record["page_id"])
    title = record.get("title", "")
    sentences = record.get("content", "").split(".")

    chunks: List[Chunk] = []
    chunk_id = 0
    for i in range(0, len(sentences), CHUNK_STEP):
        text = ". ".join(sentences[i:i + CHUNK_SIZE]).strip()
        if text:
            # Prepend the title so every chunk carries its page's context.
            chunks.append(Chunk(page_id=page_id, chunk_id=chunk_id,
                                text=f"{title}: {text}"))
            chunk_id += 1
    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
