"""Split page text into overlapping chunks for embedding.

Choices:
- Size in characters, not tokens: no tokenizer dependency, and ~1500 chars (~350 tokens)
  stays well under the embedding model's 512-token limit.
- Chunks never cross a page boundary, so every chunk has one exact page for citations.
- Split on sentence ends where possible, so chunks don't start or stop mid-sentence.
- Overlap repeats the tail of the previous chunk, so a fact split across two chunks
  still appears whole in one of them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from scout.rag.pdf import PageText

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class TextChunk:
    ord: int
    page: int
    text: str


def _pieces(text: str, max_chars: int) -> list[str]:
    """Sentences, with any sentence longer than max_chars hard-split."""
    out: list[str] = []
    for sentence in _SENTENCE_END.split(text):
        while len(sentence) > max_chars:
            out.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if sentence:
            out.append(sentence)
    return out


def _tail(pieces: list[str], max_chars: int) -> list[str]:
    """The last whole pieces that fit in max_chars (for overlap)."""
    tail: list[str] = []
    size = 0
    for piece in reversed(pieces):
        if size + len(piece) + 1 > max_chars:
            break
        tail.insert(0, piece)
        size += len(piece) + 1
    return tail


def chunk_pages(pages: list[PageText], chunk_chars: int, overlap_chars: int) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for page in pages:
        text = _WHITESPACE.sub(" ", page.text).strip()
        if not text:
            continue
        current: list[str] = []
        for piece in _pieces(text, chunk_chars - overlap_chars):
            if current and len(" ".join([*current, piece])) > chunk_chars:
                chunks.append(TextChunk(len(chunks), page.page, " ".join(current)))
                current = _tail(current, overlap_chars)
            current.append(piece)
        if current:
            chunks.append(TextChunk(len(chunks), page.page, " ".join(current)))
    return chunks
