"""Embedding and reranking models behind small interfaces.

The rest of the code depends on the Embedder / Reranker protocols, not on
sentence-transformers. Tests plug in tiny fakes, so CI never downloads models,
and swapping to an API embedder later touches only this file.

Two stages, because they trade speed for accuracy differently:
- Bi-encoder (embedder): embeds query and chunks separately, so chunk vectors are
  computed once at upload and search is a fast index lookup. Less precise.
- Cross-encoder (reranker): reads query + chunk together and scores the pair. Much more
  precise but too slow to run on every chunk, so it only reorders the top candidates.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder, SentenceTransformer

# bge models are trained with this instruction on queries (not on documents).
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        self._lock = threading.Lock()

    def _get(self) -> SentenceTransformer:
        # Loaded on first use, not at startup: keeps API startup fast and
        # means processes that never search never pay the ~130 MB of RAM.
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self._model_name, device="cpu")
            return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._get().encode(texts, batch_size=16, normalize_embeddings=True)
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        prefix = _BGE_QUERY_PREFIX if "bge" in self._model_name.lower() else ""
        return self._get().encode(prefix + text, normalize_embeddings=True).tolist()


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: CrossEncoder | None = None
        self._lock = threading.Lock()

    def _get(self) -> CrossEncoder:
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self._model_name, device="cpu")
            return self._model

    def score(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self._get().predict([(query, t) for t in texts])]
