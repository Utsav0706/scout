"""Ingest PDFs and run hybrid search over the current user's chunks."""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import Select, func, literal_column, select
from sqlalchemy.orm import Session

from scout.config import RagConfig
from scout.models import Chunk, Document
from scout.rag.chunk import chunk_pages
from scout.rag.fusion import reciprocal_rank_fusion
from scout.rag.models_ml import Embedder, Reranker
from scout.rag.pdf import extract_pages

log = logging.getLogger(__name__)


class DuplicateDocumentError(Exception):
    def __init__(self, existing_id: uuid.UUID) -> None:
        super().__init__(f"Document already uploaded: {existing_id}")
        self.existing_id = existing_id


@dataclass(frozen=True)
class SearchHit:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    page: int
    text: str
    score: float  # reranker score: higher = more relevant; only comparable within one query


def ingest_pdf(
    db: Session,
    owner_id: uuid.UUID,
    filename: str,
    data: bytes,
    embedder: Embedder,
    cfg: RagConfig,
) -> Document:
    sha256 = hashlib.sha256(data).hexdigest()
    existing = db.scalar(
        select(Document.id).where(Document.owner_id == owner_id, Document.sha256 == sha256)
    )
    if existing is not None:
        raise DuplicateDocumentError(existing)

    pages = extract_pages(data, max_pages=cfg.max_pages)
    pieces = chunk_pages(pages, cfg.chunk_chars, cfg.chunk_overlap_chars)
    vectors = embedder.embed_documents([p.text for p in pieces])

    doc = Document(owner_id=owner_id, filename=filename, sha256=sha256, pages=len(pages))
    doc.chunks = [
        Chunk(ord=p.ord, page=p.page, text=p.text, embedding=v)
        for p, v in zip(pieces, vectors, strict=True)
    ]
    db.add(doc)
    db.commit()
    db.refresh(doc)
    log.info(
        "document ingested",
        extra={"document_id": str(doc.id), "pages": doc.pages, "chunks": len(pieces)},
    )
    return doc


def _scoped(stmt: Select, owner_id: uuid.UUID, document_ids: list[uuid.UUID] | None) -> Select:
    # Every search is limited to the caller's own documents. This filter lives in one
    # place so no search path can forget it.
    stmt = stmt.join(Document, Chunk.document_id == Document.id).where(
        Document.owner_id == owner_id
    )
    if document_ids:
        stmt = stmt.where(Document.id.in_(document_ids))
    return stmt


def search(
    db: Session,
    owner_id: uuid.UUID,
    query: str,
    embedder: Embedder,
    reranker: Reranker,
    cfg: RagConfig,
    document_ids: list[uuid.UUID] | None = None,
    top_k: int | None = None,
) -> list[SearchHit]:
    """Hybrid search: vector + keyword candidates, fused with RRF, then reranked.

    Why both: vectors find paraphrases ("car" ~ "vehicle") but miss exact names,
    codes and numbers; keyword search is the opposite. Together they cover both.
    """
    if db.get_bind().dialect.name != "postgresql":
        raise NotImplementedError("Search needs Postgres with pgvector")

    top_k = top_k or cfg.top_k
    n = cfg.candidates

    qvec = embedder.embed_query(query)
    vector_ids = list(
        db.scalars(
            _scoped(select(Chunk.id), owner_id, document_ids)
            .order_by(Chunk.embedding.cosine_distance(qvec))
            .limit(n)
        )
    )

    # Literal 'english' (not a bound parameter) so the expression matches the GIN index.
    english = literal_column("'english'")
    tsv = func.to_tsvector(english, Chunk.text)
    tsq = func.websearch_to_tsquery(english, query)  # tolerant of user input, never errors
    keyword_ids = list(
        db.scalars(
            _scoped(select(Chunk.id), owner_id, document_ids)
            .where(tsv.op("@@")(tsq))
            .order_by(func.ts_rank_cd(tsv, tsq).desc())
            .limit(n)
        )
    )

    fused = [cid for cid, _ in reciprocal_rank_fusion([vector_ids, keyword_ids])][:n]
    if not fused:
        return []

    rows = db.execute(
        select(Chunk, Document.filename)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.id.in_(fused))
    ).all()
    scores = reranker.score(query, [chunk.text for chunk, _ in rows])
    ranked = sorted(zip(rows, scores, strict=True), key=lambda r: r[1], reverse=True)[:top_k]

    log.info(
        "rag search",
        extra={
            "vector_hits": len(vector_ids),
            "keyword_hits": len(keyword_ids),
            "returned": len(ranked),
        },
    )
    return [
        SearchHit(chunk.id, chunk.document_id, filename, chunk.page, chunk.text, score)
        for (chunk, filename), score in ranked
    ]
