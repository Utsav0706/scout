"""Database tables."""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DDL,
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    event,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from scout.db import Base

# Must match the embedding model in config.yaml (bge-small-en-v1.5 outputs 384 numbers).
# Changing models means re-embedding every chunk, so this is a code change, not config.
EMBEDDING_DIM = 384


class User(Base):
    __tablename__ = "users"

    # UUIDs, not 1, 2, 3...: ids show up in URLs and tokens, and sequential ids leak
    # how many users exist and invite guessing other users' ids.
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Document(Base):
    __tablename__ = "documents"
    # Same file uploaded twice by one user is a duplicate; two users may share a file.
    __table_args__ = (UniqueConstraint("owner_id", "sha256", name="uq_documents_owner_sha256"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64))
    pages: Mapped[int] = mapped_column(Integer, default=0)
    # processing -> ready | failed. Upload returns at once; a worker does the slow part.
    status: Mapped[str] = mapped_column(String(16), default="processing")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # The uploaded PDF, kept only until the worker has processed it (then set to None).
    # deferred: listing documents never loads megabytes of PDF.
    pdf_data: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True, deferred=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    ord: Mapped[int] = mapped_column(Integer)  # position within the document
    page: Mapped[int] = mapped_column(Integer)  # 1-based, so citations can say "p. 4"
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))

    document: Mapped[Document] = relationship(back_populates="chunks")


# Postgres-only search indexes. Added via DDL events so the same models still create
# plain tables on the SQLite test DB.
#  - HNSW: approximate nearest-neighbour index for cosine distance. Better recall/speed
#    trade-off than IVFFlat and needs no training step, so it works on an empty table.
#  - GIN on to_tsvector: makes full-text (keyword) search fast. An expression index
#    rather than a stored tsvector column keeps the table simpler.
for ddl in (
    "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw "
    "ON chunks USING hnsw (embedding vector_cosine_ops)",
    "CREATE INDEX IF NOT EXISTS ix_chunks_text_fts "
    "ON chunks USING gin (to_tsvector('english', text))",
):
    event.listen(Chunk.__table__, "after_create", DDL(ddl).execute_if(dialect="postgresql"))


class ToolCall(Base):
    """Audit log: one row per tool call the agent makes, successful or not.

    job_id has no foreign key yet: the research jobs table arrives in M6.
    """

    __tablename__ = "tool_calls"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    tool: Mapped[str] = mapped_column(String(64))
    args: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16))  # "ok" | "error"
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_chars: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Note(Base):
    """A fact the agent saved while researching, tied to the source it came from."""

    __tablename__ = "notes"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    job_id: Mapped[uuid.UUID] = mapped_column(index=True)
    source_id: Mapped[str] = mapped_column(String(16))  # "S3": per-job id from the tools
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ResearchJob(Base):
    """One research question, run in the background by a worker."""

    __tablename__ = "research_jobs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    question: Mapped[str] = mapped_column(Text)
    use_web: Mapped[bool] = mapped_column(default=True)
    use_docs: Mapped[bool] = mapped_column(default=True)
    # queued -> running -> complete | partial | no_evidence | failed
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    cost_gbp: Mapped[float] = mapped_column(default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class JobEvent(Base):
    """A progress line for a job ("planner: 4 sub-questions"), shown live to the user."""

    __tablename__ = "job_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)  # order of arrival
    job_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("research_jobs.id", ondelete="CASCADE"), index=True
    )
    node: Mapped[str] = mapped_column(String(32))
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
