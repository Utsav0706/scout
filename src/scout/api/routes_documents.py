"""Upload, list and search the current user's PDFs."""

from __future__ import annotations

from pathlib import PurePath

from fastapi import APIRouter, HTTPException, Request, UploadFile, status
from sqlalchemy import select

from scout.api.deps import CurrentUser, DbSession
from scout.jobs.queue import QueueUnavailableError
from scout.models import Document
from scout.rag import store
from scout.schemas import DocumentOut, SearchHitOut, SearchIn

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentOut, status_code=status.HTTP_202_ACCEPTED)
def upload(file: UploadFile, request: Request, user: CurrentUser, db: DbSession) -> Document:
    """Accept a PDF and queue it for processing. Returns at once with status
    "processing"; GET /documents shows when it is "ready" (or "failed", with the reason).
    A 300-page PDF can take minutes to embed on CPU: far too long for one request."""
    cfg = request.app.state.config.rag
    limit = cfg.max_upload_mb * 1024 * 1024
    # Read one byte past the limit: enough to know it's too big without loading a huge file.
    data = file.file.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE, f"File is larger than {cfg.max_upload_mb} MB"
        )
    # Keep only the base name: a filename like "../../etc/passwd" must not carry a path.
    filename = PurePath(file.filename or "upload.pdf").name[:255] or "upload.pdf"
    try:
        doc = store.create_upload(db, user.id, filename, data)
    except store.NotAPDFError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from None
    except store.DuplicateDocumentError as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT, {"message": "Already uploaded", "id": str(exc.existing_id)}
        ) from None

    try:
        request.app.state.job_queue.enqueue_ingest(doc.id)
    except QueueUnavailableError:
        doc.status, doc.error, doc.pdf_data = "failed", "The job queue is unavailable.", None
        db.commit()
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "The job queue is unavailable; try again soon."
        ) from None
    db.refresh(doc)  # a synchronous queue (tests) may already have processed it
    return doc


@router.get("", response_model=list[DocumentOut])
def list_documents(user: CurrentUser, db: DbSession) -> list[Document]:
    return list(
        db.scalars(
            select(Document)
            .where(Document.owner_id == user.id)
            .order_by(Document.created_at.desc())
        )
    )


@router.post("/search", response_model=list[SearchHitOut])
def search(
    body: SearchIn, request: Request, user: CurrentUser, db: DbSession
) -> list[store.SearchHit]:
    """For trying search by hand. In M3 the agent reaches the same function via the
    rag_search MCP tool."""
    state = request.app.state
    try:
        return store.search(
            db,
            user.id,
            body.query,
            state.embedder,
            state.reranker,
            state.config.rag,
            document_ids=body.document_ids,
            top_k=body.top_k,
        )
    except NotImplementedError as exc:
        raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, str(exc)) from None
