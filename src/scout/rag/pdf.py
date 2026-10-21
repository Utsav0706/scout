"""PDF text extraction, one entry per page."""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

from pypdf import PdfReader
from pypdf.errors import PdfReadError

log = logging.getLogger(__name__)


class InvalidPDFError(ValueError):
    pass


@dataclass(frozen=True)
class PageText:
    page: int  # 1-based
    text: str


def extract_pages(data: bytes, max_pages: int) -> list[PageText]:
    # Check the magic bytes ourselves: the client's content-type header can say anything.
    if not data.startswith(b"%PDF-"):
        raise InvalidPDFError("File is not a PDF")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise InvalidPDFError("Encrypted PDFs are not supported")
        if len(reader.pages) > max_pages:
            raise InvalidPDFError(f"PDF has {len(reader.pages)} pages; the limit is {max_pages}")
        pages = [PageText(i, page.extract_text() or "") for i, page in enumerate(reader.pages, 1)]
    except PdfReadError as exc:
        raise InvalidPDFError(f"Could not read PDF: {exc}") from exc

    if not any(p.text.strip() for p in pages):
        # Scanned PDFs are images with no text layer. OCR is out of scope for now.
        raise InvalidPDFError("PDF has no extractable text (scanned image?)")
    return pages
