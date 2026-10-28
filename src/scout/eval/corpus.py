"""The eval's private documents: made-up company files, so the web can't answer
questions about them and a correct answer proves document search worked.

They are kept as text (eval/corpus/*.txt, reviewable in git) and turned into PDFs at
run time, then uploaded through the normal pipeline (PDF text extraction, chunking,
embedding). A line holding only "---page---" starts a new PDF page.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

PAGE_BREAK = "---page---"
LINE_CHARS = 90
LINES_PER_PAGE = 50


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap(page: str) -> list[str]:
    lines: list[str] = []
    for para in page.strip().split("\n"):
        lines.extend(textwrap.wrap(para, LINE_CHARS) or [""])
    return lines


def to_pdf(pages: list[str]) -> bytes:
    """A minimal valid PDF (Helvetica, plain ASCII text), long lines wrapped."""
    page_lines: list[list[str]] = []
    for page in pages:
        lines = _wrap(page)
        # A page that overflows continues on the next physical page.
        page_lines.extend(
            lines[i : i + LINES_PER_PAGE] for i in range(0, max(len(lines), 1), LINES_PER_PAGE)
        )

    n = len(page_lines)
    page_ids = [4 + 2 * i for i in range(n)]
    objs: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] /Count {n} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, lines in enumerate(page_lines):
        body = " T* ".join(f"({_escape(line)}) Tj" for line in lines)
        stream = f"BT /F1 10 Tf 14 TL 60 740 Td {body} ET".encode("ascii")
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_ids[i] + 1} 0 R >>".encode()
        )
        objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for num, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def load_corpus(directory: Path) -> list[tuple[str, bytes]]:
    """(filename.pdf, pdf bytes) for every .txt file in the directory, sorted by name."""
    docs = []
    for path in sorted(directory.glob("*.txt")):
        text = path.read_text(encoding="ascii")  # Helvetica here only covers plain ASCII
        pages = [p for p in text.split(f"\n{PAGE_BREAK}\n") if p.strip()]
        docs.append((path.with_suffix(".pdf").name, to_pdf(pages)))
    return docs
