import pytest

from scout.rag.chunk import chunk_pages
from scout.rag.fusion import reciprocal_rank_fusion
from scout.rag.pdf import InvalidPDFError, PageText, extract_pages
from tests.fakes import make_pdf

# --- pdf ---


def test_extract_pages_keeps_page_numbers() -> None:
    pages = extract_pages(make_pdf(["Alpha page text.", "Beta page (with brackets)."]), 10)
    assert [p.page for p in pages] == [1, 2]
    assert "Alpha" in pages[0].text
    assert "Beta page (with brackets)" in pages[1].text


def test_non_pdf_rejected_by_magic_bytes() -> None:
    with pytest.raises(InvalidPDFError, match="not a PDF"):
        extract_pages(b"MZ\x90\x00 this is an exe", 10)


def test_corrupt_pdf_rejected() -> None:
    with pytest.raises(InvalidPDFError):
        extract_pages(b"%PDF-1.4\nthis is not really a pdf", 10)


def test_page_limit_enforced() -> None:
    with pytest.raises(InvalidPDFError, match="limit is 2"):
        extract_pages(make_pdf(["a.", "b.", "c."]), 2)


def test_pdf_without_text_rejected() -> None:
    with pytest.raises(InvalidPDFError, match="no extractable text"):
        extract_pages(make_pdf([""]), 10)


# --- chunking ---


def _sentences(n: int, page: int = 1) -> PageText:
    return PageText(page, " ".join(f"Sentence number {i} is here." for i in range(n)))


def test_short_page_is_one_chunk() -> None:
    chunks = chunk_pages([PageText(1, "  Just   one\nshort page. ")], 1500, 200)
    assert len(chunks) == 1
    assert chunks[0].text == "Just one short page."


def test_long_page_split_within_size_with_overlap() -> None:
    chunks = chunk_pages([_sentences(200)], 500, 100)
    assert len(chunks) > 1
    assert all(len(c.text) <= 500 for c in chunks)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        # The next chunk starts with the last sentence(s) of the previous one.
        assert prev.text.split(". ")[-1].rstrip(".") in nxt.text


def test_chunks_never_cross_pages_and_ords_are_sequential() -> None:
    chunks = chunk_pages([_sentences(60, page=1), PageText(2, ""), _sentences(60, page=3)], 400, 50)
    assert {c.page for c in chunks} == {1, 3}
    assert [c.ord for c in chunks] == list(range(len(chunks)))


def test_giant_sentence_is_hard_split() -> None:
    chunks = chunk_pages([PageText(1, "x" * 3000)], 1000, 100)
    assert all(len(c.text) <= 1000 for c in chunks)
    assert sum(len(c.text) for c in chunks) >= 3000


# --- fusion ---


def test_rrf_rewards_items_found_by_both_methods() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "d"]])
    ids = [item for item, _ in fused]
    assert ids[0] == "c"  # rank 3 + rank 1 beats a lone rank 1
    assert set(ids) == {"a", "b", "c", "d"}


def test_rrf_empty() -> None:
    assert reciprocal_rank_fusion([[], []]) == []
