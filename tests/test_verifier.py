"""The verifier's contract. Any rewrite of scout/agent/verifier.py must keep these passing."""

import re

import pytest

from scout.agent.report import to_markdown
from scout.agent.state import ClaimVerdict, DraftClaim, DraftOut, DraftSection, VerifyOut
from scout.agent.verifier import (
    CheckedVerdict,
    check_verdict,
    combine,
    quote_in_source,
    tokens,
)
from scout.llm.client import LLMError
from tests.test_agent import CONFIG, default_script, result, run, verify_all

SOURCE = (
    "Inflation rose to 3.5% in the year to September 2026, the ONS said — its\nhighest since 2025."
)


# --- the quote check ---


def test_tokens_drop_punctuation_but_keep_numbers_whole() -> None:
    assert tokens("The ONS’s “CPI” rose 3.5% (1,000 items) — GPAI models , may") == [
        "the",
        "ons",
        "s",
        "cpi",
        "rose",
        "3.5",
        "1,000",
        "items",
        "gpai",
        "models",
        "may",
    ]


@pytest.mark.parametrize(
    "quote",
    [
        "Inflation rose to 3.5% in the year to September 2026",  # nbsp in source
        "the ONS said - its highest since 2025",  # em dash + line break in source
        "  INFLATION ROSE TO 3.5%  ",
        '"rose to 3.5% in the year to September 2026."',  # wrapping quotes/stop stripped
        "highest since 2025, the ONS said",  # (not in source order) -> see below
    ][:4]
    + [
        "the ONS said, its highest",  # source has an em dash where the quote has a comma
    ],
)
def test_real_quotes_are_found(quote: str) -> None:
    assert quote_in_source(quote, SOURCE)


@pytest.mark.parametrize(
    "quote",
    [
        "Inflation increased to 3.5% in the year to September 2026",  # paraphrase
        "Inflation rose to 3.6% in the year to September 2026",  # wrong number
        "rose to 3.5%",  # too short to prove anything
        "",
        "said the ONS, 3.5% in the year",  # real words, wrong order
        "Inflation rose to 35% in the year",  # 3.5 and 35 are different numbers
    ],
)
def test_invented_paraphrased_or_tiny_quotes_are_rejected(quote: str) -> None:
    assert not quote_in_source(quote, SOURCE)


def _v(verdict: str, *quotes: str) -> ClaimVerdict:
    return ClaimVerdict(claim_id=0, verdict=verdict, quotes=list(quotes), reason="r")


def test_supported_without_a_real_quote_is_downgraded() -> None:
    assert check_verdict(_v("supported", "made up evidence text here"), SOURCE) == (
        CheckedVerdict("unsupported", None)
    )


def test_contradicted_needs_a_real_quote_too() -> None:
    real = "Inflation rose to 3.5% in the year to September 2026"
    assert check_verdict(_v("contradicted", real), SOURCE).verdict == "contradicted"
    assert check_verdict(_v("contradicted", "not in the source at all"), SOURCE).verdict == (
        "unsupported"
    )


def test_unsupported_keeps_a_quote_only_if_real() -> None:
    assert check_verdict(_v("unsupported"), SOURCE) == CheckedVerdict("unsupported", None)
    assert check_verdict(_v("unsupported", "not in the source at all"), SOURCE).quote is None


def test_compound_claim_backed_by_two_real_quotes_is_supported() -> None:
    # Regression (live run 4): a claim joining facts from two sentences could only be
    # backed by stitching them into one quote, which failed the exact-match check.
    checked = check_verdict(
        _v(
            "supported",
            "Inflation rose to 3.5% in the year to September 2026",
            "its highest since 2025",
        ),
        SOURCE,
    )
    assert checked.verdict == "supported"
    assert (
        checked.quote
        == "Inflation rose to 3.5% in the year to September 2026 … its highest since 2025"
    )


def test_extraction_spacing_before_commas_does_not_fail_honest_quotes() -> None:
    # Regression (live run 4): page text "GPAI models , may result" vs quote
    # "GPAI models, may result" failed the old character-level check.
    page = "Other breaches, including of the obligations for GPAI models , may result in fines"
    assert quote_in_source("obligations for GPAI models, may result in fines", page)


def test_one_invented_quote_discards_the_verdict() -> None:
    checked = check_verdict(
        _v(
            "supported",
            "Inflation rose to 3.5% in the year to September 2026",
            "and wages grew by 9% over the same period",
        ),
        SOURCE,
    )
    assert checked == CheckedVerdict("unsupported", None)


def test_supported_with_no_quotes_is_downgraded() -> None:
    assert check_verdict(_v("supported"), SOURCE).verdict == "unsupported"


# --- combining verdicts from several sources ---


@pytest.mark.parametrize(
    ("verdicts", "expected"),
    [
        (["unsupported", "supported"], "supported"),  # any support wins
        (["contradicted", "supported"], "supported"),
        (["unsupported", "contradicted"], "contradicted"),
        (["unsupported", "unsupported"], "unsupported"),
    ],
)
def test_combine(verdicts: list[str], expected: str) -> None:
    assert combine([CheckedVerdict(v, None) for v in verdicts]).verdict == expected


def test_combine_nothing_checked() -> None:
    assert combine([]) is None


# --- the node, inside a full (offline) run ---


def script_with(verifier_fn):
    def script(node, kw):
        if node == "verifier":
            return verifier_fn(kw["messages"][0]["content"])
        return default_script(node, kw)

    return script


def _claims(report):
    return [c for s in report.sections for c in s.claims]


@pytest.mark.anyio
async def test_verifier_reads_the_source_text_not_the_notes(db_factory) -> None:
    _, llm, _ = await run(db_factory)
    content = next(kw["messages"][0]["content"] for n, kw in llm.calls if n == "verifier")
    assert "Official statistics page about" in content  # page text, as fetched
    assert "Fact from" not in content  # not the reader's notes


@pytest.mark.anyio
async def test_invented_quotes_make_claims_unsupported(db_factory) -> None:
    def fake(content):
        ids = [int(i) for i in re.findall(r"^\[(\d+)\]", content.split("<claims>")[1], re.M)]
        return result(
            "verifier",
            parsed=VerifyOut(
                verdicts=[
                    ClaimVerdict(
                        claim_id=i,
                        verdict="supported",
                        quotes=["a quote that is not there"],
                        reason="r",
                    )
                    for i in ids
                ]
            ),
        )

    report, _, _ = await run(db_factory, script=script_with(fake))
    assert {c.verdict for c in _claims(report)} == {"unsupported"}
    assert report.verification.support_rate == 0.0
    assert "[not supported by source]" in to_markdown(report)


@pytest.mark.anyio
async def test_contradicted_claims_are_flagged(db_factory) -> None:
    report, _, _ = await run(
        db_factory,
        script=script_with(
            lambda content: result("verifier", parsed=verify_all(content, "contradicted"))
        ),
    )
    assert {c.verdict for c in _claims(report)} == {"contradicted"}
    assert report.verification.contradicted == 2
    assert "[contradicted by source]" in to_markdown(report)


@pytest.mark.anyio
async def test_skipped_claims_count_as_unsupported(db_factory) -> None:
    report, _, _ = await run(
        db_factory,
        script=script_with(lambda content: result("verifier", parsed=VerifyOut(verdicts=[]))),
    )
    assert {c.verdict for c in _claims(report)} == {"unsupported"}


@pytest.mark.anyio
async def test_claim_with_two_sources_is_supported_if_either_supports(db_factory) -> None:
    def script(node, kw):
        content = kw["messages"][0]["content"]
        if node == "writer":
            ids = re.findall(r"^\[(S\d+)\]", content, re.M)
            return result(
                node,
                parsed=DraftOut(
                    summary="s",
                    limitations=[],
                    sections=[
                        DraftSection(heading="h", claims=[DraftClaim(text="Both.", source_ids=ids)])
                    ],
                ),
            )
        if node == "verifier":
            first = re.search(r'<source id="(S\d+)"', content).group(1)
            verdict = "supported" if first == "S2" else "unsupported"
            return result(node, parsed=verify_all(content, verdict))
        return default_script(node, kw)

    report, llm, _ = await run(db_factory, script=script)
    [claim] = _claims(report)
    assert len(claim.source_ids) == 2
    assert llm.nodes().count("verifier") == 2  # one call per cited source
    assert claim.verdict == "supported" and claim.quote


@pytest.mark.anyio
async def test_verifier_failure_leaves_claims_unverified(db_factory) -> None:
    report, _, _ = await run(
        db_factory, script=script_with(lambda content: LLMError("boom", cost_gbp=0.001))
    )
    assert {c.verdict for c in _claims(report)} == {"unverified"}
    v = report.verification
    assert (v.unverified, v.support_rate) == (2, None)  # no rate from zero checks
    assert "[not checked]" in to_markdown(report)


@pytest.mark.anyio
async def test_limit_hit_before_verifying_gives_partial_unchecked_report(db_factory) -> None:
    tight = CONFIG.model_copy(deep=True)
    tight.limits.max_steps = 10  # planner + 6 researcher + 2 reader + writer = 10
    report, llm, _ = await run(db_factory, config=tight)
    assert "verifier" not in llm.nodes()
    assert report.status == "partial"
    assert {c.verdict for c in _claims(report)} == {"unverified"}
