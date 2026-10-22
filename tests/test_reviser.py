"""The reviser and the one-time research retry, offline (fake LLM + fake websites)."""

import re

import httpx
import pytest

from scout.agent.report import Claim, Section
from scout.agent.reviser import RETRY_HINT, gap_questions
from scout.agent.state import ClaimRevision, ClaimVerdict, Evidence, ReviseOut, VerifyOut
from scout.llm.client import LLMError
from tests.test_agent import CONFIG, SUBS, default_script, result, run, web

DRIVERS = "drove"  # the second sub-question's page path contains this word


def verify_by(rule):
    """Fake verifier: verdict per claim from rule(claim_text, source_title)."""

    def fake(content: str):
        title = re.search(r'title="([^"]*)"', content).group(1)
        source = re.search(r"<source[^>]*>\n(.*?)\n</source>", content, re.S).group(1)
        quote = source.splitlines()[-1]
        claims = re.findall(r"^\[(\d+)\] (.*)$", content.split("<claims>")[1], re.M)
        return result(
            "verifier",
            parsed=VerifyOut(
                verdicts=[
                    ClaimVerdict(
                        claim_id=int(i), verdict=rule(text, title), quotes=[quote], reason="r"
                    )
                    for i, text in claims
                ]
            ),
        )

    return fake


def fail_drivers(text: str, title: str) -> str:
    if text.startswith("Rewritten"):
        return "supported"
    return "unsupported" if DRIVERS in title else "supported"


def script(verifier=None, reviser=None):
    def s(node, kw):
        content = kw["messages"][0]["content"] if node in ("verifier", "reviser") else None
        if node == "verifier" and verifier:
            return verifier(content)
        if node == "reviser" and reviser:
            return reviser(content)
        return default_script(node, kw)

    return s


def rewrite_all(text: str):
    def fake(content: str):
        ids = [int(i) for i in re.findall(r'<failed_claim id="(\d+)"', content)]
        return result(
            "reviser",
            parsed=ReviseOut(
                summary="Revised summary.",
                revisions=[ClaimRevision(claim_id=i, action="rewrite", text=text) for i in ids],
            ),
        )

    return fake


def no_retry():
    cfg = CONFIG.model_copy(deep=True)
    cfg.agent.max_retry_questions = 0
    return cfg


def claims_of(report) -> list[Claim]:
    return [c for s in report.sections for c in s.claims]


# --- revising ---


@pytest.mark.anyio
async def test_everything_supported_means_no_reviser_call(db_factory) -> None:
    report, llm, _ = await run(db_factory)
    assert "reviser" not in llm.nodes()
    assert (report.verification.rewritten, report.verification.removed) == (0, 0)


@pytest.mark.anyio
async def test_failed_claim_is_rewritten_reverified_and_kept(db_factory) -> None:
    report, llm, _ = await run(
        db_factory,
        config=no_retry(),
        script=script(verify_by(fail_drivers), rewrite_all("Rewritten, narrower claim.")),
    )
    assert llm.nodes().count("reviser") == 1
    texts = [c.text for c in claims_of(report)]
    assert "Rewritten, narrower claim." in texts
    assert all(c.verdict == "supported" for c in claims_of(report))  # only supported survive
    v = report.verification
    assert (v.unsupported, v.rewritten, v.removed) == (1, 1, 0)
    assert v.support_rate == 0.5  # the first-check rate is kept: it measures the writer
    assert report.summary == "Revised summary."


@pytest.mark.anyio
async def test_rewrite_that_still_fails_is_removed(db_factory) -> None:
    def still_wrong(text, title):
        return "unsupported" if DRIVERS in title else "supported"

    report, llm, _ = await run(
        db_factory,
        config=no_retry(),
        script=script(verify_by(still_wrong), rewrite_all("Still a stretch.")),
    )
    assert llm.nodes().count("verifier") == 3  # 2 sources + 1 re-check of the rewrite
    assert len(claims_of(report)) == 1
    assert (report.verification.rewritten, report.verification.removed) == (0, 1)


@pytest.mark.anyio
async def test_delete_action_removes_the_claim(db_factory) -> None:
    def delete_all(content):
        ids = [int(i) for i in re.findall(r'<failed_claim id="(\d+)"', content)]
        return result(
            "reviser",
            parsed=ReviseOut(
                summary="Shorter.",
                revisions=[ClaimRevision(claim_id=i, action="delete") for i in ids],
            ),
        )

    report, llm, _ = await run(
        db_factory, config=no_retry(), script=script(verify_by(fail_drivers), delete_all)
    )
    assert len(claims_of(report)) == 1
    assert llm.nodes().count("verifier") == 2  # nothing rewritten, nothing re-checked
    assert report.verification.removed == 1


@pytest.mark.anyio
async def test_reviser_failure_falls_back_to_removing_failed_claims(db_factory) -> None:
    report, _, _ = await run(
        db_factory,
        config=no_retry(),
        script=script(verify_by(fail_drivers), lambda content: LLMError("boom", 0.001)),
    )
    assert all(c.verdict == "supported" for c in claims_of(report))
    assert len(claims_of(report)) == 1
    assert any("removed without rewriting" in item for item in report.limitations)


@pytest.mark.anyio
async def test_reviser_sees_quotes_and_notes_for_failed_claims(db_factory) -> None:
    _, llm, _ = await run(
        db_factory,
        config=no_retry(),
        script=script(verify_by(fail_drivers), rewrite_all("Rewritten.")),
    )
    content = next(kw["messages"][0]["content"] for n, kw in llm.calls if n == "reviser")
    assert '<failed_claim id="' in content and 'verdict="unsupported"' in content
    assert "checker quote: Official statistics page about" in content
    assert "Fact from S" in content  # the reader's notes for the cited source
    assert "<passed_claims>" in content


# --- the one-time research retry ---


def flaky_drivers_page(fail_times: int):
    """Fake web where the 'drivers' page returns 404 the first `fail_times` fetches."""
    fails = {"left": fail_times}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.host != "api.tavily.com" and DRIVERS in req.url.path and fails["left"] > 0:
            fails["left"] -= 1
            return httpx.Response(404)
        return web(req)

    return handler


def hinted(llm) -> list[str]:
    return [
        kw["messages"][0]["content"].split("\n")[0]
        for n, kw in llm.calls
        if n == "researcher" and RETRY_HINT in kw["messages"][0]["content"]
    ]


@pytest.mark.anyio
async def test_unanswered_sub_question_gets_one_retry(db_factory) -> None:
    report, llm, _ = await run(db_factory, handler=flaky_drivers_page(fail_times=1))
    assert set(hinted(llm)) == {SUBS[1]}  # only the gap is retried, with the hint
    assert llm.nodes().count("writer") == 2  # rewritten with the new notes
    assert llm.nodes().count("reader") == 2  # round 1: one page; retry: only the new page
    assert len(claims_of(report)) == 2
    assert not any("No sources were read" in item for item in report.limitations)
    assert report.status == "complete"


@pytest.mark.anyio
async def test_retry_happens_at_most_once(db_factory) -> None:
    report, llm, _ = await run(db_factory, handler=flaky_drivers_page(fail_times=99))
    assert len(hinted(llm)) == 3  # one retry researcher, 3 turns, then stop
    assert llm.nodes().count("writer") == 1  # nothing new found: report not rewritten
    assert llm.nodes().count("verifier") == 1  # and nothing re-checked
    assert f"No sources were read for: {SUBS[1]}" in report.limitations


@pytest.mark.anyio
async def test_retry_skipped_when_budget_is_short(db_factory) -> None:
    cfg = CONFIG.model_copy(deep=True)
    cfg.agent.retry_min_budget_gbp = 0.199  # more than is left after the first round
    report, llm, _ = await run(db_factory, config=cfg, handler=flaky_drivers_page(99))
    assert hinted(llm) == []
    assert report.stop_reason is None  # skipping the retry is a decision, not a limit hit
    assert f"No sources were read for: {SUBS[1]}" in report.limitations


@pytest.mark.anyio
async def test_sub_question_whose_claims_all_failed_is_retried(db_factory) -> None:
    # The drivers page is read, but every claim from it fails verification: that
    # sub-question is retried once; the second pass then revises instead of retrying.
    report, llm, _ = await run(
        db_factory, script=script(verify_by(fail_drivers), rewrite_all("Rewritten claim."))
    )
    assert set(hinted(llm)) == {SUBS[1]}
    assert llm.nodes().count("reviser") == 1  # only in the second pass
    assert "Rewritten claim." in [c.text for c in claims_of(report)]
    assert all(c.verdict == "supported" for c in claims_of(report))


# --- gap_questions ---


def _state(evidence_qs, claim_verdicts):
    evidence = [
        Evidence(source_id=f"S{i}", title="t", text="x", sub_question=q)
        for i, q in enumerate(evidence_qs)
    ]
    claims = [Claim(text="c", source_ids=[sid], verdict=v) for sid, v in claim_verdicts]
    return {
        "sub_questions": ["a", "b", "c"],
        "evidence": evidence,
        "sections": [Section(heading="h", claims=claims)],
    }


def test_gaps_are_unresearched_then_all_failed_questions() -> None:
    state = _state(["a", "b"], [("S0", "supported"), ("S1", "unsupported"), ("S1", "contradicted")])
    assert gap_questions(state, limit=5) == ["c", "b"]
    assert gap_questions(state, limit=1) == ["c"]


def test_one_passing_claim_means_no_retry_for_that_question() -> None:
    state = _state(["a", "b", "c"], [("S1", "unsupported"), ("S1", "supported")])
    assert gap_questions(state, limit=5) == []


def test_unverified_claims_do_not_trigger_a_retry() -> None:
    state = _state(["a", "b", "c"], [("S1", "unverified")])
    assert gap_questions(state, limit=5) == []
