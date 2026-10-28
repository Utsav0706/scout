"""The golden-set eval: case file, corpus, scoring rules and the runner, with fake models
(no money spent)."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from scout.agent.report import Claim, Report, ReportSource, Section, StepRecord
from scout.agent.runtime import Services
from scout.config import Secrets
from scout.eval.cases import Case, load_cases
from scout.eval.corpus import load_corpus
from scout.eval.runner import CaseResult, create_eval_user, ingest_corpus, run_suite
from scout.eval.scorers import (
    CitationGrade,
    CitationGrades,
    KeyPointGrade,
    TaskGrade,
    check_tools,
    judge_citations,
    task_passed,
)
from scout.eval.summary import headline, summarise, to_markdown
from scout.llm.client import LLMError
from scout.rag.pdf import extract_pages
from tests.conftest import REPO_ROOT, TEST_JWT_SECRET, requires_postgres
from tests.fakes import FakeEmbedder, FakeReranker
from tests.test_agent import FakeLLM, default_script, result, web
from tests.test_tools import CONFIG

GOLDEN = REPO_ROOT / "eval" / "golden.yaml"
CORPUS = REPO_ROOT / "eval" / "corpus"


# --- the case file ---


def test_golden_set_has_50_cases_in_the_planned_mix() -> None:
    cases = load_cases(GOLDEN)
    assert len(cases) == 50
    assert Counter(c.category for c in cases) == {
        "web": 20,
        "docs": 12,
        "both": 10,
        "unanswerable": 8,
    }
    ci = load_cases(GOLDEN, tags=["ci"])
    assert {c.category for c in ci} == {"web", "docs", "both", "unanswerable"}  # one of each kind
    assert all("rag_search" in c.must_call for c in cases if c.category in ("docs", "both"))
    assert all("web_search" in c.must_call for c in cases if c.category in ("web", "both"))


def test_docs_answers_are_really_in_the_corpus() -> None:
    """Spot-check: figures the docs cases expect appear in the made-up files."""
    text = " ".join(p.read_text(encoding="ascii") for p in CORPUS.glob("*.txt"))
    for fact in ["GBP 190", "3 hours 42 minutes", "GBP 18,500", "35 kg", "GBP 6.8 million"]:
        assert fact in text
    # ...and the web half of "both" cases is not, so the web really is needed.
    for fact in ["Northamptonshire", "Sherbourne", "3.2"]:
        assert fact not in text


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ({"id": "a", "input": "Question?", "tags": ["web"]}, "needs expected key points"),
        (
            {"id": "a", "input": "Question?", "expected": ["x"], "tags": ["unanswerable"]},
            "has no expected key points",
        ),
        (
            {"id": "a", "input": "Question?", "expected": ["x"], "tags": ["web", "docs"]},
            "exactly one category",
        ),
        (
            {
                "id": "a",
                "input": "Question?",
                "expected": ["x"],
                "must_call": ["web_search"],
                "must_not_call": ["web_search"],
                "tags": ["web"],
            },
            "both required and forbidden",
        ),
    ],
)
def test_bad_cases_are_rejected(case: dict, error: str) -> None:
    with pytest.raises(ValidationError, match=error):
        Case.model_validate(case)


# --- the corpus ---


def test_corpus_becomes_pdfs_the_normal_pipeline_can_read() -> None:
    docs = load_corpus(CORPUS)
    assert len(docs) == 4 and all(name.endswith(".pdf") for name, _ in docs)
    for name, data in docs:
        text = " ".join(p.text for p in extract_pages(data, max_pages=50))
        assert "FICTIONAL DOCUMENT" in text, name
    policy = dict(docs)["quillmere_travel_expenses_policy.pdf"]
    assert "London: GBP 190" in " ".join(p.text for p in extract_pages(policy, max_pages=50))


# --- scoring rules ---

WEB = Case(id="w", input="When?", expected=["Point one.", "Point two."], tags=["web"])
NONE = Case(id="n", input="When?", tags=["unanswerable"])


def grade(covered: list[bool], **kw: Any) -> TaskGrade:
    defaults = {
        "contradicts_key": False,
        "states_insufficient_evidence": False,
        "gives_specific_answer": False,
        "reason": "r",
    }
    points = [KeyPointGrade(point=i, covered=c) for i, c in enumerate(covered, start=1)]
    return TaskGrade(key_points=points, **{**defaults, **kw})


def test_task_pass_rules() -> None:
    assert task_passed(WEB, grade([True, True])) == (True, 1.0)
    assert task_passed(WEB, grade([True, False])) == (False, 0.5)  # every point is needed
    assert task_passed(WEB, grade([True, True], contradicts_key=True)) == (False, 1.0)
    assert task_passed(NONE, grade([], states_insufficient_evidence=True)) == (True, None)
    # Saying "not sure" but still asserting an answer is a fail.
    assert not task_passed(
        NONE, grade([], states_insufficient_evidence=True, gives_specific_answer=True)
    )[0]


def test_tool_selection() -> None:
    case = Case(
        id="b",
        input="Question?",
        expected=["x"],
        must_call=["rag_search", "web_search"],
        must_not_call=["save_note"],
        tags=["both"],
    )
    assert check_tools(case, ["rag_search", "web_search"]).ok
    check = check_tools(case, ["web_search", "save_note"])
    assert not check.ok and check.missing == ["rag_search"] and check.forbidden == ["save_note"]


def report_with(claims: list[Claim], sources: list[str]) -> Report:
    return Report(
        question="q",
        status="complete",
        summary="s",
        sections=[Section(heading="h", claims=claims)],
        sources=[ReportSource(id=s, title=s, url_or_doc=f"https://x.test/{s}") for s in sources],
        cost_gbp=0.01,
        latency_s=10,
        steps=[StepRecord(node="researcher", kind="tool", name="web_search", ok=True, ms=1)],
    )


@pytest.mark.anyio
async def test_citations_judged_and_dangling_ids_fail_without_a_call() -> None:
    claims = [
        Claim(text="A.", source_ids=["S1"], verdict="supported", quote="a"),
        Claim(text="B.", source_ids=["S9"], verdict="supported", quote="b"),  # S9 not listed
        Claim(text="C.", source_ids=["S1"], verdict="unverified"),  # no quote: not sampled
    ]

    def judge(node: str, kw: dict) -> Any:
        assert node == "judge" and "B." not in kw["messages"][0]["content"]
        return result(
            node, parsed=CitationGrades(claims=[CitationGrade(claim=1, supports=True, reason="ok")])
        )

    out = await judge_citations(FakeLLM(judge), WEB, report_with(claims, ["S1"]))
    assert (out.checked, out.accurate) == (2, 1)
    assert out.failures == ["B. (cites a source not in the report)"]


# --- the runner, end to end with fake models and a fake web ---


def judge_script(passed: bool = True):
    def script(node: str, kw: dict[str, Any]) -> Any:
        if node != "judge":
            return default_script(node, kw)
        if kw["output_format"] is TaskGrade:
            n = kw["messages"][0]["content"].count("\n", *_key_span(kw)) + 1
            return result(node, parsed=grade([passed] * n), cost=0.01)
        n = kw["messages"][0]["content"].count("Claim:")
        grades = [CitationGrade(claim=i, supports=True, reason="ok") for i in range(1, n + 1)]
        return result(node, parsed=CitationGrades(claims=grades), cost=0.005)

    return script


def _key_span(kw: dict[str, Any]) -> tuple[int, int]:
    text = kw["messages"][0]["content"]
    start = text.index("<answer_key>\n") + len("<answer_key>\n")
    return start, text.index("\n</answer_key>")


def services_for(db_factory, script) -> Services:
    secrets = Secrets(
        _env_file=None,
        database_url="sqlite://",
        jwt_secret=SecretStr(TEST_JWT_SECRET),
        tavily_api_key=SecretStr("tvly-test"),
    )
    return Services(
        config=CONFIG,
        secrets=secrets,
        session_factory=db_factory,
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        llm=FakeLLM(script),
        http_transport=httpx.MockTransport(web),
    )


CASES = [
    Case(
        id="w1",
        input="Why did UK inflation rise?",
        expected=["A.", "B."],
        must_call=["web_search"],
        tags=["web"],
    ),
    Case(
        id="d1",
        input="What is in my files?",
        expected=["A."],
        must_call=["rag_search"],
        tags=["docs"],
    ),
    Case(id="n1", input="What will happen in 2031?", tags=["unanswerable"]),
]


@pytest.mark.anyio
async def test_suite_runs_scores_and_reports(db_factory, tmp_path: Path) -> None:
    seen: list[CaseResult] = []
    results = await run_suite(
        services_for(db_factory, judge_script(passed=True)),
        CASES,
        corpus_dir=CORPUS,
        use_docs=False,  # SQLite: no document search, so the docs case is skipped
        on_result=seen.append,
    )
    assert [r.id for r in results] == ["w1", "d1", "n1"] and len(seen) == 3
    w1, d1, n1 = results
    assert w1.outcome == "scored" and w1.passed and w1.coverage == 1.0
    assert w1.tools_ok and w1.tools_called[:2] == ["web_search", "fetch_url"]
    assert w1.claims == w1.claims_supported > 0 and w1.first_checked == w1.first_supported
    assert w1.citations_checked == w1.citations_accurate > 0
    assert w1.cost_gbp > 0 and w1.judge_cost_gbp == pytest.approx(0.015)
    assert d1.outcome == "skipped" and "Postgres" in d1.note
    assert n1.outcome == "scored" and n1.tools_ok  # nothing required of unanswerable cases

    summary = summarise(results)
    assert summary["overall"]["scored"] == 2 and summary["overall"]["skipped"] == 1
    assert summary["by_category"]["web"]["claim_support"] == 1.0
    assert headline(summary["overall"]).startswith("Scout answers research questions with 100%")
    md = to_markdown(summary, {"started_at": "now"}, results)
    assert "| Task success (judge) |" in md and "**d1** (skipped" in md


@pytest.mark.anyio
async def test_a_failed_judge_keeps_the_research_cost(db_factory) -> None:
    def script(node: str, kw: dict[str, Any]) -> Any:
        return LLMError("judge down") if node == "judge" else default_script(node, kw)

    [r] = await run_suite(
        services_for(db_factory, script), CASES[:1], corpus_dir=CORPUS, use_docs=False
    )
    assert r.outcome == "error" and "judge failed" in r.note
    assert r.cost_gbp > 0 and r.claims > 0  # the run itself is still recorded
    assert summarise([r])["overall"]["task_success"] == 0.0  # errors count as failures


@pytest.mark.anyio
async def test_spend_cap_stops_new_cases(db_factory) -> None:
    results = await run_suite(
        services_for(db_factory, judge_script()),
        [CASES[0], CASES[2]],
        corpus_dir=CORPUS,
        use_docs=False,
        concurrency=1,
        max_cost_gbp=0.001,  # the first case spends more than this
    )
    assert results[0].outcome == "scored"
    assert results[1].outcome == "skipped" and "spend cap" in results[1].note


@requires_postgres
@pytest.mark.anyio
async def test_corpus_ingests_for_a_fresh_eval_user(db_factory) -> None:
    services = services_for(db_factory, judge_script())
    user_id = create_eval_user(services)
    assert await ingest_corpus(services, user_id, CORPUS) == 4
