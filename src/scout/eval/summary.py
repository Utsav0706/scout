"""Turn per-case results into the numbers for the README: overall and per category."""

from __future__ import annotations

import statistics
from typing import Any

from scout.eval.cases import CATEGORIES
from scout.eval.runner import CaseResult


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 3) if den else None


def _pct(q: float, values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def aggregate(results: list[CaseResult]) -> dict[str, Any]:
    scored = [r for r in results if r.outcome == "scored"]
    answerable = [r for r in scored if r.category != "unanswerable"]
    latencies = [r.latency_s for r in scored if r.latency_s is not None]
    return {
        "cases": len(results),
        "scored": len(scored),
        "skipped": sum(r.outcome == "skipped" for r in results),
        "errors": sum(r.outcome == "error" for r in results),
        # An error counts as a failed task: the user got no usable report.
        "task_success": _rate(
            sum(bool(r.passed) for r in scored),
            len(scored) + sum(r.outcome == "error" for r in results),
        ),
        "key_point_coverage": round(statistics.fmean(r.coverage for r in answerable), 3)
        if answerable
        else None,
        "tool_selection": _rate(sum(bool(r.tools_ok) for r in scored), len(scored)),
        "claims": sum(r.claims for r in scored),
        "claim_support": _rate(
            sum(r.claims_supported for r in scored), sum(r.claims for r in scored)
        ),
        "first_check_support": _rate(
            sum(r.first_supported for r in scored), sum(r.first_checked for r in scored)
        ),
        "citation_accuracy": _rate(
            sum(r.citations_accurate for r in scored), sum(r.citations_checked for r in scored)
        ),
        "avg_steps": round(statistics.fmean(r.steps for r in scored), 1) if scored else None,
        "avg_cost_gbp": round(statistics.fmean(r.cost_gbp for r in scored), 4) if scored else None,
        "latency_p50_s": _pct(0.5, latencies),
        "latency_p95_s": _pct(0.95, latencies),
        "research_cost_gbp": round(sum(r.cost_gbp for r in results), 4),
        "judge_cost_gbp": round(sum(r.judge_cost_gbp for r in results), 4),
    }


def summarise(results: list[CaseResult]) -> dict[str, Any]:
    by_cat = {
        cat: aggregate([r for r in results if r.category == cat])
        for cat in CATEGORIES
        if any(r.category == cat for r in results)
    }
    return {"overall": aggregate(results), "by_category": by_cat}


def headline(overall: dict[str, Any]) -> str | None:
    if not overall["scored"] or overall["claim_support"] is None:
        return None
    return (
        f"Scout answers research questions with {overall['claim_support']:.0%} of claims "
        f"verified against sources, in {overall['latency_p50_s']:.0f} s (median), for "
        f"£{overall['avg_cost_gbp']:.3f} per report; {overall['task_success']:.0%} of "
        f"golden-set questions answered correctly."
    )


def _fmt(value: Any, kind: str) -> str:
    if value is None:
        return "–"
    if kind == "pct":
        return f"{value:.0%}"
    if kind == "gbp":
        return f"£{value:.4f}"
    if kind == "s":
        return f"{value:.0f}s"
    return str(value)


ROWS = [
    ("Cases scored", "scored", "n"),
    ("Task success (judge)", "task_success", "pct"),
    ("Key-point coverage", "key_point_coverage", "pct"),
    ("Tool selection", "tool_selection", "pct"),
    ("Claim support (final report)", "claim_support", "pct"),
    ("Claim support (first check)", "first_check_support", "pct"),
    ("Citation accuracy (judge)", "citation_accuracy", "pct"),
    ("Avg steps", "avg_steps", "n"),
    ("Avg cost per report", "avg_cost_gbp", "gbp"),
    ("Latency p50", "latency_p50_s", "s"),
    ("Latency p95", "latency_p95_s", "s"),
    ("Errors", "errors", "n"),
    ("Skipped", "skipped", "n"),
]


def to_markdown(summary: dict[str, Any], meta: dict[str, Any], results: list[CaseResult]) -> str:
    cats = list(summary["by_category"])
    cols = ["Overall", *cats]
    groups = [summary["overall"], *(summary["by_category"][c] for c in cats)]
    lines = [f"# Scout golden-set eval — {meta.get('started_at', '')}", ""]
    if line := headline(summary["overall"]):
        lines += [f"> {line}", ""]
    lines += ["| Metric | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for label, key, kind in ROWS:
        lines.append(f"| {label} | " + " | ".join(_fmt(g[key], kind) for g in groups) + " |")
    o = summary["overall"]
    lines += [
        "",
        f"Spend: research £{o['research_cost_gbp']:.4f} + judges £{o['judge_cost_gbp']:.4f}.",
        f"Commit `{meta.get('git_commit', '?')}` · cases file `{meta.get('cases_sha256', '?')}` · "
        f"judge prompts {meta.get('judge_prompts', {})}",
        "",
        "## Cases that did not pass",
        "",
    ]
    failed = [r for r in results if r.outcome != "scored" or not r.passed or not r.tools_ok]
    if not failed:
        lines.append("None.")
    for r in failed:
        why = r.note or r.judge_reason or ""
        if r.tools_missing or r.tools_forbidden:
            why += f" Tools missing {r.tools_missing}, forbidden {r.tools_forbidden}."
        lines.append(f"- **{r.id}** ({r.outcome}, passed={r.passed}): {why.strip()}")
    return "\n".join(lines) + "\n"
