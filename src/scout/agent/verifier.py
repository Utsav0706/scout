"""The verifier node: check every claim against the source text it cites.

Self-contained on purpose (the builder notes suggest rewriting it yourself; the tests in
tests/test_verifier.py pin down the behaviour any rewrite must keep).

How it works:
1. Group claims by cited source, so each source is read ONCE with all its claims
   (a few calls per report instead of one per claim).
2. A model gives each claim a verdict plus 1-3 verbatim quotes from the source (more
   than one when a claim combines facts from different sentences).
3. Code, not the model, checks every quote really appears in the source. A supported or
   contradicted verdict with a quote that isn't there is downgraded to "unsupported":
   the checker can't make up its own evidence.
4. A claim citing several sources: supported if any source supports it; otherwise
   contradicted if any contradicts it; otherwise unsupported.
It checks against the source text the researcher read (not the reader's paraphrased
notes), so a fact garbled while taking notes is caught too.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from langgraph.runtime import Runtime

from scout.agent.budget import BudgetExceeded
from scout.agent.calls import llm_call
from scout.agent.report import Claim, Section, Verification
from scout.agent.state import AgentDeps, ClaimVerdict, Evidence, ResearchState, VerifyOut
from scout.llm.client import LLMError

_CONCURRENCY = 4
# A quote shorter than this proves nothing ("in 2026", "the UK" appear everywhere).
MIN_QUOTE_WORDS = 4

# Words and numbers; a number keeps its decimal point / thousands separator ("3.5", "1,000").
_TOKEN = re.compile(r"[0-9]+(?:[.,][0-9]+)*|[a-z0-9]+")


def tokens(text: str) -> list[str]:
    """Lower-case words and numbers, all punctuation dropped.

    Comparing words instead of characters makes the check immune to what HTML extraction
    and typography do to text (curly vs straight quotes, dashes, line breaks, and the
    space it puts before commas: "models , may"), while still requiring every word and
    number of the quote, in order. Found live: honest quotes failed on " , " spacing.
    """
    return _TOKEN.findall(text.lower())


def quote_in_source(quote: str, source_text: str) -> bool:
    q = tokens(quote)
    if len(q) < MIN_QUOTE_WORDS:
        return False
    return f" {' '.join(q)} " in f" {' '.join(tokens(source_text))} "


@dataclass(frozen=True)
class CheckedVerdict:
    verdict: str  # supported | unsupported | contradicted
    quote: str | None


def check_verdict(v: ClaimVerdict, source_text: str) -> CheckedVerdict:
    """Apply the quote check to one model verdict.

    A claim can combine facts from different sentences, so the model may give up to
    three quotes. Supported/contradicted needs at least one quote and EVERY quote must
    be found: one invented quote is enough to discard the verdict.
    """
    quotes = [q.strip() for q in v.quotes[:3] if q.strip()]
    real = [q for q in quotes if quote_in_source(q, source_text)]
    if v.verdict in ("supported", "contradicted") and (not quotes or len(real) < len(quotes)):
        return CheckedVerdict("unsupported", None)
    return CheckedVerdict(v.verdict, " … ".join(real) or None)


def combine(per_source: list[CheckedVerdict]) -> CheckedVerdict | None:
    """One verdict for a claim from its per-source verdicts (None = never checked)."""
    for wanted in ("supported", "contradicted"):
        for v in per_source:
            if v.verdict == wanted:
                return v
    return per_source[0] if per_source else None


def _source_texts(evidence: list[Evidence], max_chars: int) -> dict[str, tuple[str, str]]:
    """source_id -> (title, text), truncated exactly as the reader saw it, so claims are
    judged against the same text their notes came from."""
    out: dict[str, tuple[str, str]] = {}
    for ev in evidence:
        out.setdefault(ev.source_id, (ev.title, ev.text[:max_chars]))
    return out


async def check_claims(
    deps: AgentDeps, claims: list[Claim], evidence: list[Evidence]
) -> list[Claim]:
    """Return copies of `claims` with verdict and quote set. Used by the verifier node
    and again by the reviser for rewritten claims, so both follow the same rules."""
    texts = _source_texts(evidence, deps.config.agent.max_source_chars)
    by_source: dict[str, list[int]] = {}
    for i, claim in enumerate(claims):
        for sid in claim.source_ids:
            if sid in texts:
                by_source.setdefault(sid, []).append(i)

    system = deps.prompts["verifier"].render()
    gate = asyncio.Semaphore(_CONCURRENCY)
    results: dict[int, list[CheckedVerdict]] = {}

    async def check_source(sid: str, ids: list[int]) -> None:
        title, text = texts[sid]
        numbered = "\n".join(f"[{i}] {claims[i].text}" for i in ids)
        content = (
            f'<source id="{sid}" title="{title}">\n{text}\n</source>\n\n'
            f"<claims>\n{numbered}\n</claims>"
        )
        async with gate:
            try:
                res = await llm_call(
                    deps,
                    "verifier",
                    system=system,
                    output_format=VerifyOut,
                    messages=[{"role": "user", "content": content}],
                )
            except (BudgetExceeded, LLMError):
                return  # these claims stay unverified for this source
        answered = {v.claim_id: v for v in res.parsed.verdicts}  # type: ignore[union-attr]
        for i in ids:
            v = answered.get(i)
            # A claim the model skipped wasn't shown to be supported: count it as unsupported.
            checked = check_verdict(v, text) if v else CheckedVerdict("unsupported", None)
            results.setdefault(i, []).append(checked)

    await asyncio.gather(*(check_source(sid, ids) for sid, ids in by_source.items()))

    out = []
    for i, claim in enumerate(claims):
        final = combine(results.get(i, []))
        out.append(
            claim.model_copy(
                update={
                    "verdict": final.verdict if final else "unverified",
                    "quote": final.quote if final else None,
                }
            )
        )
    return out


def count_verdicts(claims: list[Claim]) -> Verification:
    counts = {"supported": 0, "unsupported": 0, "contradicted": 0, "unverified": 0}
    for c in claims:
        counts[c.verdict] += 1
    return Verification(checked=len(claims), **counts)


async def verifier(state: ResearchState, runtime: Runtime[AgentDeps]) -> dict[str, Any]:
    deps = runtime.context
    sections: list[Section] = state.get("sections") or []
    claims = [c for sec in sections for c in sec.claims]
    todo = [i for i, c in enumerate(claims) if c.verdict == "unverified"]
    if not todo:
        return {}  # nothing new to check (e.g. a retry round that changed nothing)
    checked = await check_claims(deps, [claims[i] for i in todo], state.get("evidence", []))
    verified = list(claims)
    for i, claim in zip(todo, checked, strict=True):
        verified[i] = claim
    # Rebuild sections in the same order with the verdicts attached.
    it = iter(verified)
    new_sections = [
        Section(heading=s.heading, claims=[next(it) for _ in s.claims]) for s in sections
    ]
    return {"sections": new_sections, "verification": count_verdicts(verified)}
