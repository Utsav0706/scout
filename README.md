# Scout

An autonomous research agent. Ask a question; Scout plans the research, searches the
web and a private document store, reads sources, verifies every claim, and returns a
report where each claim links to its source.

> **Scout answers research questions in 30 s (median) for £0.024 per report.** On a
> 50-question golden set it fully answers 68%, says "not enough evidence" instead of
> inventing an answer on 8/8 trick questions, and every claim it keeps passes its own
> verifier; an independent judge agrees with 66% of sampled citations.
> ([baseline](eval/results/baseline-2026-10-28/summary.md), 28 Oct 2026)

Status: **M11a — system design doc.** `POST /research` returns a job id
at once; `GET /research/{id}/stream` streams live progress; `GET /research/{id}` returns
the report. PDF uploads are processed by the worker too.
Every claim in the report has been checked against the source text it cites.

## Architecture

```mermaid
flowchart LR
    ui["Streamlit UI"] -- "HTTP + JWT" --> api["FastAPI"]
    api -- "job id" --> q[("Redis queue")]
    q --> w["Worker: agent graph"]
    w --> mcp["MCP tools<br/>web_search · fetch_url<br/>rag_search · save_note"]
    mcp --> ext["Tavily + web"]
    mcp --> pg[("Postgres + pgvector")]
    api <--> pg
    w <--> pg
    w --> llm["Claude Sonnet / Haiku"]
    w -. "traces" .-> lf["Langfuse"]
```

Agent graph: **planner → parallel researchers → reader → writer → verifier → reviser**
(one retry round if needed) **→ finaliser**, inside a hard budget of 50 model calls,
£0.20 and 180 s per report. The full design, with every guardrail and the reasons
behind each choice, is in **[docs/design.md](docs/design.md)**.

## Ask a question

```bash
py -m scout.agent.run "What is the current UK CPI inflation rate?" --user you@example.com
```

Without Postgres running, use a local SQLite file (web search only):

```bash
py -m scout.agent.run "your question" --user you@example.com --create-user --database-url sqlite:///scout_local.db --json runs/report.json
```

### How the agent works

```
START -> planner -> researcher x N (in parallel) -> reader -> writer -> verifier -> reviser
                          ^                                                       |
                          +--------- retry round for gaps (at most once) ---------+
                                                                                  v
                                                                      finaliser -> END
```

| Step | Model | Job |
|---|---|---|
| planner | Sonnet 5.5 | Split the question into 2-4 sub-questions + a search plan |
| researcher | Haiku 5.5 | Per sub-question, up to 3 rounds of tool calls (search, read pages, search docs) |
| reader | Haiku 5.5 | Extract facts from each source read; saved as notes tied to the source id |
| writer | Sonnet 5.5 | Write the report from the notes only; every claim cites source ids |
| verifier | Haiku 5.5 | Check each claim against the source text it cites: supported / unsupported / contradicted, with verbatim quotes |
| reviser | Sonnet 5.5 | Only if something failed: rewrite (then re-verify) or delete failed claims; rewrite the summary. Sends unanswered sub-questions back for one more research round |
| finaliser | (code) | Drop invented source ids, list cited sources, add cost/steps; markdown + JSON |

Limits on every run: 50 model calls, £0.20, 180 s. Research stops early enough to keep
money for writing, checking and revising; if a limit is still hit, you get a partial
report saying why.
First live runs: **£0.023 / 35 s** and **£0.028 / 40 s** per report (M4); with
verification **£0.030–0.033 / 40 s**.

### How claims are verified

Claims are grouped by the source they cite, so each source is read once (3-6 cheap
calls per report). For every claim the model returns a verdict and 1-3 **verbatim
quotes**; then plain code (not the model) checks every quote really occurs in the source,
word for word and in order. A "supported" or "contradicted" verdict with a quote that
isn't there is downgraded to "unsupported", so the checker can't invent its own
evidence. The report's `verification` block gives the support rate, and the markdown
flags any claim that isn't supported.

Found while building it (live runs, checked by hand):
- The model sometimes stitched two sentences into one quote to back a claim that
  combines two facts; it now gives up to 3 separate quotes.
- Honest quotes failed because HTML extraction puts a space before commas
  (`models , may`); quotes are now compared word by word, ignoring punctuation.
- A loose "contradicted" rule flagged a correct claim; the rubric now requires a
  conflicting statement about the *same* measure and period (prompt `verifier_v2`→`v3`).

### Revision and the retry round

If every claim passes, the reviser makes no model call. Otherwise it rewrites or deletes
each failed claim; every rewrite goes through the same verifier again and is dropped if
it still fails, so the final report contains only claims that passed a check (or are
explicitly flagged "not checked" when a limit stopped verification). The `verification`
block keeps the **first-check** support rate (how faithful the writer was) plus how many
claims were rewritten and removed.

Sub-questions that ended with no sources read, or whose claims all failed, get **one**
more research round with a hint to search differently (only if £0.10+ of budget is left).
A per-job fetch cache stops parallel researchers re-downloading, or re-trying, the same URL.

**Blocked pages and PDFs:** many UK policy sites (parliament.uk, health.org.uk, ...)
answer 403 to non-browser clients, and HTML parsing can't read PDFs. When the direct fetch
is refused (401/403/406/429/451) or returns a non-HTML file, `fetch_url` reads the page
through [Tavily Extract](https://docs.tavily.com) instead (1 credit per 5 successful
pages; failures free). Scout never disguises its own client as a browser. Never used for
404s, timeouts, or URLs the SSRF guard refused. Same NHS question, before/after:

| | fetches that worked | pages read | claims | unanswered sub-questions | cost |
|---|---|---|---|---|---|
| without fallback | 3 / 13 | 2 | 13 | 1 | £0.027 |
| with fallback | 7 / 7 | 6 | 19 | 0 | £0.038 |

Each report's footer shows its Tavily use (searches + pages via Extract).

## Run it

With Docker (Postgres, Redis, API, worker):

```bash
docker compose up --build
```

Then open http://localhost:8501 for the web UI, or http://localhost:8000/docs for the
interactive API docs.

| Endpoint | What |
|---|---|
| `GET /health` | `200` if the API and DB are up, `503` if the DB is down |
| `POST /auth/register` | `{email, password}` → creates a user |
| `POST /auth/login` | `{email, password}` → `{access_token, expires_in}` |
| `GET /auth/me` | Needs `Authorization: Bearer <token>` |
| `POST /documents` | Upload a PDF (multipart `file`, max 20 MB) → **202**, `status: processing`; the worker chunks + embeds it |
| `GET /documents` | Your documents with `status` (`processing` / `ready` / `failed` + `error`) |
| `POST /documents/search` | `{query, document_ids?, top_k?}` → best chunks with page numbers |
| `POST /research` | `{question, use_web?, use_docs?}` → **202** `{job_id, status_url}` |
| `GET /research/{id}` | Status, progress events, the report (JSON) when finished, and your feedback |
| `GET /research/{id}/trajectory` | Every model and tool call of a finished run (inputs/outputs masked), plus sources and claims; for evaluation tools |
| `GET /research/{id}/stream` | Server-sent events: `progress` per step, then `done`; resumes from `Last-Event-ID` |
| `GET /research` | Your past jobs, newest first (`limit` 1–100; next page: `after=<id of the last job>`), with your rating |
| `GET /research/stats` | Your last `days` (1–90): jobs, spend, average latency, support rate, thumbs up/down, jobs per day |
| `POST /research/{id}/feedback` | `{rating: up\|down, comment?}` on a finished job; rating again replaces it. Also sent to Langfuse as a `user_feedback` score on that run's trace |

Research jobs run on an **RQ worker** (Redis) in production. For local development on
Windows, where RQ workers can't run, set `JOBS_MODE=inline` and the API runs jobs itself
(same job code). Per user: at most 2 jobs running and 10 per hour (429 + `Retry-After`).
Someone else's job id answers 404.

```bash
set JOBS_MODE=inline
uvicorn scout.api.main:create_app --factory
```

## Web UI (Streamlit)

A Streamlit app talks to the API over HTTP (it never touches the database), so it can
run anywhere the API is reachable. Log in or create an account, ask a question, watch
each step stream in, then read the report: every claim has a verdict badge and a
**Sources** pop-up with the link and the exact words the verifier matched. Under the
report, rate it 👍/👎 with an optional comment.

| Page | What |
|---|---|
| Ask | Question form, live progress, the report and its feedback form |
| History | Your past reports, newest first, 20 per page; **Open** shows one again |
| Documents | Upload PDFs and see their status (processing / ready / failed) |
| Usage | Reports, average cost and time, failure rate and 👍/👎 for the last 7/30/90 days |

In Docker Compose the `ui` service has its own small image (`Dockerfile.ui`: Streamlit +
httpx only, no torch or models) and calls the API at `http://api:8000`.

Model-written text is escaped before display: no links, HTML or formatting from a model
or a web page reach the screen, and prices like "$5" don't turn into maths.

Local development on Windows, no Docker needed (two terminals):

```bash
pip install -e ".[dev]"
py scripts/dev_api.py
py scripts/dev_ui.py
```

`dev_api.py` runs the API on http://localhost:8001 with `JOBS_MODE=inline` and a local
SQLite file (`scout_dev.db`; document search needs Postgres). `dev_ui.py` opens the UI on
http://localhost:8501. Point the UI at another API with `SCOUT_API_URL`.

The first upload or search downloads two small local models (~200 MB total) from
Hugging Face; after that they are cached.

### How search works

```
query ──► embed (bge-small) ──► pgvector cosine top 20 ──┐
     └──► Postgres full-text (websearch_to_tsquery) top 20 ─┴─► RRF fuse ──► cross-encoder rerank ──► top 5
```

Vectors catch paraphrases ("money back" ≈ "refund"); keywords catch exact names, codes
and numbers. RRF merges the two ranked lists without having to compare their raw scores.
The cross-encoder then reads query + chunk together to put the best ones first. Every
search is limited to the caller's own documents.

## Tracing (Langfuse)

Set `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` and every run becomes one trace in
[Langfuse](https://langfuse.com): a span per graph step, a *generation* per model call
(model, tokens, cost) and a *tool* observation per tool call, each with its inputs and
outputs (personal data masked, cut to 4,000 chars; `telemetry.capture_content: false`
sends only names, tokens, cost and timing). Every span carries the user and job ids.
Reports include their `trace_id`.

Built on OpenTelemetry (exported over OTLP), not Langfuse's SDK: vendor-neutral, and the
same standard Gearbox and Dyno use. Only Scout's own spans are exported: the Anthropic and
MCP SDKs trace themselves too, and on the first live trace their copies duplicated every
call (137 observations for 62 calls) and doubled the cost Langfuse showed. Now one
observation per call, and Langfuse's cost matches Scout's to the penny (about 77 units a
run; the free tier covers ~650 runs a month).

## Evaluation (golden set)

`eval/golden.yaml` holds 50 questions with the key points a correct report must cover:

| Category | Cases | What it tests |
|---|---|---|
| web | 20 | Stable public facts (laws, dates, science) |
| docs | 12 | Facts only in `eval/corpus/`: made-up company files, so the web can't answer them |
| both | 10 | One fact from the files plus one from the web |
| unanswerable | 8 | The right report says the evidence isn't there, without inventing an answer |

Five cases tagged `ci` make a small subset. The format (`id, input, expected, must_call,
must_not_call, max_steps, tags`) matches Dyno's, so Dyno can load the same file.

| Metric | How it's measured |
|---|---|
| Task success | A judge model checks every key point is in the report (or, for unanswerable cases, that it says so and doesn't make up an answer). The pass rule is in code. |
| Tool selection | The run called the tools the case needs (`must_call`) and none it mustn't |
| Claim support | Share of claims in the final report the verifier marked supported, and first-check rate |
| Citation accuracy | A judge re-checks up to 8 claims per report against their verified quotes. Sonnet judges, Haiku verifies, so it's not the verifier marking its own work. |
| Steps, cost, latency | From each report; latency as p50 and p95 |

The judge prompts are versioned like the agent's (`prompts/judge_task_v1.md`,
`prompts/judge_citation_v1.md`). Every case gets every tool, because choosing them is part of the test.

```bash
py -m scout.eval run --tags ci                        # 5 cases, about £0.17
py -m scout.eval run                                  # all 50, about £1.65
py -m scout.eval run --tags web --database-url sqlite:///scout_eval.db   # no Postgres
```

This **costs money**: it runs the real agent and judges. It stops starting new cases at
`--max-cost-gbp` (default £5). Results go to `runs/eval/<time>/` (`results.jsonl` with
each case's Langfuse trace id, `summary.json`, `summary.md`). `--min-task-success 0.8`
exits 1 below that, which is the eval gate for the deploy pipeline. Docs cases need
Postgres + pgvector; on SQLite they're skipped.

On GitHub: **Actions → eval → Run workflow**. Set `tags` to `ci` or `all`, or list `ids`. It needs the
`ANTHROPIC_API_KEY` and `TAVILY_API_KEY` repo secrets; results show on the run's summary page.

### Baseline results (28 Oct 2026, commit `1ee463a`)

All 50 cases ran on real models (GitHub Actions, Postgres + pgvector). Spend £1.63:
£1.18 research plus £0.45 judges.

| Metric | Overall | web | docs | both | unanswerable |
|---|---|---|---|---|---|
| Task success (judge) | **68%** | 65% | **100%** | **10%** | **100%** |
| Key-point coverage | 81% | 81% | 100% | 57% | – |
| Tool selection | 100% | 100% | 100% | 100% | 100% |
| Claim support, final report | 100% | 100% | 100% | 100% | 100% |
| Claim support, first check | 87% | 88% | 94% | 78% | 83% |
| Citation accuracy (independent judge) | 66% | 74% | 76% | 54% | 45% |
| Avg cost per report | £0.024 | £0.028 | £0.017 | £0.023 | £0.024 |
| Latency p50 / p95 | 30 s / 49 s | 39 / 54 s | 24 / 26 s | 27 / 45 s | 31 / 42 s |

What it shows:
- **Strong:**
  - Document questions (12/12).
  - Refusing to invent answers (8/8 unanswerable questions).
  - Picking the right tools (50/50).
- **Weakest: questions needing both sources (1/10).** Scout finds the document fact,
  then misses the web half and says so, instead of guessing. 11 of the 16 failures
  overall are "couldn't find it", not wrong answers.
- **Verifier vs independent judge.** The reviser removes or rewrites claims that fail
  verification, so every kept claim passes Scout's own verifier. Yet an independent
  judge (Sonnet, checking claim against quote) agrees with only 66% of sampled
  citations. Either the verifier is too lenient or the judge too strict. Settling
  that needs human-labelled examples, which is Dyno's calibration step.
- **Strict key points.** 4 failures miss a key point the question didn't strictly ask
  for (e.g. "at Google" for the transformer paper).
- **One answer key was wrong.** HMRC raised its mileage rate to 55p in 2026-27. Scout
  reported this correctly and the key had the old 45p. The key is fixed; the
  baseline is left as it ran (graded against the fixed key, it would be 35/50, 70%).

These are the "before" numbers. Fixing the failures (starting with "both"
questions) is left for Dyno, so the before/after comparison is measured.

## Safety: outside text is data, never instructions

Web pages, search snippets and uploaded PDFs can contain text aimed at the AI ("ignore
your instructions and..."). Scout's defence has three layers; only the last one tries to
*spot* attacks, and the first one is what actually limits the damage:

1. **Capability limits.** No tool can send, post, buy or delete (tools carry MCP
   `readOnlyHint` annotations, and a test fails if a write-capable tool appears).
   `fetch_url` opens only links this job's searches returned or the question contained,
   so a page can't steer the agent to `https://attacker/?q=<your data>`; it also refuses
   internal addresses (SSRF). Document search is bound to the job's user; notes must cite
   real source ids.
2. **Structure.** Outside text reaches the models inside `<source>` tags (which pages
   can't close), and every prompt says it is data, never instructions.
3. **Detection.** A rule-based scanner (free, deterministic, tested in CI) checks every
   page, snippet and PDF passage for override phrases, fake role markers, tool directives,
   exfiltration links and **hidden Unicode** (invisible "tag" characters, zero-width
   spaces). Hidden characters are stripped; a flagged source is kept (an article *about*
   prompt injection is legitimate) but the model gets a warning, the finding is logged,
   and the report says which sources contained such text.

**Logs and the audit table mask personal data and secrets** (emails, UK phone and NI
numbers, Luhn-checked card numbers, API keys, tokens). Report text is made safe to render:
images from pages become `[image removed]` (an image URL could otherwise leak data to
whoever's server it points at the moment the report is viewed), and raw HTML can't run.

### Attack tests and a live red-team

`tests/security/test_attacks.py` (runs in CI) scripts a **worst-case model** that does
whatever a page tells it: fetch the attacker's URL, call tools it wasn't given, cite
invented sources, read another user's documents. Every attack fails because of what the
system allows, not because the model said no.

`scripts/redteam.py` measures the **real** models: four pages with planted instructions
(visible override, instructions hidden in invisible Unicode, a data-exfiltration fetch, a
fake system message), each with a unique canary, served from a fake web (no Tavily
credits). Result: **the model obeyed 0 of 4**, and all 4 attack pages were flagged.
The first run found a scanner gap (an exfiltration instruction phrased as "verify by
fetching ...?d=<your notes>" wasn't flagged); it's fixed and kept as a regression test.
Total cost of the red-team: £0.055.

Deliberately **no keyword topic filter**: keyword lists block innocent questions and miss
real ones; Claude's own safety training and Scout's refusal handling cover harmful requests.

## Tools (MCP server)

| Tool | What | Guardrails |
|---|---|---|
| `web_search(query)` | Tavily web search, top 5 results | Needs `TAVILY_API_KEY`; 1 credit per call |
| `fetch_url(url)` | Main text of a web page | Public http/https only, checked on every redirect (SSRF); 2 MB / 20k chars caps |
| `rag_search(query, document_ids?, top_k?)` | Passages from your uploaded PDFs | Scoped to the job's user, never an LLM-supplied id |
| `save_note(text, source_id)` | Save a fact for the report | Only accepts source ids the tools returned |

Every result carries a short source id (`S1`, `S2`...) used for citations. Each call has a
timeout, returns errors the agent can act on, and writes a row to the `tool_calls` audit
table, including calls rejected for bad arguments.

Try the tools from any MCP client (here MCP Inspector) as an existing Scout user:

```bash
set SCOUT_MCP_USER_EMAIL=you@example.com
npx @modelcontextprotocol/inspector py -m scout.tools.server
```

## Dev setup

Needs Python 3.12+.

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
copy .env.example .env
```

Set `JWT_SECRET` in `.env` (instructions are in the file).

Check everything. Locally, tests use in-memory SQLite and skip the search tests (they
need pgvector). CI runs everything on real Postgres. To run them locally with Docker:

```bash
docker compose up -d db
set TEST_DATABASE_URL=postgresql+psycopg://scout:scout@localhost:5432/scout
pytest
```

Lint and test:

```bash
ruff check .
ruff format --check .
pytest
```

## Layout

| Path | What |
|---|---|
| `config.yaml` | Models per agent node, cost/step limits. No secrets. |
| `.env` | API keys (copy from `.env.example`, never committed). |
| `src/scout/` | The package. |
| `src/scout/rag/` | PDF ingest, chunking, embeddings, hybrid search. |
| `src/scout/tools/` | MCP server and the four tools. |
| `src/scout/ui/` | `ScoutClient` (HTTP client for the API) and the Streamlit pages. |
| `ui/app.py` | Streamlit entry point. |
| `scripts/` | Dev launchers, compose smoke test, red-team run. |
| `Dockerfile`, `Dockerfile.ui` | API/worker image; lean UI image. |
| `prompts/` | Versioned prompt files (from M4). |
| `src/scout/eval/` | Golden-set eval: cases, corpus, scorers, judges, runner. |
| `eval/` | `golden.yaml` (50 cases) and `corpus/` (made-up company files). |
| `docs/` | [System design](docs/design.md) and notes. |
| `tests/` | pytest suite, run in CI on every push/PR. |
