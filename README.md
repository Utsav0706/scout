# Scout

An autonomous research agent. Ask a question; Scout plans the research, searches the
web and a private document store, reads sources, verifies every claim, and returns a
report where each claim links to its source.

> Headline (filled in once measured): *"Scout answers research questions with X% of
> claims verified against sources, in Y seconds, for £Z per report."*

Status: **M5a — verified claims.** Ask a question from the command line and get a cited
report in which every claim has been checked against the source text it cites.

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
START -> planner -> researcher x N (in parallel) -> reader -> writer -> verifier -> finaliser -> END
```

| Step | Model | Job |
|---|---|---|
| planner | Sonnet 5.5 | Split the question into 2-4 sub-questions + a search plan |
| researcher | Haiku 5.5 | Per sub-question, up to 3 rounds of tool calls (search, read pages, search docs) |
| reader | Haiku 5.5 | Extract facts from each source read; saved as notes tied to the source id |
| writer | Sonnet 5.5 | Write the report from the notes only; every claim cites source ids |
| verifier | Haiku 5.5 | Check each claim against the source text it cites: supported / unsupported / contradicted, with verbatim quotes |
| finaliser | (code) | Drop invented source ids, list cited sources, add cost/steps; markdown + JSON |

Limits on every run: 25 model calls, £0.20, 180 s. Research stops early enough to keep
money for the writer; if a limit is still hit, you get a partial report saying why.
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

## Run it

With Docker (API + Postgres):

```bash
docker compose up --build
```

Then open http://localhost:8000/docs for the interactive API docs.

| Endpoint | What |
|---|---|
| `GET /health` | `200` if the API and DB are up, `503` if the DB is down |
| `POST /auth/register` | `{email, password}` → creates a user |
| `POST /auth/login` | `{email, password}` → `{access_token, expires_in}` |
| `GET /auth/me` | Needs `Authorization: Bearer <token>` |
| `POST /documents` | Upload a PDF (multipart `file`, max 20 MB) → chunked + embedded |
| `GET /documents` | Your uploaded documents |
| `POST /documents/search` | `{query, document_ids?, top_k?}` → best chunks with page numbers |

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
| `prompts/` | Versioned prompt files (from M4). |
| `docs/` | Design notes. |
| `tests/` | pytest suite, run in CI on every push/PR. |
