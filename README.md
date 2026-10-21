# Scout

An autonomous research agent. Ask a question; Scout plans the research, searches the
web and a private document store, reads sources, verifies every claim, and returns a
report where each claim links to its source.

> Headline (filled in once measured): *"Scout answers research questions with X% of
> claims verified against sources, in Y seconds, for £Z per report."*

Status: **M3 — MCP tools.** Sign-up/login, PDF upload + hybrid search, and the agent's
four tools served over MCP. No research agent yet.

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
