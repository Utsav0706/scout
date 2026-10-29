# Scout FAQ

Common questions from people looking at Scout for the first time. For how everything
fits together, see the [system design](design.md).

## What is it?

**What does Scout do?**
You ask a research question. Scout splits it into sub-questions, searches the web and
your own PDFs, reads the sources, and writes a short report. Every claim in the report
cites a source, and each one is checked against that source's own words before you
see it.

**What does a report look like?**
- A summary, then sections of claims.
- Each claim shows a verdict badge and a **Sources** button. The button opens the source
  link and the exact passage that supports the claim.
- At the end: the limitations Scout noticed, the full source list, and the cost, time and
  number of model calls.

**Is it a chatbot?**
No. Each question is one research job that runs to completion (usually 25–50 seconds)
and produces one report. There's no back-and-forth conversation.

## Running it

**How do I run it?**
- **With Docker:** `docker compose up --build`, then open http://localhost:8501.
- **Without Docker** (e.g. on Windows): run `scripts/dev_api.py` and `scripts/dev_ui.py`
  in two terminals. This mode uses a local SQLite file, so PDF search is unavailable,
  but web research works fully.

The [README](../README.md) has the full steps.

**What do I need?**
- **Python 3.12+.**
- **An Anthropic API key.**
- **A Tavily API key** for web search (free tier: 1,000 searches a month).
- **Langfuse keys** only if you want tracing.
- Copy `.env.example` to `.env` and fill in the keys.

**What does it cost to run?**
- About **£0.02–0.04 per report**, from your Anthropic credit (average £0.024 across 50
  test questions).
- Each report is capped at **£0.20, 50 model calls and 3 minutes**.
- The embedding and reranking models run locally, so they're free.

**Is there a hosted demo?**
Not at the moment. A production setup (HTTPS, migrations, sign-up invite code, daily
spend cap) is in [`deploy/`](../deploy), ready for when it's hosted.

## Can I trust the answers?

**How does it avoid made-up citations?**
- **Notes only:** the writer sees only notes, each tagged with its source, so every claim
  points to a real source.
- **A verified quote:** a separate verifier must quote the exact sentence it relied on,
  and code checks that the sentence really appears in the source. If it doesn't, the
  claim doesn't count as supported.
- **Revision:** claims that fail are rewritten (and checked again) or deleted.

**What if it can't find an answer?**
It says so, rather than guessing. In testing it handled all 8 deliberately unanswerable
questions this way, such as a made-up company's revenue or a future event.

**How accurate is it?**
On a 50-question test set with known answers (28 Oct 2026):
- **68%** of questions fully answered
- **12/12** questions about uploaded documents
- **8/8** unanswerable questions handled correctly
- **50/50** correct tool choices
- **Weakest:** questions that need both a document *and* the web (1/10)

The full results and what they mean are in the [README](../README.md#baseline-results-28-oct-2026-commit-1ee463a).

**Can I check how an answer was produced?**
Yes:
- `GET /research/{id}/trajectory` returns every model and tool call of a run.
- With Langfuse keys set, each run is also a browsable trace with costs and timings.

## Privacy and safety

**Are my PDFs sent anywhere?**
- **Indexing:** embeddings and search run on your machine, and only you can search your
  documents.
- **Answering:** the passages Scout picks from your documents are sent to the model, the
  same as web text is.

**What if a web page contains instructions aimed at the AI?**
- **Pages are data, never instructions.** Hidden text is stripped, instruction-like text
  is flagged and noted in the report's limitations, and the agent's tools are tightly
  limited.
- **Links:** it can only open links its own searches returned, or links you put in the
  question.
- **Nothing can send data out.**
- **Tested:** 14 attack scenarios run in CI on every change.

**Can it be made to reach internal servers?**
No. Page fetching only allows public addresses on standard ports, and every redirect is
checked again.

## Using it from other tools

**Is there an API?**
Yes, a REST API with JWT login. Interactive docs are at `/docs` when the API is running.
- Start a job with `POST /research`.
- Follow progress live with `GET /research/{id}/stream` (server-sent events).
- Fetch the report with `GET /research/{id}`.

**Can I use its tools from another MCP client?**
Yes. The four tools (web search, page fetch, document search, notes) are an MCP server.
Run `python -m scout.tools.server` to use them over stdio, for example from Claude
Desktop.

**Can I use a different model or provider?**
- **Which model each step uses** is set in `config.yaml`.
- **Every model call goes through one client** (`src/scout/llm/client.py`), so a router
  or another provider can be added in one place.
- **`llm.base_url`** points the client at an Anthropic-compatible endpoint.

## Development

**How is it tested?**
- **The logic,** API, UI and attack suite are tested with fake models, on real Postgres
  with pgvector in CI.
- **A Docker Compose smoke test** checks the whole stack end to end.
- **The model's actual behaviour** is measured by a separate evaluation run on real
  models. It's started manually, because it costs money.

**How do I run the evaluation?**
`python -m scout.eval run --tags ci` runs 5 cases (about £0.17); `python -m scout.eval run`
runs all 50 (about £1.65). It's also on GitHub under **Actions → eval**.

**What doesn't it do?**
Multi-user collaboration, actions that change things (sending email, posting), browser
automation, voice, and mobile apps are out of scope.
