# ⚖️ Agent El-Dostor

An AI **agent** (not a workflow) that helps people in Egypt understand the legal
implications of their contracts. You ask a question; the agent autonomously
searches a knowledge base of **Egyptian legislation** (and, from a later milestone,
your uploaded contract) with tool calls, and answers with **citations** to the
articles it relied on. It never invents law — if the knowledge base doesn't cover
something, it says so.

> **Not legal advice.** This is general legal information and does not replace a
> licensed Egyptian lawyer.

Everything runs in **Docker**. The only paid dependency is **OpenRouter** (the
reasoning LLM). Embeddings, the vector database, OCR, and the GUI are all free and
open-source.

---

## What's here today (Milestone M0 — Dockerized walking skeleton)

- A goal-driven, tool-calling **agent loop** over OpenRouter (`search_legislation`,
  `get_legal_article`).
- **Postgres + pgvector** store and a **local multilingual embedding model**.
- A legislation **ingestion CLI** and a small **bilingual (AR + EN) sample law**.
- A minimal **Streamlit** GUI to ask questions and inspect the tool-call trace.

> ℹ️ The seed legislation in `data/legislation/seed_labor_law.json` is clearly
> labelled **SAMPLE** placeholder text for testing the pipeline — it is **not**
> authoritative law. Sourcing the real corpus is a later milestone.

---

## Prerequisites

- **Docker Desktop** (Apple Silicon build for M1). Nothing else — no local Python,
  Postgres, or model downloads on the host.
- An **OpenRouter API key**: <https://openrouter.ai/keys>.

> **M1 / Docker note:** containers on macOS don't get Metal/MPS, so the local
> embedding model runs **CPU-only** inside the container. The default model
> (`intfloat/multilingual-e5-base`) is chosen to stay light on 16 GB RAM.

---

## Run it

```bash
# 1. Configure
cp .env.example .env
#   → open .env and set OPENROUTER_API_KEY=...

# 2. Build & start the database + app
docker compose up --build
#   Postgres (pgvector) + the Streamlit app start together.
#   First run downloads the embedding model into the `models` volume (once).

# 3. In another terminal, load the sample legislation into the knowledge base
docker compose run --rm app python -m app.ingestion.legislation \
  --seed data/legislation/seed_labor_law.json

# 4. Open the GUI
#   http://localhost:8501
```

Ask something the sample data covers, e.g.:

- "What is the maximum probation period for an employee?"
- "ما هو الحد الأقصى لساعات العمل الأسبوعية؟"
- "How much notice is required to terminate an indefinite contract?"

You'll get an answer with `(Law, article_ref)` citations, plus an expandable
**Trace** showing exactly which tools the agent chose to call.

Try an out-of-coverage question (e.g. about tax law) — the agent should say it
doesn't have that in its knowledge base rather than guess.

---

## Configuration (`.env`)

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | **Required.** The only paid dependency. | — |
| `REASONING_MODEL` | Any **tool-calling** model on OpenRouter (free default for testing). | `deepseek/deepseek-chat-v3-0324:free` |
| `EMBEDDING_MODEL` | Free local model. `BAAI/bge-m3` for higher quality (bigger). | `intfloat/multilingual-e5-base` |
| `ENABLE_RERANK` | Reranking (better precision, more RAM). Off by default on 16 GB. | `false` |
| `MAX_AGENT_ITERATIONS` | Tool-call loop bound. | `6` |
| `DATABASE_URL` | Set automatically by Docker Compose. | `…@db:5432/eldostor` |

Change a value in `.env`, then restart: `docker compose up`.

---

## Common tasks

```bash
# Re-run ingestion (idempotent upsert)
docker compose run --rm app python -m app.ingestion.legislation --seed <path>

# Open a psql shell
docker compose exec db psql -U eldostor -d eldostor

# Tail the app logs (tool calls, retrieval, timings)
docker compose logs -f app

# Stop everything
docker compose down
```

---

## Troubleshooting

- **"OPENROUTER_API_KEY is not set"** — you didn't create `.env` (or left the key
  blank). `cp .env.example .env` and fill it in.
- **Sidebar says the knowledge base is empty** — run the ingestion command (step 3).
- **First question is slow** — the embedding model downloads on first use and runs
  on CPU. It's cached in the `models` volume afterwards.
- **"Database not reachable"** — give the `db` container a few seconds on first
  boot; the app retries automatically.
- **Reset the database** — `docker compose down -v` removes the `pgdata` volume
  (this deletes all ingested data), then `docker compose up` and re-ingest.
- **Change the embedding model dimension** — switching `EMBEDDING_MODEL` between
  models with different vector sizes requires a fresh table: `docker compose down -v`
  then re-ingest.

---

## Project layout

```
app/
  llm/client.py            # OpenRouter via the openai SDK
  agent/{loop,tools,prompts}.py          # the tool-calling agent
  retrieval/{store,embeddings}.py        # pgvector + local embeddings
  ingestion/legislation.py               # ingestion CLI
  core/{config,logging}.py
ui/streamlit_app.py        # testing GUI
data/legislation/          # sample corpus (replace with sourced law later)
Dockerfile · docker-compose.yml · .env.example
```

Roadmap: M1 ingestion & retrieval → M2 GUI → M3 full agent + verification gate →
M4 hardening + real corpus.
