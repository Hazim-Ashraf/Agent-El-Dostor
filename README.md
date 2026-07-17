# ⚖️ Agent El-Dostor

An AI **agent** (not a workflow) that **reviews and generates** contracts under
**Egyptian law**, in **Arabic and English**.

- **Review** — upload a contract and ask questions; the agent autonomously reads the
  **contract** and searches a knowledge base of **Egyptian legislation** with tool
  calls, then answers with **citations** — comparing what the contract says against
  what the law requires.
- **Generate** — describe the terms you want; the agent researches the relevant
  **Egyptian Civil Code** articles, drafts the contract clause-by-clause (bilingual
  or single-language), **cites the article each legal clause is grounded in**, and
  exports a correctly-formatted **PDF** with proper Arabic (RTL) typesetting.

It never invents law: if the knowledge base doesn't cover something, it says so, and
every cited article is checked against the real corpus.

> **Not legal advice.** General legal information only; it does not replace a licensed
> Egyptian lawyer.

Everything runs in **Docker**. The only paid dependency is **OpenRouter** (the
reasoning LLM). Embeddings, the vector database, OCR, and the GUI are all free and
open-source.

---

## What's here today (Milestones M0–M5)

- A goal-driven, tool-calling **agent loop** over OpenRouter, working over both the
  **uploaded contract** and the **legislation** KB (`search_contract`,
  `get_contract_clause`, `search_legislation`, `get_legal_article`).
- A **hard verification gate**: the agent must finish via `submit_answer` with
  structured findings, and every finding's citation is checked against the real
  retrieved text (verbatim-quote match) — ungrounded claims are rejected and sent
  back for revision. Retrieved content is wrapped as untrusted data (prompt-injection
  defense).
- **Contract ingestion**: PDF / DOCX / TXT / image, with **OCR** (Tesseract ara+eng)
  for scanned files, clause segmentation, and bilingual embeddings.
- **Postgres + pgvector** store for legislation and contract clauses.
- A **full Streamlit GUI**: upload a contract, chat, inspect the tool-call trace and
  citations, and watch a **token + USD usage** panel.
- A legislation ingestion CLI + a small **bilingual (AR + EN) sample law**, and a
  **sample employment contract** for testing.
- **M4 hardening**: an **eval harness** (groundedness / citation-recall / refusal
  metrics), **run tracing** (JSONL with tokens / cost / latency), a query-embedding
  **cache**, an optional **FastAPI** service, and a **real-legislation PDF ingester**
  that parses the **Egyptian Civil Code (Law 131/1948)** into ~1,149 articles (Arabic,
  by the `مادة N` marker, with OCR fallback).
- **M5 contract generation → PDF**: a second agent loop that drafts a contract
  grounded in the ingested legislation (same discipline — it researches with
  `search_legislation`, attaches a `legal_basis` to each legal clause, and a
  verification step **drops any cited article that doesn't resolve**). It exports a
  formatted **PDF** via **WeasyPrint** (Amiri Arabic font) with correct **RTL Arabic**
  + LTR English. Bilingual PDFs support three **user-selectable** layouts:
  **side-by-side** (EN | AR), **sequential** (EN then AR), or **two separate PDFs**.

> ℹ️ The **Egyptian Civil Code (Law 131/1948)** is the **real, authoritative** law
> source (ingested from its PDF — step 3a). The one remaining `seed_labor_law.json` and
> the sample contract are still **SAMPLE** placeholders (the Labour Law is a separate
> statute not yet supplied) — see [data/legislation/SOURCING.md](data/legislation/SOURCING.md).

---

## Prerequisites

- **Docker Desktop** (Apple Silicon build for M1). Give it ~6–8 GB RAM in
  Settings → Resources.
- An **OpenRouter API key**: <https://openrouter.ai/keys>.

> **M1 / Docker note:** containers on macOS get no Metal/MPS, so the local embedding
> model + OCR run **CPU-only**. Defaults are chosen to stay light on 16 GB.

---

## Run it

```bash
# 1. Configure
cp .env.example .env       # then set OPENROUTER_API_KEY=... in .env

# 2. Build & start the database + app
docker compose up --build

# 3a. Load the real Egyptian Civil Code (Law 131/1948).
#     Put the PDF at data/legislation/law-131-1948.pdf first, then:
docker compose run --rm app python -m app.ingestion.legislation \
  --pdf data/legislation/law-131-1948.pdf --ocr

# 3b. (optional) Also load the SAMPLE labour-law placeholder for employment demos:
docker compose run --rm app python -m app.ingestion.legislation \
  --seed data/legislation/seed_labor_law.json

# 4. Open the GUI
#    http://localhost:8501
```

The GUI has two modes (sidebar radio):

**📄 Review a contract**
1. Under **Contract**, upload a file (or the bundled
   `data/contracts/sample_employment_contract_en.txt`) and click **Ingest contract**.
   You can also ingest it from the CLI:
   ```bash
   docker compose run --rm app python -m app.ingestion.contracts \
     --file data/contracts/sample_employment_contract_en.txt --type employment
   ```
2. In **Ask**, try: *"Is the probation period in my contract legal?"* — the agent
   reads Clause 2 (six months) and compares it to the law (three months), with
   citations.

**🖊️ Generate a contract**
1. Pick a **contract type** and **language** (Arabic / English / Bilingual), then
   **describe the terms** (parties, amounts, duration, special conditions).
2. Click **Generate contract** — the agent researches the Civil Code and drafts the
   clauses; the preview shows each clause and the article it's grounded in.
3. Under **Export PDF**, pick the bilingual **layout** (side-by-side / sequential /
   two files), click **Build PDF**, and **download** it.

Watch the **Usage** panel (sidebar) tally tokens and USD across both modes.

---

## Configuration (`.env`)

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | **Required.** The only paid dependency. | — |
| `REASONING_MODEL` | Any **tool-calling** model on OpenRouter. Free options (rate-limited): [list](https://openrouter.ai/models?supported_parameters=tools&max_price=0). | `deepseek/deepseek-chat-v3-0324` |
| `EMBEDDING_MODEL` | Free local model. `BAAI/bge-m3` for higher quality (bigger). | `intfloat/multilingual-e5-base` |
| `ENABLE_RERANK` | Reranking (better precision, more RAM). | `false` |
| `MAX_AGENT_ITERATIONS` | Review tool-call loop bound. | `6` |
| `MAX_GENERATION_ITERATIONS` | Generation tool-call loop bound. | `8` |
| `GENERATION_MAX_TOKENS` | Output budget for a generated contract. | `8000` |
| `DATABASE_URL` | Set automatically by Docker Compose. | `…@db:5432/eldostor` |

Change a value, then restart: `docker compose down && docker compose up`.

---

## Usage / cost tracking

Every OpenRouter call is asked to report cost (`usage: {include: true}`). The agent
accumulates **prompt / completion / total tokens** and **USD cost** across all model
calls in a run; the GUI shows per-answer usage (in each Trace) and a session running
total in the sidebar (with a reset button). If a model/provider doesn't report cost,
the panel shows tokens only and flags that cost was unavailable.

---

## Evaluation (M4)

A golden set of question/expected-citation cases measures the agent's quality:

```bash
docker compose run --rm app python eval/run_eval.py
```

It ingests the seeds + sample contract, runs each case through the agent, and reports:
**verification pass rate** (are answers grounded?), **citation recall** (did it cite the
expected articles/clauses?), and **refusal accuracy** (does it refuse out-of-scope
questions instead of inventing law?). A full report is written to `logs/eval_report.json`.
Edit `eval/dataset/cases.json` to add cases. (Runs against OpenRouter, so it costs tokens.)

## REST API (M4, optional)

A FastAPI service exposes the same agents:

```bash
docker compose --profile api up          # http://localhost:8000/docs
# Review:   POST /contracts (multipart file + contract_type)  ·  POST /ask {question, contract_id?}
#           GET /contracts  ·  GET /health
# Generate: POST /generate      {contract_type, language, brief}          -> structured contract + verification
#           POST /generate/pdf  {contract_type, language, brief, layout}  -> application/pdf (zip if layout="separate")
```

## Observability (M4)

Every run appends one JSON line to `logs/agent_runs.jsonl` (run id, tools called,
verification status, tokens, USD cost, latency) and logs a one-line summary. Tail it:

```bash
docker compose exec app tail -f logs/agent_runs.jsonl
```

---

## Rebuilds & caching (when do I need `--build`?)

- **Code changes → no rebuild.** The source is bind-mounted (`./:/app` in
  `docker-compose.yml`), so `docker compose up` already runs your latest code (Streamlit
  even hot-reloads). You only need `--build` when **`pyproject.toml` or the `Dockerfile`**
  changes.
- **Rebuilds are incremental.** The image uses **BuildKit cache mounts** for the apt and
  pip download caches, so a rebuild re-downloads **only new packages** — the heavy stack
  (torch, sentence-transformers, …) is reused from cache, not fetched again. (Requires
  BuildKit, which is the default in Docker 23+ / Compose v2.)
- **Model weights download once.** The embedding model is cached in the `models` named
  volume, so it isn't re-downloaded across restarts.
- Force a clean rebuild (rarely needed): `docker compose build --no-cache`.

## Common tasks

```bash
# Ingest more legislation / another contract
docker compose run --rm app python -m app.ingestion.legislation --seed <path>
docker compose run --rm app python -m app.ingestion.contracts --file <path> --type <type>

# psql shell / logs
docker compose exec db psql -U eldostor -d eldostor
docker compose logs -f app

# Reset all data (drops the DB volume) then re-ingest
docker compose down -v && docker compose up
```

---

## Troubleshooting

- **"OPENROUTER_API_KEY is not set"** — create `.env` and fill the key in.
- **Sidebar KB is empty** — run legislation ingestion (step 3).
- **Model 404 / "unavailable for free"** — OpenRouter retired that free endpoint; set
  `REASONING_MODEL` to a current tool-calling model and restart.
- **First contract/question is slow** — the embedding model downloads once (cached in
  the `models` volume) and runs on CPU.
- **Changing `EMBEDDING_MODEL` dimension** — different vector sizes need a fresh table:
  `docker compose down -v` then re-ingest.
- **"PDF rendering failed" / WeasyPrint errors** — the PDF exporter needs the Pango/Cairo
  libraries and the Amiri font baked into the image. Rebuild after pulling M5:
  `docker compose up --build`.
- **Generated clauses show no legal basis** — the legislation KB is empty; run the
  Civil Code ingestion (step 3a) so clauses can be grounded and cited.

---

## Project layout

```
app/
  llm/client.py                              # OpenRouter via the openai SDK (+ usage.include)
  core/{config,logging,usage,trace}.py       # config, usage/cost, run tracing
  agent/{loop,tools,prompts,context,schema,verify}.py     # REVIEW agent + verification gate
  generation/{loop,tools,prompts,schema,verify,pdf}.py    # GENERATE agent + PDF export
  retrieval/{store,embeddings}.py            # pgvector + cached local embeddings
  ingestion/{legislation,contracts}.py       # ingestion CLIs
  api/main.py                                # optional FastAPI service (review + generate)
ui/streamlit_app.py                          # full GUI (review chat + generate/PDF + usage)
eval/{dataset/, run_eval.py}                 # golden cases + metrics harness
data/legislation/  data/contracts/           # real Civil Code PDF + SAMPLE labour seed + SOURCING.md
Dockerfile · docker-compose.yml · .env.example
```

Status: **M0–M5 delivered** — contract **review + generation (with PDF export)**,
running on the real Egyptian Civil Code (Law 131/1948). Remaining work to broaden
coverage: source the **Labour Law** and other statutes (same PDF ingester — see
[data/legislation/SOURCING.md](data/legislation/SOURCING.md)), plus scale/quality
options (reranking, larger embedding model, managed vector store).
