# ⚖️ Agent El-Dostor

An AI **agent** (not a workflow) that helps people in Egypt understand the legal
implications of their contracts. You upload a contract and ask questions; the agent
autonomously reads the **contract** and searches a knowledge base of **Egyptian
legislation** with tool calls, then answers with **citations** — comparing what the
contract says against what the law requires. It never invents law: if the knowledge
base doesn't cover something, it says so.

> **Not legal advice.** General legal information only; it does not replace a licensed
> Egyptian lawyer.

Everything runs in **Docker**. The only paid dependency is **OpenRouter** (the
reasoning LLM). Embeddings, the vector database, OCR, and the GUI are all free and
open-source.

---

## What's here today (Milestones M0–M2)

- A goal-driven, tool-calling **agent loop** over OpenRouter, working over both the
  **uploaded contract** and the **legislation** KB (`search_contract`,
  `get_contract_clause`, `search_legislation`, `get_legal_article`).
- **Contract ingestion**: PDF / DOCX / TXT / image, with **OCR** (Tesseract ara+eng)
  for scanned files, clause segmentation, and bilingual embeddings.
- **Postgres + pgvector** store for legislation and contract clauses.
- A **full Streamlit GUI**: upload a contract, chat, inspect the tool-call trace and
  citations, and watch a **token + USD usage** panel.
- A legislation ingestion CLI + a small **bilingual (AR + EN) sample law**, and a
  **sample employment contract** for testing.

> ℹ️ The seed legislation and sample contract in `data/` are clearly labelled
> **SAMPLE** placeholder text — not authoritative law. Sourcing the real corpus is a
> later milestone.

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

# 3. Load the sample legislation (creates the KB tables)
docker compose run --rm app python -m app.ingestion.legislation \
  --seed data/legislation/seed_labor_law.json

# 4. Open the GUI
#    http://localhost:8501
```

In the GUI:
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
3. Watch the **Usage** panel (sidebar) tally tokens and USD as you go.

---

## Configuration (`.env`)

| Variable | Purpose | Default |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | **Required.** The only paid dependency. | — |
| `REASONING_MODEL` | Any **tool-calling** model on OpenRouter. Free options (rate-limited): [list](https://openrouter.ai/models?supported_parameters=tools&max_price=0). | `deepseek/deepseek-chat-v3-0324` |
| `EMBEDDING_MODEL` | Free local model. `BAAI/bge-m3` for higher quality (bigger). | `intfloat/multilingual-e5-base` |
| `ENABLE_RERANK` | Reranking (better precision, more RAM). | `false` |
| `MAX_AGENT_ITERATIONS` | Tool-call loop bound. | `6` |
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

---

## Project layout

```
app/
  llm/client.py              # OpenRouter via the openai SDK (+ usage.include)
  core/{config,logging,usage}.py
  agent/{loop,tools,prompts,context}.py   # the tool-calling agent
  retrieval/{store,embeddings}.py         # pgvector + local embeddings
  ingestion/{legislation,contracts}.py    # ingestion CLIs
ui/streamlit_app.py          # full GUI (upload + chat + usage)
data/legislation/  data/contracts/        # SAMPLE corpus + sample contract
Dockerfile · docker-compose.yml · .env.example
```

Roadmap: **M3** full agent + `submit_answer` + hard verification gate; **M4**
hardening (eval harness, prefix caching, optional FastAPI) + real sourced corpus.
