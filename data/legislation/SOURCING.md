# Sourcing the Egyptian legislation corpus

The `seed_*.json` files in this folder are **illustrative SAMPLE data** — paraphrased
placeholder text used to exercise ingestion, retrieval, and the verification gate.
**They are not authoritative law and must not be relied on.** This guide describes how
to replace them with sourced, verified legislation.

> ⚠️ Do not let an AI model *write* the legal text. Every article must be transcribed
> from an authoritative published source and then verified by a qualified reviewer.
> The agent only cites what is in this corpus, so corpus quality is the product.

## Authoritative sources

- **Official Gazette (الجريدة الرسمية)** — the primary source of enacted laws and
  amendments.
- **Al-Waqa'i al-Misriyya (الوقائع المصرية)** — decrees and regulations.
- **State Council / Majlis al-Dawla** and the **Ministry of Justice** legislative
  portals.
- Reputable consolidated legal databases, used only to locate the gazette citation —
  then verify against the gazette text itself.

Always record the **gazette issue/date** each article was published in.

## Scope (the five contract domains)

| Domain | Core laws to source |
| --- | --- |
| Employment | Labour Law (and its executive regulations) |
| Rental / tenancy | Civil Code lease provisions + special rent laws (old vs new rent) |
| Shareholder | Companies Law + its executive regulations |
| Service / commercial | Civil Code obligations + Commercial (Trade) Law |

## Legal currency (critical)

Egyptian law changes — articles get amended or repealed and replaced (e.g. a new
Labour Law superseding the previous one; 2025 changes to rent law). A citation to a
**repealed** article is a dangerous failure. Therefore every article record carries:

- `effective_date` — when the article (as worded) came into force.
- `repealed` — `true` once superseded; repealed rows are excluded from retrieval.
- `source` — the gazette citation / provenance string.

When a law is amended, add the **new** version as a new record (new `effective_date`)
and set `repealed: true` on the old one — never edit history in place.

## Record schema

Each file: `{ "law": ..., "source": ..., "articles": [ ... ] }`. Each article:

```json
{
  "article_ref": "Art-33",
  "lang": "ar",
  "title": "فترة الاختبار",
  "effective_date": "2003-04-07",
  "repealed": false,
  "text": "…verbatim article text…"
}
```

Provide both `ar` and `en` where an official/authoritative translation exists; the
Arabic gazette text is the controlling version. Keep `article_ref` stable and unique
per law so citations resolve.

## Ingesting

```bash
# one file
docker compose run --rm app python -m app.ingestion.legislation --seed data/legislation/<file>.json
# a whole directory (all *.json)
docker compose run --rm app python -m app.ingestion.legislation --dir data/legislation
```

Ingestion is an idempotent upsert keyed on `law::article_ref::lang`, so re-running
after edits updates in place.
