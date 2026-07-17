# Sourcing the Egyptian legislation corpus

## Current corpus status

- ✅ **Egyptian Civil Code — Law No. 131 of 1948** (real, authoritative). Ingested from
  the official PDF (`law-131-1948.pdf`) via the PDF parser (see below). This covers the
  Civil Code's ~1,149 articles: obligations, contracts (sale, **lease/rent**, company,
  agency, deposit, insurance, suretyship), property, and real securities.
- ⚠️ **Labour** — `seed_labor_law.json` is still **illustrative SAMPLE data** (its `source`
  field says so). The Egyptian Labour Law is a *separate* statute; replace this placeholder
  once you have the authoritative Labour Law text.

The remaining `seed_*.json` file is a **SAMPLE** placeholder used to exercise the pipeline.
**Sample data is not authoritative law and must not be relied on.** This guide describes
how to source and ingest real legislation.

## Ingesting a real legislation PDF

Place the PDF in this folder, then:

```bash
docker compose run --rm app python -m app.ingestion.legislation \
  --pdf data/legislation/law-131-1948.pdf --ocr \
  --law "Egyptian Civil Code (Law 131 of 1948)" --effective-date 1949-10-15
```

The parser splits the text into articles on the Arabic header marker `مادة N`, embeds
each article, and upserts it into the same `legislation_chunks` table. `--ocr` uses
Tesseract (`ara`) on rasterised pages — recommended for Arabic PDFs whose text layer is
broken; omit it to try the faster `pdfplumber` text layer (the parser auto-falls back to
OCR if it finds too few article markers).

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
