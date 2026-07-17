"""Legislation ingestion CLI.

Two sources:
  1. Structured JSON  (--seed <file> | --dir <folder of *.json>)  — the sample seeds.
  2. A legislation PDF (--pdf <file>) — parses the real law text into articles by the
     Arabic article marker "مادة N". Extraction uses pdfplumber, and falls back to
     OCR (Tesseract `ara`) when the PDF's text layer is poor (common for Arabic PDFs).

Ingest the real Egyptian Civil Code (place the PDF first):
    docker compose run --rm app python -m app.ingestion.legislation \
        --pdf data/legislation/law-131-1948.pdf --ocr
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from datetime import date

from app.core.logging import get_logger
from app.retrieval import embeddings, store

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Structured JSON ingestion (sample seeds)
# --------------------------------------------------------------------------- #


def _parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def ingest(path: str) -> int:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    law = data["law"]
    source = data.get("source")
    articles = data["articles"]
    if not articles:
        log.warning("No articles in %s", path)
        return 0

    conn = store.connect()
    try:
        dim = embeddings.dimension()
        store.init_schema(conn, dim)
        vectors = embeddings.embed_passages([a["text"] for a in articles])
        rows = [
            {
                "id": f"{law}::{a['article_ref']}::{a['lang']}",
                "law": law,
                "article_ref": a["article_ref"],
                "lang": a["lang"],
                "title": a.get("title"),
                "text": a["text"],
                "effective_date": _parse_date(a.get("effective_date")),
                "repealed": bool(a.get("repealed", False)),
                "source": source,
                "embedding": vector,
            }
            for a, vector in zip(articles, vectors)
        ]
        n = store.upsert_chunks(conn, rows)
        log.info("Ingested %s chunks from %s. Total in KB: %s", n, path, store.count(conn))
        return n
    finally:
        conn.close()


def ingest_dir(directory: str) -> int:
    paths = sorted(glob.glob(os.path.join(directory, "*.json")))
    if not paths:
        log.warning("No .json files found in %s", directory)
        return 0
    total = sum(ingest(p) for p in paths)
    log.info("Ingested %s chunks from %s file(s) in %s", total, len(paths), directory)
    return total


# --------------------------------------------------------------------------- #
# PDF ingestion (real legislation, e.g. the Egyptian Civil Code)
# --------------------------------------------------------------------------- #

# Matches an article HEADER "مادة N" but not an inline reference. In every inline
# form (المادة / للمادة / بالمادة …) the letter immediately before "مادة" is "ل";
# a header is preceded by whitespace/start. Handles Western + Arabic-Indic digits
# and an optional dash before the number.
_ARTICLE_RE = re.compile(r"(?<!ل)مادة\s*[-–]?\s*([0-9٠-٩]{1,4})")
_AR_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_NOISE = "القانون رقم 131 لسنة 1948 بإصدار القانون المدني"


def _pdf_text_pdfplumber(path: str) -> str:
    import pdfplumber

    parts: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def _pdf_text_ocr(path: str) -> str:
    # Page-by-page to keep memory low on 16 GB; reads rendered glyphs, so it
    # produces correct logically-ordered Arabic even when the text layer is bad.
    from pdf2image import convert_from_path, pdfinfo_from_path

    import pytesseract

    n_pages = int(pdfinfo_from_path(path)["Pages"])
    log.info("OCR (Tesseract ara) over %s pages — this takes a few minutes on CPU…", n_pages)
    parts: list[str] = []
    for i in range(1, n_pages + 1):
        images = convert_from_path(path, dpi=200, first_page=i, last_page=i)
        parts.append(pytesseract.image_to_string(images[0], lang="ara"))
        if i % 10 == 0:
            log.info("  …OCR'd %s/%s pages", i, n_pages)
    return "\n".join(parts)


def extract_pdf_text(path: str, use_ocr: bool | None = None) -> str:
    """use_ocr: True=force OCR, False=pdfplumber only, None=auto (OCR fallback)."""
    if use_ocr is True:
        return _pdf_text_ocr(path)
    text = _pdf_text_pdfplumber(path)
    n_markers = len(_ARTICLE_RE.findall(text))
    if use_ocr is None and n_markers < 50:
        log.warning(
            "pdfplumber found only %s article markers (Arabic text layer likely broken); "
            "falling back to OCR.",
            n_markers,
        )
        return _pdf_text_ocr(path)
    log.info("pdfplumber extraction: %s article markers found.", n_markers)
    return text


def parse_pdf_articles(text: str, lang: str = "ar") -> list[dict]:
    text = text.replace(_NOISE, " ")
    matches = list(_ARTICLE_RE.finditer(text))
    best: dict[str, str] = {}
    for i, m in enumerate(matches):
        num = m.group(1).translate(_AR_DIGITS)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = re.sub(r"\s+", " ", text[start:end]).strip()
        if len(body) < 25:  # inline reference / heading noise, not a real article
            continue
        if len(body) > 6000:  # guard against a merged/runaway body
            body = body[:6000]
        # Keep the longest body seen for each article number.
        if num not in best or len(body) > len(best[num]):
            best[num] = body

    return [
        {
            "article_ref": f"مادة {num}",
            "lang": lang,
            "title": None,
            "text": body,
        }
        for num, body in sorted(best.items(), key=lambda kv: int(kv[0]))
    ]


def ingest_pdf(
    path: str,
    law: str,
    source: str,
    effective_date: str | None,
    lang: str = "ar",
    use_ocr: bool | None = None,
) -> int:
    text = extract_pdf_text(path, use_ocr)
    articles = parse_pdf_articles(text, lang=lang)
    if not articles:
        log.warning("No articles parsed from %s", path)
        return 0
    log.info("Parsed %s articles from %s; embedding…", len(articles), path)

    conn = store.connect()
    try:
        dim = embeddings.dimension()
        store.init_schema(conn, dim)
        vectors = embeddings.embed_passages([a["text"] for a in articles])
        eff = _parse_date(effective_date)
        rows = [
            {
                "id": f"{law}::{a['article_ref']}::{a['lang']}",
                "law": law,
                "article_ref": a["article_ref"],
                "lang": a["lang"],
                "title": a["title"],
                "text": a["text"],
                "effective_date": eff,
                "repealed": False,
                "source": source,
                "embedding": vector,
            }
            for a, vector in zip(articles, vectors)
        ]
        n = store.upsert_chunks(conn, rows)
        log.info("Ingested %s articles from %s. Total in KB: %s", n, path, store.count(conn))
        return n
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Egyptian legislation (JSON or PDF).")
    parser.add_argument("--seed", help="Path to a single legislation JSON file.")
    parser.add_argument("--dir", help="Directory to ingest (all *.json).")
    parser.add_argument("--pdf", help="Path to a legislation PDF to parse into articles.")
    parser.add_argument("--ocr", action="store_true", help="Force OCR extraction (recommended for Arabic).")
    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR fallback (pdfplumber only).")
    parser.add_argument("--law", default="Egyptian Civil Code (Law 131 of 1948)")
    parser.add_argument(
        "--source",
        default="Egyptian Civil Code — Law No. 131 of 1948 (published in the Official Gazette; in force 15 Oct 1949).",
    )
    parser.add_argument("--effective-date", default="1949-10-15")
    parser.add_argument("--lang", default="ar")
    args = parser.parse_args()

    if args.pdf:
        use_ocr = True if args.ocr else (False if args.no_ocr else None)
        ingest_pdf(args.pdf, args.law, args.source, args.effective_date, args.lang, use_ocr)
    elif args.dir:
        ingest_dir(args.dir)
    else:
        ingest(args.seed or "data/legislation/seed_labor_law.json")


if __name__ == "__main__":
    main()
