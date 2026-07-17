"""Legislation ingestion CLI (M0).

Reads a structured legislation JSON file, embeds each article, and upserts it
into the pgvector store with law / article_ref / language / effective-date /
repeal metadata.

Run inside the container:
    docker compose run --rm app python -m app.ingestion.legislation --seed data/legislation/seed_labor_law.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import date

from app.core.logging import get_logger
from app.retrieval import embeddings, store

log = get_logger(__name__)


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
        rows = []
        for article, vector in zip(articles, vectors):
            rows.append(
                {
                    "id": f"{law}::{article['article_ref']}::{article['lang']}",
                    "law": law,
                    "article_ref": article["article_ref"],
                    "lang": article["lang"],
                    "title": article.get("title"),
                    "text": article["text"],
                    "effective_date": _parse_date(article.get("effective_date")),
                    "repealed": bool(article.get("repealed", False)),
                    "source": source,
                    "embedding": vector,
                }
            )

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
    total = 0
    for path in paths:
        total += ingest(path)
    log.info("Ingested %s chunks from %s file(s) in %s", total, len(paths), directory)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Egyptian legislation JSON.")
    parser.add_argument("--seed", help="Path to a single legislation JSON file.")
    parser.add_argument("--dir", help="Directory to ingest (all *.json).")
    args = parser.parse_args()
    if args.dir:
        ingest_dir(args.dir)
    else:
        ingest(args.seed or "data/legislation/seed_labor_law.json")


if __name__ == "__main__":
    main()
