"""Postgres + pgvector storage for legislation (and, later, contract) chunks."""
from __future__ import annotations

import time
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from app.core.config import settings
from app.core.logging import get_logger

log = get_logger(__name__)


def connect(retries: int = 30, delay: float = 2.0) -> psycopg.Connection:
    """Connect to Postgres, ensure the vector extension, register the adapter.

    Retries so the app can start alongside the db container.
    """
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            conn = psycopg.connect(settings.database_url, row_factory=dict_row)
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.commit()
            register_vector(conn)
            return conn
        except Exception as err:  # noqa: BLE001 - retry on any connection error
            last_err = err
            log.warning("DB not ready (attempt %s/%s): %s", attempt, retries, err)
            time.sleep(delay)
    raise RuntimeError(f"Could not connect to Postgres: {last_err}")


def init_schema(conn: psycopg.Connection, dim: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS legislation_chunks (
                id             TEXT PRIMARY KEY,
                law            TEXT NOT NULL,
                article_ref    TEXT NOT NULL,
                lang           TEXT NOT NULL,
                title          TEXT,
                text           TEXT NOT NULL,
                effective_date DATE,
                repealed       BOOLEAN NOT NULL DEFAULT FALSE,
                source         TEXT,
                embedding      vector({dim})
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_leg_law ON legislation_chunks (law)")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_leg_article ON legislation_chunks (law, article_ref)"
        )
    conn.commit()


def upsert_chunks(conn: psycopg.Connection, rows: list[dict[str, Any]]) -> int:
    with conn.cursor() as cur:
        for r in rows:
            cur.execute(
                """
                INSERT INTO legislation_chunks
                    (id, law, article_ref, lang, title, text, effective_date, repealed, source, embedding)
                VALUES
                    (%(id)s, %(law)s, %(article_ref)s, %(lang)s, %(title)s, %(text)s,
                     %(effective_date)s, %(repealed)s, %(source)s, %(embedding)s)
                ON CONFLICT (id) DO UPDATE SET
                    text = EXCLUDED.text,
                    title = EXCLUDED.title,
                    effective_date = EXCLUDED.effective_date,
                    repealed = EXCLUDED.repealed,
                    source = EXCLUDED.source,
                    embedding = EXCLUDED.embedding
                """,
                r,
            )
    conn.commit()
    return len(rows)


def search(
    conn: psycopg.Connection,
    query_embedding: list[float],
    law: str | None = None,
    as_of_date: str | None = None,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Cosine-distance nearest neighbours over in-force articles."""
    sql = [
        "SELECT id, law, article_ref, lang, title, text, effective_date,",
        "       (embedding <=> %s) AS distance",
        "FROM legislation_chunks",
        "WHERE repealed = FALSE",
    ]
    params: list[Any] = [query_embedding]
    if law:
        sql.append("AND law = %s")
        params.append(law)
    if as_of_date:
        sql.append("AND (effective_date IS NULL OR effective_date <= %s)")
        params.append(as_of_date)
    sql.append("ORDER BY embedding <=> %s")
    params.append(query_embedding)
    sql.append("LIMIT %s")
    params.append(limit)

    with conn.cursor() as cur:
        cur.execute("\n".join(sql), params)
        return cur.fetchall()


def get_article(
    conn: psycopg.Connection, law: str, article_ref: str
) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, law, article_ref, lang, title, text, effective_date, repealed, source
            FROM legislation_chunks
            WHERE law = %s AND article_ref = %s
            ORDER BY lang
            """,
            (law, article_ref),
        )
        return cur.fetchall()


def count(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM legislation_chunks")
        row = cur.fetchone()
        return int(row["n"]) if row else 0
