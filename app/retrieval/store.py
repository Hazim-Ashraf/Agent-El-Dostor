"""Postgres + pgvector storage for legislation (and, later, contract) chunks."""
from __future__ import annotations

import time
from typing import Any

import numpy as np

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


def table_exists(conn: psycopg.Connection, name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s) AS t", (name,))
        row = cur.fetchone()
        return bool(row and row["t"] is not None)


def search(
    conn: psycopg.Connection,
    query_embedding: list[float],
    law: str | None = None,
    as_of_date: str | None = None,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Cosine-distance nearest neighbours over in-force articles."""
    if not table_exists(conn, "legislation_chunks"):
        return []
    sql = [
        "SELECT id, law, article_ref, lang, title, text, effective_date,",
        "       (embedding <=> %s) AS distance",
        "FROM legislation_chunks",
        "WHERE repealed = FALSE",
    ]
    vec = np.array(query_embedding, dtype=np.float32)
    params: list[Any] = [vec]
    if law:
        sql.append("AND law = %s")
        params.append(law)
    if as_of_date:
        sql.append("AND (effective_date IS NULL OR effective_date <= %s)")
        params.append(as_of_date)
    sql.append("ORDER BY embedding <=> %s")
    params.append(vec)
    sql.append("LIMIT %s")
    params.append(limit)

    with conn.cursor() as cur:
        cur.execute("\n".join(sql), params)
        return cur.fetchall()


def get_article(
    conn: psycopg.Connection, law: str, article_ref: str
) -> list[dict[str, Any]]:
    if not table_exists(conn, "legislation_chunks"):
        return []
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


def find_articles_by_ref(conn: psycopg.Connection, article_ref: str) -> list[dict[str, Any]]:
    """Resolve an article by its ref across any law (lenient citation lookup)."""
    if not table_exists(conn, "legislation_chunks"):
        return []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, law, article_ref, lang, title, text FROM legislation_chunks "
            "WHERE article_ref = %s AND repealed = FALSE ORDER BY lang",
            (article_ref,),
        )
        return cur.fetchall()


def count(conn: psycopg.Connection) -> int:
    if not table_exists(conn, "legislation_chunks"):
        return 0
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) AS n FROM legislation_chunks")
        row = cur.fetchone()
        return int(row["n"]) if row else 0


# --------------------------------------------------------------------------- #
# Contracts (M1)
# --------------------------------------------------------------------------- #


def init_contract_schema(conn: psycopg.Connection, dim: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS contracts (
                id            TEXT PRIMARY KEY,
                filename      TEXT,
                contract_type TEXT,
                lang          TEXT,
                n_clauses     INTEGER NOT NULL DEFAULT 0,
                uploaded_at   TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS contract_clauses (
                id           TEXT PRIMARY KEY,
                contract_id  TEXT NOT NULL REFERENCES contracts(id) ON DELETE CASCADE,
                clause_index INTEGER NOT NULL,
                clause_ref   TEXT,
                lang         TEXT,
                text         TEXT NOT NULL,
                embedding    vector({dim})
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_clause_contract ON contract_clauses (contract_id)"
        )
    conn.commit()


def upsert_contract(conn: psycopg.Connection, contract: dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO contracts (id, filename, contract_type, lang, n_clauses)
            VALUES (%(id)s, %(filename)s, %(contract_type)s, %(lang)s, %(n_clauses)s)
            ON CONFLICT (id) DO UPDATE SET
                filename = EXCLUDED.filename,
                contract_type = EXCLUDED.contract_type,
                lang = EXCLUDED.lang,
                n_clauses = EXCLUDED.n_clauses
            """,
            contract,
        )
    conn.commit()


def replace_contract_clauses(
    conn: psycopg.Connection, contract_id: str, rows: list[dict[str, Any]]
) -> int:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM contract_clauses WHERE contract_id = %s", (contract_id,))
        for r in rows:
            cur.execute(
                """
                INSERT INTO contract_clauses
                    (id, contract_id, clause_index, clause_ref, lang, text, embedding)
                VALUES
                    (%(id)s, %(contract_id)s, %(clause_index)s, %(clause_ref)s,
                     %(lang)s, %(text)s, %(embedding)s)
                """,
                r,
            )
    conn.commit()
    return len(rows)


def search_contract(
    conn: psycopg.Connection,
    contract_id: str,
    query_embedding: list[float],
    limit: int = 6,
) -> list[dict[str, Any]]:
    if not table_exists(conn, "contract_clauses"):
        return []
    vec = np.array(query_embedding, dtype=np.float32)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, clause_index, clause_ref, lang, text,
                   (embedding <=> %s) AS distance
            FROM contract_clauses
            WHERE contract_id = %s
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (vec, contract_id, vec, limit),
        )
        return cur.fetchall()


def get_clause(
    conn: psycopg.Connection, contract_id: str, clause_ref: str
) -> list[dict[str, Any]]:
    if not table_exists(conn, "contract_clauses"):
        return []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, clause_index, clause_ref, lang, text
            FROM contract_clauses
            WHERE contract_id = %s
              AND (clause_ref ILIKE %s OR clause_index::text = %s)
            ORDER BY clause_index
            """,
            (contract_id, f"%{clause_ref}%", clause_ref),
        )
        return cur.fetchall()


def get_contract(conn: psycopg.Connection, contract_id: str) -> dict[str, Any] | None:
    if not table_exists(conn, "contracts"):
        return None
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM contracts WHERE id = %s", (contract_id,))
        return cur.fetchone()


def list_contracts(conn: psycopg.Connection) -> list[dict[str, Any]]:
    if not table_exists(conn, "contracts"):
        return []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, filename, contract_type, lang, n_clauses, uploaded_at "
            "FROM contracts ORDER BY uploaded_at DESC"
        )
        return cur.fetchall()


def delete_contract(conn: psycopg.Connection, contract_id: str) -> None:
    if not table_exists(conn, "contracts"):
        return
    with conn.cursor() as cur:
        cur.execute("DELETE FROM contracts WHERE id = %s", (contract_id,))
    conn.commit()
