"""The chunk table in Postgres (`chunks`, `index_meta`; schema in sql/init.sql)."""

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from lacuna.deepread.config import ROOT
from lacuna.settings import get_settings


def connect(dsn: str | None = None) -> psycopg.Connection:
    """Connect to the shared Lacuna database and make sure the chunk tables exist."""
    conn = psycopg.connect(dsn or get_settings().postgres_dsn, autocommit=True)
    conn.execute((ROOT / "sql" / "init.sql").read_text())
    register_vector(conn)
    return conn


def load(conn: psycopg.Connection, chunks: list[dict], vectors: np.ndarray, index_hash: str) -> None:
    """Replace the chunk table with this chunk set and record which config built it."""
    with conn.transaction():
        conn.execute("TRUNCATE chunks")
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (id, paper_id, page, section, text, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                [
                    (c["id"], c["paper_id"], c["page"], c["section"], c["text"], v)
                    for c, v in zip(chunks, vectors, strict=True)
                ],
            )
        conn.execute(
            "INSERT INTO index_meta (key, value) VALUES ('index_hash', %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
            (index_hash,),
        )


def stored_index_hash(conn: psycopg.Connection) -> str | None:
    row = conn.execute("SELECT value FROM index_meta WHERE key = 'index_hash'").fetchone()
    return row[0] if row else None
