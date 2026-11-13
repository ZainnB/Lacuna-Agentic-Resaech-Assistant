"""Key/value caches for API and LLM responses.

Blueprint section 0: every response is cached so re-running anything never re-spends quota.
`MemoryCache` is for tests; `PostgresCache` is the real one. Keys come from `cache_key`, which
hashes the full request so that any change in parameters is a different entry.
"""

from __future__ import annotations

from typing import Protocol

import psycopg
from psycopg.types.json import Jsonb

from lacuna.util import stable_hash


def cache_key(namespace: str, *parts) -> str:
    return f"{namespace}:{stable_hash(*parts, n=32)}"


class Cache(Protocol):
    def get(self, key: str) -> dict | None: ...

    def set(self, key: str, namespace: str, value: dict) -> None: ...


class MemoryCache:
    def __init__(self) -> None:
        self.data: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> dict | None:
        v = self.data.get(key)
        if v is None:
            self.misses += 1
        else:
            self.hits += 1
        return v

    def set(self, key: str, namespace: str, value: dict) -> None:
        self.data[key] = value


class PostgresCache:
    """Backed by `api_cache` or `llm_cache` (same shape; see lacuna.db.SCHEMA)."""

    def __init__(self, conn: psycopg.Connection, table: str) -> None:
        if table not in ("api_cache", "llm_cache"):
            raise ValueError(table)
        self.conn = conn
        self.table = table

    def get(self, key: str) -> dict | None:
        row = self.conn.execute(f"SELECT value FROM {self.table} WHERE key = %s", (key,)).fetchone()
        return row["value"] if row else None

    def set(self, key: str, namespace: str, value: dict) -> None:
        self.conn.execute(
            f"INSERT INTO {self.table} (key, namespace, value) VALUES (%s, %s, %s) "
            "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, created_at = now()",
            (key, namespace, Jsonb(value)),
        )
