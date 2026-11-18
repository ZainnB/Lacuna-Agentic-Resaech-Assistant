"""Paper-API clients. `make_clients(conn=conn)` is the entry point for scout and evals."""

from __future__ import annotations

from dataclasses import dataclass

from lacuna.cache import Cache, MemoryCache, PostgresCache
from lacuna.ingest.arxiv import ArxivClient, normalise_arxiv_id
from lacuna.ingest.http import BudgetExceeded, CachedClient
from lacuna.ingest.openalex import OpenAlexClient
from lacuna.ingest.resolve import resolve_hits
from lacuna.ingest.semantic_scholar import S2Client
from lacuna.settings import Settings, get_settings

__all__ = [
    "ArxivClient", "BudgetExceeded", "CachedClient", "Clients", "OpenAlexClient", "S2Client",
    "make_clients", "normalise_arxiv_id", "resolve_hits",
]  # fmt: skip


@dataclass
class Clients:
    openalex: OpenAlexClient
    s2: S2Client
    arxiv: ArxivClient

    def close(self) -> None:
        for c in (self.openalex, self.s2, self.arxiv):
            c.http.close()


def make_clients(settings: Settings | None = None, conn=None, cache: Cache | None = None) -> Clients:
    settings = settings or get_settings()
    if cache is None:
        cache = PostgresCache(conn, "api_cache") if conn is not None else MemoryCache()
    return Clients(
        openalex=OpenAlexClient(settings, cache, conn),
        s2=S2Client(settings, cache, conn),
        arxiv=ArxivClient(settings, cache, conn),
    )
