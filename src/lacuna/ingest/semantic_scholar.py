"""Semantic Scholar Graph API client. Results are `SearchHit`s, never `PaperRecord`s.

Observed 2026-11-13: an unauthenticated request already got HTTP 429 (shared pool), so expect to
rely on the retry path or set S2_API_KEY.
"""

from __future__ import annotations

import httpx

from lacuna.cache import Cache
from lacuna.ingest.http import CachedClient, user_agent
from lacuna.models import SearchHit
from lacuna.ratelimit import RateLimiter
from lacuna.settings import Settings

BASE = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,year,externalIds,citationCount"
MAX_BATCH = 500
MAX_PAGE = 100


def to_hit(p: dict | None, score: float | None = None) -> SearchHit | None:
    if not p or not p.get("paperId"):
        return None
    ext = p.get("externalIds") or {}
    doi = (ext.get("DOI") or "").lower() or None
    return SearchHit(
        source="s2",
        source_id=p["paperId"],
        title=p.get("title") or "",
        doi=doi,
        arxiv_id=ext.get("ArXiv"),
        year=p.get("year"),
        citation_count=p.get("citationCount"),
        score=score,
    )


class S2Client:
    def __init__(
        self,
        settings: Settings,
        cache: Cache,
        conn=None,
        *,
        limiter: RateLimiter | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep=None,
    ) -> None:
        headers = {"User-Agent": user_agent(settings.contact_email)}
        if settings.s2_api_key:
            headers["x-api-key"] = settings.s2_api_key
        rate = 1.0 if settings.s2_api_key else 1 / 3
        kw = {"sleep": sleep} if sleep else {}
        self.http = CachedClient(
            "s2", cache, limiter or RateLimiter(rate), conn, headers, transport=transport, **kw
        )

    def search(self, query: str, limit: int = 20, year: str | None = None) -> list[SearchHit]:
        out: list[SearchHit] = []
        offset = 0
        while len(out) < limit:
            n = min(MAX_PAGE, limit - len(out))
            params = {"query": query, "limit": n, "offset": offset, "fields": FIELDS}
            if year:
                params["year"] = year
            data = self.http.get(f"{BASE}/paper/search", params)
            rows = data.get("data") or []
            out += [h for p in rows if (h := to_hit(p))]
            if len(rows) < n or "next" not in data:
                break
            offset = data["next"]
        return out[:limit]

    def paper(self, paper_id: str) -> SearchHit | None:
        """`paper_id` may be an S2 id, "DOI:...", "ARXIV:..."."""
        try:
            return to_hit(self.http.get(f"{BASE}/paper/{paper_id}", {"fields": FIELDS}))
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise

    def _edges(self, paper_id: str, kind: str, key: str, limit: int) -> list[SearchHit]:
        fields = ",".join(f"{key}.{f}" for f in FIELDS.split(","))
        data = self.http.get(f"{BASE}/paper/{paper_id}/{kind}", {"fields": fields, "limit": min(limit, 1000)})
        return [h for r in data.get("data") or [] if (h := to_hit(r.get(key)))][:limit]

    def references(self, paper_id: str, limit: int = 100) -> list[SearchHit]:
        return self._edges(paper_id, "references", "citedPaper", limit)

    def citations(self, paper_id: str, limit: int = 100) -> list[SearchHit]:
        return self._edges(paper_id, "citations", "citingPaper", limit)

    def batch(self, ids: list[str]) -> list[SearchHit]:
        out: list[SearchHit] = []
        for i in range(0, len(ids), MAX_BATCH):
            data = self.http.post(
                f"{BASE}/paper/batch", {"fields": FIELDS}, json={"ids": ids[i : i + MAX_BATCH]}
            )
            rows = data["_json"] if "_json" in data else data
            out += [h for p in rows if (h := to_hit(p))]
        return out
