"""OpenAlex client. `paper_id` everywhere is the OpenAlex work ID ("W...").

Facts verified 2026-11-13 against help.openalex.org and live responses:
  * Key goes in the `api_key` query parameter (or `Authorization: Bearer`). Requests without a key
    still work but share a $0.10/day budget per IP (response `meta.notice`, `x-ratelimit-limit-usd`).
  * `per_page` max 200; a pipe-OR filter takes at most 100 values (101 -> HTTP 400).
  * Prices per call: single ID/DOI lookup $0, list/filter $0.0001, search $0.001 (PRICES below);
    each response reports its own cost in `meta.cost_usd` and `x-ratelimit-cost-usd`.
  * Budgets reset at midnight UTC.
"""

from __future__ import annotations

import re
from datetime import date

import httpx

from lacuna.cache import Cache
from lacuna.ingest.http import BudgetExceeded, CachedClient, user_agent
from lacuna.models import AuthorRecord, PaperRecord
from lacuna.ratelimit import RateLimiter
from lacuna.settings import Settings

__all__ = ["BudgetExceeded", "OpenAlexClient", "PRICES", "work_to_record", "normalise_work_id"]

BASE = "https://api.openalex.org"
# https://help.openalex.org (pricing), checked 2026-11-13. USD per call.
PRICES = {"lookup": 0.0, "list": 0.0001, "search": 0.001}
UNAUTH_DAILY_USD = 0.10  # shared per-IP allowance without a key
UNAUTH_BUDGET_USD = 0.09
MAX_PER_PAGE = 200
MAX_PIPE_IDS = 100
SELECT = ",".join(
    [
        "id", "doi", "title", "display_name", "publication_year", "publication_date", "cited_by_count",
        "locations", "primary_location", "authorships", "referenced_works", "abstract_inverted_index",
    ]
)  # fmt: skip

_ARXIV_RE = re.compile(
    r"arxiv(?:\.org/(?:abs|pdf)/|\.org:|[:/.])\s*([a-z\-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})", re.I
)


def normalise_work_id(x: str) -> str:
    return x.strip().rstrip("/").rsplit("/", 1)[-1].upper() if "openalex.org" in x else x.strip().upper()


def _bare_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = re.sub(r"^(https?://(dx\.)?doi\.org/|doi:)", "", doi.strip(), flags=re.I)
    return doi.lower() or None


def _find_arxiv(work: dict) -> str | None:
    doi = _bare_doi(work.get("doi")) or ""
    if doi.startswith("10.48550/arxiv."):
        return re.sub(r"v\d+$", "", doi[len("10.48550/arxiv.") :])
    for loc in work.get("locations") or []:
        for field in ("landing_page_url", "pdf_url", "id"):
            v = loc.get(field) or ""
            if "arxiv" in v.lower() and (m := _ARXIV_RE.search(v)):
                return m.group(1)
    return None


def _abstract(inv: dict | None) -> str | None:
    if not inv:
        return None
    words: dict[int, str] = {}
    for word, positions in inv.items():
        for p in positions:
            words[p] = word
    return " ".join(words[i] for i in sorted(words)) or None


def work_to_record(w: dict) -> PaperRecord:
    authors = []
    for i, a in enumerate(w.get("authorships") or []):
        author = a.get("author") or {}
        name = author.get("display_name") or a.get("raw_author_name") or ""
        aid = (author.get("id") or "").rsplit("/", 1)[-1] or "name:" + re.sub(
            r"\s+", " ", name.lower()
        ).strip()
        authors.append(AuthorRecord(author_id=aid, name=name, position=i))
    pdate = None
    if w.get("publication_date"):
        try:
            pdate = date.fromisoformat(w["publication_date"])
        except ValueError:
            pass
    source = ((w.get("primary_location") or {}).get("source")) or {}
    return PaperRecord(
        paper_id=normalise_work_id(w["id"]),
        title=w.get("title") or w.get("display_name") or "",
        doi=_bare_doi(w.get("doi")),
        arxiv_id=_find_arxiv(w),
        year=w.get("publication_year"),
        publication_date=pdate,
        venue=source.get("display_name"),
        citation_count=w.get("cited_by_count"),
        abstract=_abstract(w.get("abstract_inverted_index")),
        authors=authors,
        referenced_ids=[normalise_work_id(r) for r in w.get("referenced_works") or []],
        source="openalex",
    )


class OpenAlexClient:
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
        self.settings = settings
        budget = settings.openalex_daily_budget_usd
        if not settings.openalex_api_key:
            budget = min(budget, UNAUTH_BUDGET_USD)  # the shared no-key allowance is only $0.10/day
        self.budget_usd = budget
        kw = {"sleep": sleep} if sleep else {}
        self.http = CachedClient(
            "openalex",
            cache,
            limiter or RateLimiter(5.0, burst=2),
            conn,
            {"User-Agent": user_agent(settings.contact_email)},
            budget_usd=budget,
            transport=transport,
            **kw,
        )
        self._auth = {}
        if settings.openalex_api_key:
            self._auth["api_key"] = settings.openalex_api_key
        if settings.contact_email:
            self._auth["mailto"] = settings.contact_email

    def _get(self, path: str, params: dict | None = None, cost: float = PRICES["list"]):
        return self.http.get(f"{BASE}{path}", params, cost=cost, auth_params=self._auth)

    # -- single works ---------------------------------------------------------------------------

    def work(self, identifier: str) -> PaperRecord | None:
        """Free lookup by OpenAlex ID, DOI (bare or URL) or arXiv ID. None if unknown."""
        ident = identifier.strip()
        if re.fullmatch(r"(https?://openalex\.org/)?[Ww]\d+", ident):
            path = f"/works/{normalise_work_id(ident)}"
        elif ident.lower().startswith(("10.", "doi:", "http://doi.org", "https://doi.org", "http://dx.doi")):
            path = f"/works/doi:{_bare_doi(ident)}"
        else:
            from lacuna.ingest.arxiv import normalise_arxiv_id

            aid = normalise_arxiv_id(ident)
            if not aid:
                raise ValueError(f"unrecognised identifier: {identifier!r}")
            path = f"/works/doi:10.48550/arxiv.{aid}"
        try:
            data = self._get(path, {"select": SELECT}, cost=PRICES["lookup"])
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            raise
        return work_to_record(data)

    def resolve(self, dois=(), arxiv_ids=()) -> dict[str, PaperRecord]:
        """Free lookups. Keys are "doi:<bare doi>" and "arxiv:<id>"; unknown identifiers are absent."""
        out: dict[str, PaperRecord] = {}
        for d in dois:
            if (b := _bare_doi(d)) and (rec := self.work(f"doi:{b}")):
                out[f"doi:{b}"] = rec
        for a in arxiv_ids:
            if rec := self.work(f"arxiv:{a}"):
                out[f"arxiv:{a}"] = rec
        return out

    # -- lists ----------------------------------------------------------------------------------

    def works_by_ids(self, ids) -> list[PaperRecord]:
        ids = list(dict.fromkeys(normalise_work_id(i) for i in ids))
        found: dict[str, PaperRecord] = {}
        for i in range(0, len(ids), MAX_PIPE_IDS):
            batch = ids[i : i + MAX_PIPE_IDS]
            data = self._get(
                "/works",
                {"filter": "openalex:" + "|".join(batch), "per_page": len(batch), "select": SELECT},
            )
            for w in data.get("results", []):
                rec = work_to_record(w)
                found[rec.paper_id] = rec
        order = {pid: n for n, pid in enumerate(ids)}
        return sorted(found.values(), key=lambda r: order.get(r.paper_id, len(order)))

    def _paged(self, params: dict, max_results: int, cost: float) -> list[PaperRecord]:
        out: list[PaperRecord] = []
        page = 1
        while len(out) < max_results:
            per_page = min(MAX_PER_PAGE, max_results - len(out))
            data = self._get("/works", {**params, "per_page": per_page, "page": page, "select": SELECT}, cost)
            results = data.get("results", [])
            out.extend(work_to_record(w) for w in results)
            if len(results) < per_page:
                break
            page += 1
        return out[:max_results]

    def search_works(
        self, query: str, *, from_date: date | None = None, to_date: date | None = None, max_results: int = 50
    ) -> list[PaperRecord]:
        """Relevance-ordered search; date filters are applied server-side."""
        params: dict = {"search": query}
        filters = []
        if from_date:
            filters.append(f"from_publication_date:{from_date.isoformat()}")
        if to_date:
            filters.append(f"to_publication_date:{to_date.isoformat()}")
        if filters:
            params["filter"] = ",".join(filters)
        return self._paged(params, max_results, PRICES["search"])

    def cited_by(self, work_id: str, max_results: int = 50) -> list[PaperRecord]:
        """Works citing `work_id`, most cited first."""
        params = {"filter": f"cites:{normalise_work_id(work_id)}", "sort": "cited_by_count:desc"}
        return self._paged(params, max_results, PRICES["list"])
