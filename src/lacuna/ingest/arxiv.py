"""arXiv API client (Atom). One request per 3 s, per arXiv's API terms of use."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

from lacuna.cache import Cache
from lacuna.ingest.http import CachedClient, user_agent
from lacuna.models import SearchHit
from lacuna.ratelimit import RateLimiter
from lacuna.settings import Settings

BASE = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
_ID_RE = re.compile(r"(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Za-z]{2})?/\d{7})(?:v\d+)?", re.I)


def normalise_arxiv_id(x: str) -> str | None:
    """ "arXiv:2010.11929v2", "https://arxiv.org/pdf/2010.11929v2.pdf" -> "2010.11929". None if no id."""
    s = x.strip()
    s = re.sub(r"^(https?://)?(www\.)?(export\.)?arxiv\.org/(abs|pdf)/", "", s, flags=re.I)
    s = re.sub(r"^arxiv:", "", s, flags=re.I)
    s = re.sub(r"\.pdf$", "", s, flags=re.I)
    m = _ID_RE.fullmatch(s)
    return m.group(1) if m else None


def parse_atom(xml_text: str) -> list[SearchHit]:
    root = ET.fromstring(xml_text)
    hits = []
    for e in root.findall("a:entry", NS):
        raw_id = (e.findtext("a:id", "", NS) or "").strip()
        aid = normalise_arxiv_id(raw_id)
        if not aid:
            continue  # error entries carry a non-paper id
        published = e.findtext("a:published", "", NS) or ""
        doi = (e.findtext("x:doi", None, NS) or "").strip().lower() or None
        hits.append(
            SearchHit(
                source="arxiv",
                source_id=aid,
                title=re.sub(r"\s+", " ", e.findtext("a:title", "", NS) or "").strip(),
                doi=doi,
                arxiv_id=aid,
                year=int(published[:4]) if published[:4].isdigit() else None,
            )
        )
    return hits


class ArxivClient:
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
        kw = {"sleep": sleep} if sleep else {}
        self.http = CachedClient(
            "arxiv",
            cache,
            limiter or RateLimiter(1 / 3),
            conn,
            {"User-Agent": user_agent(settings.contact_email)},
            transport=transport,
            **kw,
        )

    def search(self, query: str, max_results: int = 10, start: int = 0) -> list[SearchHit]:
        params = {
            "search_query": query if ":" in query else f"all:{query}",
            "start": start,
            "max_results": max_results,
            "sortBy": "relevance",
        }
        return parse_atom(self.http.get(BASE, params, as_text=True))

    def download_pdf(self, arxiv_id: str, dest_dir: Path) -> Path:
        aid = normalise_arxiv_id(arxiv_id)
        if not aid:
            raise ValueError(f"bad arXiv id: {arxiv_id!r}")
        return self.http.download(
            f"https://arxiv.org/pdf/{aid}", Path(dest_dir) / f"{aid.replace('/', '_')}.pdf"
        )
