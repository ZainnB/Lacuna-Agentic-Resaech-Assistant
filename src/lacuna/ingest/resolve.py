"""Turn S2/arXiv/OpenAlex `SearchHit`s into `PaperRecord`s with OpenAlex IDs."""

from __future__ import annotations

from lacuna.ingest.arxiv import normalise_arxiv_id
from lacuna.ingest.openalex import OpenAlexClient, _bare_doi
from lacuna.models import PaperRecord, SearchHit


def resolve_hits(hits: list[SearchHit], openalex: OpenAlexClient) -> list[PaperRecord]:
    """Resolve in hit order. Lookup order per hit: OpenAlex ID, DOI, then arXiv ID (all free except
    the batched OpenAlex-ID fetch). Duplicates collapse to the first hit; unknown papers are dropped."""
    oa_ids = [h.source_id for h in hits if h.source == "openalex"]
    by_id = {r.paper_id: r for r in openalex.works_by_ids(oa_ids)} if oa_ids else {}
    out: list[PaperRecord] = []
    seen: set[str] = set()
    memo: dict[str, PaperRecord | None] = {}

    def lookup(key: str) -> PaperRecord | None:
        if key not in memo:
            memo[key] = openalex.work(key)
        return memo[key]

    for h in hits:
        rec = None
        if h.source == "openalex":
            rec = by_id.get(h.source_id.upper())
        if rec is None and (doi := _bare_doi(h.doi)):
            rec = lookup(f"doi:{doi}")
        if rec is None and (aid := normalise_arxiv_id(h.arxiv_id or "")):
            rec = lookup(f"arxiv:{aid}")
        if rec is None or rec.paper_id in seen:
            continue
        seen.add(rec.paper_id)
        if h.source == "s2" and not rec.s2_id:
            rec = rec.model_copy(update={"s2_id": h.source_id})
        out.append(rec)
    return out
