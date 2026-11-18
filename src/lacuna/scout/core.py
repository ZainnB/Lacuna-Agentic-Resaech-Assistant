"""Seed search -> depth-1 citation snowball -> ranked paper set.

Phase 1 uses no LLM: this is the $0 "snowball" baseline that E1 compares against plain keyword
search. Ranking is deterministic and every cap is enforced here, not in a prompt (blueprint 8).

Ranking: seeds keep their search order (they answer the question directly). Snowball candidates
are scored by how many seeds link to them (the strongest signal that a paper belongs to the
topic), then by log citations and recency. Papers published after `to_date` are dropped, which is
the leakage rule for E1 (nothing the survey could not have cited).
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import date

import httpx

from lacuna.ingest.openalex import BudgetExceeded, OpenAlexClient
from lacuna.models import PaperRecord

log = logging.getLogger(__name__)


@dataclass
class ScoutConfig:
    max_papers: int = 60
    seeds: int = 20
    depth: int = 1  # 0 = keyword search only; 1 = add references and citations of the seeds
    ref_candidates: int = 300  # most-shared references whose metadata is fetched
    cites_per_seed: int = 50
    from_date: date | None = None
    to_date: date | None = None  # leakage freeze: nothing published after this date

    def as_dict(self) -> dict:
        return {k: (v.isoformat() if isinstance(v, date) else v) for k, v in self.__dict__.items()}


@dataclass
class Ranked:
    record: PaperRecord
    role: str  # seed | snowball_ref | snowball_cite
    score: float
    reason: str
    rank: int = 0


@dataclass
class ScoutResult:
    question: str
    status: str  # ok | no_results | budget_hit | rate_limited
    papers: list[Ranked]
    n_seeds: int = 0
    n_ref_candidates: int = 0
    n_cite_candidates: int = 0
    notes: list[str] = field(default_factory=list)

    def ids(self) -> list[str]:
        return [r.record.paper_id for r in self.papers]


def in_window(rec: PaperRecord, cfg: ScoutConfig) -> bool:
    d = rec.publication_date or (date(rec.year, 12, 31) if rec.year else None)
    if d is None:
        return False  # undated papers cannot be checked against the freeze, so they are excluded
    if cfg.to_date and d > cfg.to_date:
        return False
    if cfg.from_date and d < cfg.from_date:
        return False
    return True


def candidate_score(rec: PaperRecord, links: int, today: date) -> float:
    cites = math.log1p(rec.citation_count or 0)
    age = max(0, today.year - (rec.year or today.year))
    recency = 1.0 / (1.0 + age / 5.0)
    return links * 3.0 + cites * 0.5 + recency  # links dominate: log1p(50k cites)*0.5 ~ 5.4 < 2 links


def scout(
    question: str,
    openalex: OpenAlexClient,
    cfg: ScoutConfig | None = None,
    *,
    exclude: set[str] | None = None,
    today: date | None = None,
) -> ScoutResult:
    cfg = cfg or ScoutConfig()
    exclude = exclude or set()
    today = today or date.today()
    notes: list[str] = []

    try:
        seeds = [
            r
            for r in openalex.search_works(
                question, from_date=cfg.from_date, to_date=cfg.to_date, max_results=cfg.seeds
            )
            if r.paper_id not in exclude and in_window(r, cfg)
        ]
        if not seeds:
            return ScoutResult(question, "no_results", [], notes=["seed search returned nothing"])
        ranked = [
            Ranked(r, "seed", float(cfg.seeds - i), f"search rank {i + 1}") for i, r in enumerate(seeds)
        ]
        seen = {r.paper_id for r in seeds} | exclude
        n_ref_candidates = n_cite_candidates = 0

        if cfg.depth >= 1:
            # References: count how many seeds cite each work, fetch metadata for the most shared.
            ref_links = Counter(ref for s in seeds for ref in s.referenced_ids if ref not in seen)
            top_refs = [w for w, _ in ref_links.most_common(cfg.ref_candidates)]
            n_ref_candidates = len(ref_links)
            cand: dict[str, tuple[PaperRecord, int, str]] = {}
            if top_refs:
                for rec in openalex.works_by_ids(top_refs):
                    cand[rec.paper_id] = (rec, ref_links[rec.paper_id], "snowball_ref")

            # Citations: works that cite a seed. Each seed is one list call.
            cite_links: Counter[str] = Counter()
            cite_recs: dict[str, PaperRecord] = {}
            for s in seeds:
                for rec in openalex.cited_by(s.paper_id, max_results=cfg.cites_per_seed):
                    if rec.paper_id in seen:
                        continue
                    cite_links[rec.paper_id] += 1
                    cite_recs.setdefault(rec.paper_id, rec)
            n_cite_candidates = len(cite_links)
            for pid, rec in cite_recs.items():
                links = cite_links[pid] + (cand[pid][1] if pid in cand else 0)
                # A paper both cited by and citing seeds counts both links; keep the ref role.
                role = cand[pid][2] if pid in cand else "snowball_cite"
                cand[pid] = (rec, links, role)

            scored = [
                Ranked(rec, role, candidate_score(rec, links, today), f"{links} seed link(s)")
                for rec, links, role in cand.values()
                if in_window(rec, cfg)
            ]
            scored.sort(key=lambda r: (-r.score, r.record.paper_id))
            ranked += scored

        status = "ok"
    except BudgetExceeded as e:
        notes.append(str(e))
        return ScoutResult(question, "budget_hit", [], notes=notes)
    except httpx.HTTPStatusError as e:
        if e.response.status_code != 429:
            raise
        notes.append("rate limited after retries")
        return ScoutResult(question, "rate_limited", [], notes=notes)

    ranked = ranked[: cfg.max_papers]
    for i, r in enumerate(ranked):
        r.rank = i + 1
    log.info(
        "scout %r: %d seeds, %d ref candidates, %d cite candidates -> %d papers",
        question,
        len(seeds),
        n_ref_candidates,
        n_cite_candidates,
        len(ranked),
    )
    return ScoutResult(question, status, ranked, len(seeds), n_ref_candidates, n_cite_candidates, notes)
