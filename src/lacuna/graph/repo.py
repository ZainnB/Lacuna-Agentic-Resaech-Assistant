"""Neo4j repository. The only module that talks to the driver; Cypher lives in `cypher.py`.

Writes are idempotent (MERGE) and batched in chunks of `BATCH` rows so one UNWIND never carries a
whole topic. Citations are resolved in Python against the set of `Paper` nodes already present:
outside references stay IDs in Postgres (`paper_references`), never stub nodes (blueprint 6).
"""

from __future__ import annotations

from itertools import batched
from typing import Any

from neo4j import Driver, GraphDatabase, RoutingControl

from lacuna.graph import cypher
from lacuna.models import PaperRecord
from lacuna.settings import Settings, get_settings

BATCH = 500

_COUNTERS = ("nodes_created", "relationships_created", "properties_set")


def paper_row(record: PaperRecord, extra_props: dict[str, Any] | None = None) -> dict:
    """Shape one PaperRecord for `UPSERT_PAPERS`.

    None values are dropped so a sparse re-ingest never erases a property a richer source set.
    An author without an explicit position takes its index in the record's author list.
    """
    extra = dict(extra_props or {})
    props = {
        "title": record.title,
        "doi": record.doi,
        "arxiv_id": record.arxiv_id,
        "year": record.year,
        "venue": record.venue,
        "citation_count": record.citation_count,
    }
    props = {k: v for k, v in props.items() if v is not None} | extra
    authors = [
        {
            "author_id": a.author_id,
            "position": a.position if a.position is not None else i,
            "props": {"name": a.name} | extra,
        }
        for i, a in enumerate(record.authors)
    ]
    return {"paper_id": record.paper_id, "props": props, "authors": authors}


def citation_pairs(records: list[PaperRecord], present: set[str]) -> list[dict]:
    """(src, dst) pairs whose endpoints are both known `Paper` nodes, deduplicated, no self-loops."""
    seen: set[tuple[str, str]] = set()
    rows = []
    for rec in records:
        if rec.paper_id not in present:
            continue
        for ref in rec.referenced_ids:
            pair = (rec.paper_id, ref)
            if ref == rec.paper_id or ref not in present or pair in seen:
                continue
            seen.add(pair)
            rows.append({"src": rec.paper_id, "dst": ref, "source": rec.source})
    return rows


class GraphRepo:
    def __init__(self, driver: Driver) -> None:
        self.driver = driver

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> GraphRepo:
        cfg = settings or get_settings()
        return cls(GraphDatabase.driver(cfg.neo4j_uri, auth=(cfg.neo4j_user, cfg.neo4j_password)))

    def close(self) -> None:
        self.driver.close()

    def __enter__(self) -> GraphRepo:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------ plumbing

    def _write(self, query: str, **params):
        return self.driver.execute_query(query, parameters_=params, routing_=RoutingControl.WRITE)

    def _read(self, query: str, **params):
        return self.driver.execute_query(query, parameters_=params, routing_=RoutingControl.READ)

    def _write_batched(self, query: str, rows: list[dict], **params) -> dict[str, int]:
        """Run `query` once per chunk of `rows`; sum the summary counters and any `n` column."""
        totals = dict.fromkeys((*_COUNTERS, "n"), 0)
        for chunk in batched(rows, BATCH):
            result = self._write(query, rows=list(chunk), **params)
            counters = result.summary.counters
            for name in _COUNTERS:
                totals[name] += getattr(counters, name, 0)
            totals["n"] += sum(r["n"] for r in result.records if "n" in r.keys())
        return totals

    # ------------------------------------------------------------ schema

    def ensure_constraints(self) -> None:
        for statement in cypher.ENSURE_CONSTRAINTS:
            self._write(statement)

    # ------------------------------------------------------------ writes

    def upsert_papers(self, records: list[PaperRecord], *, extra_props: dict | None = None) -> dict:
        """MERGE papers, their authors and AUTHORED edges. `extra_props` lands on every node."""
        rows = [paper_row(r, extra_props) for r in records]
        totals = self._write_batched(cypher.UPSERT_PAPERS, rows)
        return {
            "papers": len(rows),
            "authors": len({a["author_id"] for row in rows for a in row["authors"]}),
            **{k: totals[k] for k in _COUNTERS},
        }

    def upsert_topic(
        self, topic_id: str, question: str, memberships: list[dict], *, extra_props: dict | None = None
    ) -> dict:
        """MERGE the Topic and its INCLUDES edges. Memberships for papers not in the graph are skipped."""
        self._write(cypher.UPSERT_TOPIC, topic_id=topic_id, question=question, props=dict(extra_props or {}))
        rows = [
            {
                "paper_id": m["paper_id"],
                "role": m.get("role"),
                "rank": m.get("rank"),
                "score": m.get("score"),
            }
            for m in memberships
        ]
        totals = self._write_batched(cypher.UPSERT_INCLUDES, rows, topic_id=topic_id)
        return {
            "topic_id": topic_id,
            "memberships": len(rows),
            "included": totals["n"],
            "relationships_created": totals["relationships_created"],
        }

    def link_citations(self, records: list[PaperRecord]) -> int:
        """Create CITES for referenced_ids that are present `Paper` nodes. Returns edges created."""
        present = self.paper_ids()
        rows = citation_pairs(records, present)
        if not rows:
            return 0
        return self._write_batched(cypher.LINK_CITATIONS, rows)["relationships_created"]

    def delete_by_test_run(self, test_run: str) -> int:
        """Remove every node (and its edges) tagged by a live test. Returns nodes deleted."""
        result = self._write(cypher.DELETE_BY_TEST_RUN, test_run=test_run)
        return result.records[0]["n"] if result.records else 0

    def reset(self, *, confirm: bool) -> int:
        """Delete the whole graph. Dev only; refuses without an explicit `confirm=True`."""
        if confirm is not True:
            raise ValueError("reset() deletes every node; pass confirm=True")
        result = self._write(cypher.RESET_ALL)
        return result.records[0]["n"] if result.records else 0

    # ------------------------------------------------------------ reads

    def paper_ids(self) -> set[str]:
        return {r["paper_id"] for r in self._read(cypher.PAPER_IDS).records}

    def stats(self) -> dict:
        nodes = {r["label"]: r["n"] for r in self._read(cypher.NODE_COUNTS).records}
        rels = {r["type"]: r["n"] for r in self._read(cypher.RELATIONSHIP_COUNTS).records}
        return {"nodes": nodes, "relationships": rels}

    def neighbourhood(self, paper_id: str) -> dict | None:
        """Depth 1: the paper, what it cites, what cites it, its authors. None if unknown."""
        result = self._read(cypher.NEIGHBOURHOOD, paper_id=paper_id)
        if not result.records:
            return None
        r = result.records[0]
        return {"paper": r["paper"], "cites": r["cites"], "cited_by": r["cited_by"], "authors": r["authors"]}

    def topic_papers(self, topic_id: str) -> list[dict]:
        return [r.data() for r in self._read(cypher.TOPIC_PAPERS, topic_id=topic_id).records]

    def top_cited_in_topic(self, topic_id: str, n: int = 10) -> list[dict]:
        return [r.data() for r in self._read(cypher.TOP_CITED_IN_TOPIC, topic_id=topic_id, n=int(n)).records]

    def topic_summary(self, topic_id: str, *, top_n: int = 10) -> dict | None:
        """Member count, CITES edges between members, and the most cited members. None if unknown."""
        result = self._read(cypher.TOPIC_COUNTS, topic_id=topic_id)
        if not result.records:
            return None
        r = result.records[0]
        return {
            "topic_id": topic_id,
            "question": r["question"],
            "n_papers": r["n_papers"],
            "n_cites_within": r["n_cites_within"],
            "top_cited": self.top_cited_in_topic(topic_id, top_n),
        }
