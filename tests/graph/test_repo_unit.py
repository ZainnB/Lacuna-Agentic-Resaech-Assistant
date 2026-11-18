"""Unit tests: a fake driver captures (query, params); no Neo4j needed."""

from types import SimpleNamespace

from lacuna.graph import cypher
from lacuna.graph.repo import BATCH, GraphRepo, citation_pairs, paper_row
from lacuna.models import AuthorRecord, PaperRecord


class FakeDriver:
    def __init__(self, records=None):
        self.calls: list[tuple[str, dict]] = []
        self.records = records or []

    def execute_query(self, query, parameters_=None, routing_=None, **kw):
        self.calls.append((query, dict(parameters_ or {})))
        counters = SimpleNamespace(nodes_created=0, relationships_created=1, properties_set=0)
        return SimpleNamespace(records=self.records, summary=SimpleNamespace(counters=counters))

    def close(self):
        pass


def _rec(pid, refs=(), authors=(), **kw):
    return PaperRecord(paper_id=pid, title=f"T {pid}", referenced_ids=list(refs), authors=list(authors), **kw)


def test_all_cypher_constants_are_static_and_merge_only():
    stmts = [*cypher.ENSURE_CONSTRAINTS, cypher.UPSERT_PAPERS, cypher.UPSERT_TOPIC, cypher.UPSERT_INCLUDES]
    stmts.append(cypher.LINK_CITATIONS)
    for s in stmts:
        assert "{" not in s.replace("{paper_id", "") or "$" in s  # no f-string leftovers
    for s in (cypher.UPSERT_PAPERS, cypher.UPSERT_TOPIC, cypher.UPSERT_INCLUDES, cypher.LINK_CITATIONS):
        assert "MERGE" in s
        assert "CREATE" not in s.replace("ON CREATE", "")
        assert "$" in s
    assert "MATCH (src:Paper" in cypher.LINK_CITATIONS and "MATCH (dst:Paper" in cypher.LINK_CITATIONS
    assert len(cypher.ENSURE_CONSTRAINTS) == 3


def test_ensure_constraints_runs_three_statements():
    d = FakeDriver()
    GraphRepo(d).ensure_constraints()
    assert [q for q, _ in d.calls] == list(cypher.ENSURE_CONSTRAINTS)


def test_paper_row_maps_fields_positions_and_drops_none():
    rec = _rec(
        "W1",
        doi="10.1/x",
        year=2020,
        authors=[
            AuthorRecord(author_id="A1", name="Ann", position=0),
            AuthorRecord(author_id="A2", name="Bo"),  # None -> list index
            AuthorRecord(author_id="A3", name="Cy", position=7),
        ],
    )
    row = paper_row(rec, {"test_run": "r"})
    assert row["paper_id"] == "W1"
    assert row["props"] == {"title": "T W1", "doi": "10.1/x", "year": 2020, "test_run": "r"}
    assert "venue" not in row["props"] and "arxiv_id" not in row["props"]
    assert [(a["author_id"], a["position"]) for a in row["authors"]] == [("A1", 0), ("A2", 1), ("A3", 7)]
    assert row["authors"][0]["props"] == {"name": "Ann", "test_run": "r"}


def test_upsert_papers_sends_parameterised_rows():
    d = FakeDriver()
    out = GraphRepo(d).upsert_papers([_rec("W1"), _rec("W2")])
    ((q, params),) = d.calls
    assert q == cypher.UPSERT_PAPERS
    assert [r["paper_id"] for r in params["rows"]] == ["W1", "W2"]
    assert out["papers"] == 2


def test_upsert_papers_chunks_at_500():
    d = FakeDriver()
    GraphRepo(d).upsert_papers([_rec(f"W{i}") for i in range(BATCH * 2 + 1)])
    assert [len(p["rows"]) for _, p in d.calls] == [500, 500, 1]


def test_citation_pairs_only_in_set_no_self_or_dupes():
    recs = [_rec("W1", ["W2", "W9", "W1", "W2"]), _rec("W2", ["W1"]), _rec("W3")]
    pairs = citation_pairs(recs, {"W1", "W2", "W3"})
    assert [(p["src"], p["dst"]) for p in pairs] == [("W1", "W2"), ("W2", "W1")]


def test_link_citations_sends_only_in_set_pairs_in_chunks():
    n = BATCH + 5
    recs = [_rec("HUB", [f"W{i}" for i in range(n)] + ["OUTSIDE"])] + [_rec(f"W{i}") for i in range(n)]
    d = FakeDriver(records=[{"paper_id": r.paper_id} for r in recs])
    GraphRepo(d).link_citations(recs)
    link = [p for q, p in d.calls if q == cypher.LINK_CITATIONS]
    assert [len(p["rows"]) for p in link] == [500, 5]
    assert all(r["dst"] != "OUTSIDE" for p in link for r in p["rows"])


def test_link_citations_no_pairs_no_write():
    d = FakeDriver(records=[{"paper_id": "W1"}])
    assert GraphRepo(d).link_citations([_rec("W1", ["W9"])]) == 0
    assert all(q != cypher.LINK_CITATIONS for q, _ in d.calls)


def test_upsert_topic_params():
    d = FakeDriver(records=[{"n": 1}])
    GraphRepo(d).upsert_topic("t1", "why?", [{"paper_id": "W1", "role": "seed", "rank": 1, "score": 0.5}])
    (q1, p1), (q2, p2) = d.calls
    assert q1 == cypher.UPSERT_TOPIC and p1["topic_id"] == "t1" and p1["question"] == "why?"
    assert q2 == cypher.UPSERT_INCLUDES
    assert p2["rows"] == [{"paper_id": "W1", "role": "seed", "rank": 1, "score": 0.5}]


def test_reset_requires_confirm():
    import pytest

    d = FakeDriver()
    with pytest.raises(ValueError):
        GraphRepo(d).reset(confirm=False)
    assert d.calls == []
