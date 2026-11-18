"""Live tests against Neo4j. Every node is tagged with a `test_run` uuid and only those are deleted."""

import uuid

import pytest

from lacuna.graph.repo import GraphRepo
from lacuna.models import AuthorRecord, PaperRecord

pytestmark = pytest.mark.live


@pytest.fixture
def run_id():
    return uuid.uuid4().hex


@pytest.fixture
def repo(neo4j_driver, run_id):
    r = GraphRepo(neo4j_driver)
    r.ensure_constraints()
    yield r
    r.delete_by_test_run(run_id)  # does not close the shared session driver


def _records(run_id):
    a, b, c = (f"Wt{run_id[:8]}{x}" for x in "abc")
    au = [AuthorRecord(author_id=f"At{run_id[:8]}1", name="Ann", position=0)]
    return (
        a,
        b,
        c,
        [
            PaperRecord(paper_id=a, title="A", year=2020, authors=au, referenced_ids=[b, c, "W_OUTSIDE"]),
            PaperRecord(
                paper_id=b,
                title="B",
                authors=au + [AuthorRecord(author_id=f"At{run_id[:8]}2", name="Bo", position=1)],
                referenced_ids=[c],
            ),
            PaperRecord(paper_id=c, title="C", citation_count=9),
        ],
    )


def _count(driver, run_id, query):
    recs, _, _ = driver.execute_query(query, run=run_id)
    return recs[0]["n"]


def test_constraints_idempotent(repo, neo4j_driver):
    repo.ensure_constraints()
    repo.ensure_constraints()
    recs, _, _ = neo4j_driver.execute_query("SHOW CONSTRAINTS YIELD name RETURN collect(name) AS names")
    for name in ("paper_paper_id", "author_author_id", "topic_topic_id"):
        assert name in recs[0]["names"]


def test_upsert_twice_same_counts_and_exact_cites(repo, neo4j_driver, run_id):
    a, b, c, recs = _records(run_id)
    tag = {"test_run": run_id}
    for _ in range(2):
        repo.upsert_papers(recs, extra_props=tag)
        repo.link_citations(recs)
    nodes = "MATCH (n {test_run: $run}) RETURN count(n) AS n"
    assert _count(neo4j_driver, run_id, nodes) == 5  # 3 papers + 2 authors
    assert (
        _count(neo4j_driver, run_id, "MATCH (:Paper {test_run: $run})-[r:CITES]->() RETURN count(r) AS n")
        == 3
    )
    assert (
        _count(neo4j_driver, run_id, "MATCH (:Author {test_run: $run})-[r:AUTHORED]->() RETURN count(r) AS n")
        == 3
    )
    # no stub for the outside reference
    assert _count(neo4j_driver, run_id, "MATCH (p:Paper {paper_id: 'W_OUTSIDE'}) RETURN count(p) AS n") == 0
    pos = neo4j_driver.execute_query(
        "MATCH (a:Author)-[r:AUTHORED]->(:Paper {paper_id: $p}) RETURN a.author_id AS id, r.position AS pos",
        p=b,
    ).records
    assert {r["id"]: r["pos"] for r in pos} == {f"At{run_id[:8]}1": 0, f"At{run_id[:8]}2": 1}


def test_topic_includes_and_summary(repo, run_id):
    a, b, c, recs = _records(run_id)
    tag = {"test_run": run_id}
    repo.upsert_papers(recs, extra_props=tag)
    repo.link_citations(recs)
    tid = f"t_{run_id[:8]}"
    members = [
        {"paper_id": a, "role": "seed", "rank": 1, "score": 0.9},
        {"paper_id": c, "role": "cited", "rank": 2, "score": 0.5},
        {"paper_id": "W_NOT_IN_GRAPH", "role": "x", "rank": 3, "score": 0.1},
    ]
    for _ in range(2):
        out = repo.upsert_topic(tid, "q?", members, extra_props=tag)
    assert out["included"] == 2
    papers = repo.topic_papers(tid)
    assert [(p["paper_id"], p["role"], p["rank"]) for p in papers] == [(a, "seed", 1), (c, "cited", 2)]
    s = repo.topic_summary(tid)
    assert s["n_papers"] == 2 and s["n_cites_within"] == 1 and s["question"] == "q?"
    assert s["top_cited"][0]["paper_id"] == c and s["top_cited"][0]["in_set_citations"] == 1
    assert repo.topic_summary("nope") is None


def test_neighbourhood_shape(repo, run_id):
    a, b, c, recs = _records(run_id)
    repo.upsert_papers(recs, extra_props={"test_run": run_id})
    repo.link_citations(recs)
    h = repo.neighbourhood(b)
    assert set(h) == {"paper", "cites", "cited_by", "authors"}
    assert h["paper"]["paper_id"] == b
    assert [x["paper_id"] for x in h["cites"]] == [c]
    assert [x["paper_id"] for x in h["cited_by"]] == [a]
    assert [(x["name"], x["position"]) for x in h["authors"]] == [("Ann", 0), ("Bo", 1)]
    assert repo.neighbourhood("W_MISSING") is None
