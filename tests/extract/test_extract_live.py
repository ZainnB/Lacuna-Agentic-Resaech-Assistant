"""Live checks: Neo4j writes on TEST_ ids (cleaned up) and ONE real extraction call."""

# ruff: noqa: E501
import uuid

import pytest

from lacuna.extract.llm import extract_paper
from lacuna.extract.pipeline import resolve_extraction
from lacuna.extract.resolve import Resolver
from lacuna.extract.schemas import Extraction
from lacuna.graph import entities
from lacuna.graph.repo import GraphRepo
from lacuna.models import PaperRecord

pytestmark = pytest.mark.live

VIT_ABSTRACT = (
    "While the Transformer architecture has become the de-facto standard for natural language "
    "processing tasks, its applications to computer vision remain limited. In vision, attention is "
    "either applied in conjunction with convolutional networks, or used to replace certain components "
    "of convolutional networks while keeping their overall structure in place. We show that this "
    "reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image "
    "patches can perform very well on image classification tasks. When pre-trained on large amounts "
    "of data and transferred to multiple mid-sized or small image recognition benchmarks (ImageNet, "
    "CIFAR-100, VTAB, etc.), Vision Transformer (ViT) attains excellent results compared to "
    "state-of-the-art convolutional networks while requiring substantially fewer computational "
    "resources to train."
)


@pytest.fixture
def run_id():
    return uuid.uuid4().hex[:10]


@pytest.fixture
def repo(neo4j_driver, run_id):
    r = GraphRepo(neo4j_driver)
    r.ensure_constraints()
    entities.ensure_entity_constraints(r)
    yield r
    r.delete_by_test_run(run_id)


def _resolved(run_id):
    t = f"TEST_{run_id}"
    c = "chunk"
    return {
        "methods": [
            {"entity_id": f"method:{t}-vit", "canonical_name": f"{t} ViT", "variant": "B/16", "role": "proposed",
             "extends_id": f"method:{t}-tf", "extends_name": f"{t} Transformer", "chunk_id": c, "confidence": 0.9},
            {"entity_id": f"method:{t}-rn", "canonical_name": f"{t} ResNet", "variant": None, "role": "compared",
             "extends_id": None, "extends_name": None, "chunk_id": c, "confidence": 0.6},
        ],
        "datasets": [{"entity_id": f"dataset:{t}-in", "canonical_name": f"{t} ImageNet", "chunk_id": c,
                      "confidence": 0.8}],
        "tasks": [{"entity_id": f"task:{t}-cls", "canonical_name": f"{t} classification", "chunk_id": c,
                   "confidence": 0.8}],
        "metrics": [{"entity_id": f"metric:{t}-acc", "canonical_name": f"{t} accuracy", "higher_is_better": True,
                     "chunk_id": c, "confidence": 0.7}],
        "results": [{"method_id": f"method:{t}-vit", "method": f"{t} ViT", "variant": "B/16",
                     "dataset_id": f"dataset:{t}-in", "dataset": f"{t} ImageNet",
                     "metric_id": f"metric:{t}-acc", "metric": f"{t} accuracy", "value": 88.5, "setting": None,
                     "chunk_id": c, "confidence": 0.9}],
        "limitations": [{"text": "needs a lot of data", "chunk_id": c}],
    }  # fmt: skip


def _count(repo, query, **p):
    return repo.driver.execute_query(query, parameters_=p).records[0]["n"]


def test_graph_write_is_idempotent_and_readable(repo, run_id):
    tag = {"test_run": run_id}
    pid = f"TEST_W{run_id}"
    repo.upsert_papers([PaperRecord(paper_id=pid, title="t")], extra_props=tag)
    repo.upsert_topic(f"TEST_T{run_id}", "q", [{"paper_id": pid, "role": "seed", "rank": 1, "score": 1.0}],
                      extra_props=tag)  # fmt: skip
    res = _resolved(run_id)
    first = entities.write_extraction(repo, pid, res, extracted_by="api:test/x", extra_props=tag)
    assert first["written"] and first["edges"]["proposes"] == 1 and first["edges"]["reports"] == 1
    edges = "MATCH (:Paper {paper_id: $p})-[r]->() RETURN count(r) AS n"
    n1 = _count(repo, edges, p=pid)
    again = entities.write_extraction(repo, pid, res, extracted_by="api:test/x", extra_props=tag)
    assert again["written"] and _count(repo, edges, p=pid) == n1  # replace + MERGE: no duplicates
    nodes = "MATCH (n) WHERE n.test_run = $r RETURN count(n) AS n"
    n_nodes = _count(repo, nodes, r=run_id)
    entities.write_extraction(repo, pid, res, extracted_by="api:test/x", extra_props=tag)
    assert _count(repo, nodes, r=run_id) == n_nodes

    got = entities.paper_entities(repo, pid)
    assert [m["role"] for m in got["methods"]] == ["compared", "proposed"]
    assert got["results"][0]["value"] == 88.5 and got["results"][0]["variant"] == "B/16"
    assert got["methods"][1]["extracted_by"] == "api:test/x"
    ext = "MATCH (a:Method)-[r:EXTENDS {paper_id: $p}]->(b:Method) RETURN count(r) AS n"
    assert _count(repo, ext, p=pid) == 1
    top = entities.topic_entities(repo, f"TEST_T{run_id}")
    assert {m["name"] for m in top["methods"]} == {f"TEST_{run_id} ViT", f"TEST_{run_id} ResNet"}
    assert top["datasets"][0]["papers"] == 1 and top["metrics"] and top["tasks"]


def test_missing_paper_writes_nothing(repo, run_id):
    out = entities.write_extraction(repo, f"TEST_NOPE{run_id}", _resolved(run_id), extracted_by="api:t/x",
                                    extra_props={"test_run": run_id})  # fmt: skip
    assert out["written"] is False
    assert _count(repo, "MATCH (n) WHERE n.test_run = $r RETURN count(n) AS n", r=run_id) == 0


def test_live_extraction_of_vit_abstract(pg_conn):
    """ONE real call: Gemini over the ViT abstract. Run with -s to see the entities."""
    pid, cid = "TEST_VIT", "TEST_VIT:abs"
    pg_conn.execute("DELETE FROM paper_chunks WHERE paper_id = %s", (pid,))
    pg_conn.execute("DELETE FROM extraction_cache WHERE paper_id = %s", (pid,))
    pg_conn.execute(
        "INSERT INTO paper_chunks (chunk_id, paper_id, kind, ord, text, content_hash) "
        "VALUES (%s, %s, 'abstract', 0, %s, 'test-vit-abstract-1')",
        (cid, pid, VIT_ABSTRACT),
    )
    before = {r["entity_id"] for r in pg_conn.execute("SELECT entity_id FROM entities").fetchall()}
    try:
        res = extract_paper(pg_conn, pid, chain=["gemini/gemini-3.5-flash-lite"])
        print("\nEXTRACTED_BY", res.extracted_by, "calls", res.calls, "tokens", res.tokens,
              "rejected", res.rejected_citations)  # fmt: skip
        print(res.extraction.model_dump_json(indent=1))
        assert res.calls == 1 and res.extracted_by.startswith("api:gemini/")
        assert not res.extraction.is_empty()
        assert res.extraction.cited_chunk_ids() <= {cid}
        assert extract_paper(pg_conn, pid, chain=["gemini/gemini-3.5-flash-lite"]).cached

        resolved = resolve_extraction(Resolver(pg_conn), Extraction.model_validate(res.extraction))
        print("RESOLVED_METHODS", [(m["entity_id"], m["variant"], m["role"]) for m in resolved["methods"]])
        print("RESOLVED_DATASETS", [d["entity_id"] for d in resolved["datasets"]])
        print("RESOLVED_TASKS", [t["entity_id"] for t in resolved["tasks"]])
    finally:
        after = {r["entity_id"] for r in pg_conn.execute("SELECT entity_id FROM entities").fetchall()}
        pg_conn.execute("DELETE FROM entities WHERE entity_id = ANY(%s)", (sorted(after - before),))
        pg_conn.execute("DELETE FROM paper_chunks WHERE paper_id = %s", (pid,))
        pg_conn.execute("DELETE FROM extraction_cache WHERE paper_id = %s", (pid,))
