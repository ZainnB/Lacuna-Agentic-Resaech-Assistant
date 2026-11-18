import re

import pytest

from lacuna.graph import entities
from lacuna.graph.entities import build_rows

RESOLVED = {
    "methods": [
        {"entity_id": "method:vit", "canonical_name": "ViT", "variant": "B/16", "role": "proposed",
         "extends_id": "method:transformer", "extends_name": "Transformer", "chunk_id": "W1:abs",
         "confidence": 0.9},
        {"entity_id": "method:resnet", "canonical_name": "ResNet", "variant": "152", "role": "compared",
         "extends_id": None, "extends_name": None, "chunk_id": "W1:abs", "confidence": 0.7},
    ],
    "datasets": [{"entity_id": "dataset:imagenet-1k", "canonical_name": "ImageNet-1k", "chunk_id": "W1:abs",
                  "confidence": 0.8}],
    "tasks": [{"entity_id": "task:image-classification", "canonical_name": "image classification",
               "chunk_id": "W1:abs", "confidence": 0.8}],
    "metrics": [
        {"entity_id": "metric:top-1-accuracy", "canonical_name": "top-1 accuracy", "higher_is_better": True,
         "chunk_id": "W1:abs", "confidence": 0.7},
        {"entity_id": "metric:f1", "canonical_name": "F1", "higher_is_better": None, "chunk_id": "W1:abs",
         "confidence": 0.4},
    ],
    "results": [
        {"method_id": "method:vit", "method": "ViT", "variant": "B/16", "dataset_id": "dataset:cifar-100",
         "dataset": "CIFAR-100", "metric_id": "metric:top-1-accuracy", "metric": "top-1 accuracy",
         "value": 94.55, "setting": None, "chunk_id": "W1:abs", "confidence": 0.9},
    ],
    "limitations": [{"text": "needs big data", "chunk_id": "W1:abs"}],
}  # fmt: skip


def test_all_statements_are_parameterised_merge_only():
    stmts = {n: v for n, v in vars(entities).items() if n.isupper() and isinstance(v, str)}
    assert len(stmts) >= 15
    for name, q in stmts.items():
        assert not re.search(r"\bCREATE\s+(?!CONSTRAINT)", q), name
        assert "%s" not in q and "f'" not in q, name
    for q in entities.WRITE_STATEMENTS:
        assert "MERGE" in q and "$extracted_by" in q and "MATCH (p:Paper {paper_id: $paper_id})" in q
    assert "MERGE (n:Method {entity_id: row.entity_id})" in entities.UPSERT_PROPOSES
    assert "PROPOSES" in entities.UPSERT_PROPOSES and "USES_METHOD" in entities.UPSERT_USES_METHOD
    assert "EXTENDS {paper_id: $paper_id}" in entities.UPSERT_EXTENDS
    for read in (entities.TOPIC_METHODS, entities.TOPIC_DATASETS, entities.PAPER_METHODS):
        assert "$topic_id" in read or "$paper_id" in read


def test_build_rows_edges():
    rows = build_rows(RESOLVED, "W1")
    assert [r["entity_id"] for r in rows["proposes"]] == ["method:vit"]
    assert [r["entity_id"] for r in rows["uses"]] == ["method:resnet"]
    assert rows["extends"] == [
        {"src_id": "method:vit", "src_name": "ViT", "dst_id": "method:transformer", "dst_name": "Transformer",
         "chunk_id": "W1:abs", "confidence": 0.9}
    ]  # fmt: skip
    ev = {(r["entity_id"], r["method"]) for r in rows["evaluates_on"]}
    assert ev == {("dataset:imagenet-1k", ""), ("dataset:cifar-100", "ViT")}  # result implies EVALUATES_ON
    rep = {r["entity_id"]: r for r in rows["reports"]}
    assert rep["metric:top-1-accuracy"]["value"] == 94.55
    assert rep["metric:top-1-accuracy"]["higher_is_better"] is True
    assert (
        rep["metric:top-1-accuracy"]["method"] == "ViT" and rep["metric:top-1-accuracy"]["variant"] == "B/16"
    )
    assert rep["metric:f1"]["value"] is None and rep["metric:f1"]["method"] == ""
    assert len(rows["limitations"]) == 1 and len(rows["limitations"][0]["limitation_id"]) == 24
    assert rows["limitations"] == build_rows(RESOLVED, "W1")["limitations"]  # stable id
    assert (
        rows["limitations"][0]["limitation_id"]
        != build_rows(RESOLVED, "W2")["limitations"][0]["limitation_id"]
    )


def test_self_extends_dropped():
    r = {"methods": [{**RESOLVED["methods"][0], "extends_id": "method:vit"}]}
    assert build_rows(r, "W1")["extends"] == []


def test_write_extraction_runs_parameterised_statements_in_one_tx(FakeRepo):
    repo = FakeRepo()
    out = entities.write_extraction(
        repo, "W1", RESOLVED, extracted_by="api:gemini/x", extra_props={"test_run": "t"}
    )
    assert out["written"] and out["edges"]["proposes"] == 1 and out["edges"]["reports"] == 2
    queries = [q for q, _ in repo.tx.runs]
    assert queries[0] == entities.CLEAR_PAPER_EXTRACTION
    for q, params in repo.tx.runs:
        assert params["paper_id"] == "W1"
        if q != entities.CLEAR_PAPER_EXTRACTION:
            assert params["extracted_by"] == "api:gemini/x" and params["props"] == {"test_run": "t"}
            assert "W1" not in q and "ViT" not in q  # data only ever travels as parameters


def test_no_paper_node_no_write(FakeRepo):
    repo = FakeRepo(exists=0)
    out = entities.write_extraction(repo, "WX", RESOLVED, extracted_by="api:x/y")
    assert out["written"] is False and repo.tx.runs == []


def test_replace_false_skips_clear(FakeRepo):
    repo = FakeRepo()
    entities.write_extraction(repo, "W1", RESOLVED, extracted_by="api:x/y", replace=False)
    assert entities.CLEAR_PAPER_EXTRACTION not in [q for q, _ in repo.tx.runs]


@pytest.mark.parametrize("fn", ["paper_entities", "topic_entities"])
def test_reads_exist(fn):
    assert callable(getattr(entities, fn))
