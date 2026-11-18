# ruff: noqa: E501
from lacuna.extract import pipeline
from lacuna.extract.resolve import Resolver
from lacuna.extract.schemas import Extraction

EX = {
    "methods": [
        {"name": "ViT-B/16", "variant": None, "role": "proposed", "extends": "Transformers", "chunk_id": "W1:abs",
         "confidence": 0.9},
        {"name": "Residual Network", "variant": "152", "role": "compared", "extends": None, "chunk_id": "W1:abs",
         "confidence": 0.7},
    ],
    "datasets": [{"name": "ILSVRC-2012", "chunk_id": "W1:abs", "confidence": 0.8}],
    "tasks": [],
    "metrics": [{"name": "Top-1 Acc", "higher_is_better": True, "chunk_id": "W1:abs", "confidence": 0.7}],
    "results": [
        {"method": "ViT-B/16", "dataset": "ImageNet", "metric": "top-1 accuracy", "value": 77.9, "setting": None,
         "chunk_id": "W1:abs", "confidence": 0.9}
    ],
    "limitations": [],
}  # fmt: skip


def test_resolve_extraction_merges_aliases(conn, embedder):
    out = pipeline.resolve_extraction(Resolver(conn, embedder=embedder), Extraction.model_validate(EX))
    vit, resnet = out["methods"]
    assert (vit["entity_id"], vit["variant"], vit["extends_id"]) == (
        "method:vit",
        "B/16",
        "method:transformer",
    )
    assert (resnet["entity_id"], resnet["variant"]) == ("method:resnet", "152")
    assert out["datasets"][0]["entity_id"] == out["results"][0]["dataset_id"] == "dataset:imagenet-1k"
    assert out["metrics"][0]["entity_id"] == out["results"][0]["metric_id"] == "metric:top-1-accuracy"
    assert out["results"][0]["variant"] == "B/16"


def test_store_extraction_writes_graph_and_marks_paper(conn, embedder, FakeRepo):
    repo = FakeRepo()
    out = pipeline.store_extraction(
        conn, repo, "W1", Extraction.model_validate(EX), "api:x/y", resolver=Resolver(conn, embedder=embedder)
    )
    assert out["graph"]["written"] and out["graph"]["edges"]["proposes"] == 1
    assert any("UPDATE papers SET ingest_status = 'extracted'" in q for q in conn.queries)
