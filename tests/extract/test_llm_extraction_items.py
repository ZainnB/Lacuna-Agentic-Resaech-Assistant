"""Item-level validation: one bad item must not discard a paper's whole extraction."""

from lacuna.extract.llm import validate_items


def test_invalid_items_are_dropped_and_counted():
    raw = {
        "methods": [
            {
                "name": "ViT",
                "variant": None,
                "role": "proposed",
                "extends": None,
                "chunk_id": "p:abs",
                "confidence": 0.9,
            }
        ],
        "datasets": [{"name": "", "chunk_id": "p:abs", "confidence": 0.5}],
        "tasks": [],
        "metrics": [],
        "results": [
            {
                "method": "ViT",
                "dataset": "",
                "metric": "top-1",
                "value": 88.5,
                "setting": None,
                "chunk_id": "p:abs",
                "confidence": 0.8,
            },
            {
                "method": "ViT",
                "dataset": "ImageNet",
                "metric": "top-1",
                "value": 88.5,
                "setting": None,
                "chunk_id": "p:abs",
                "confidence": 0.8,
            },
        ],
        "limitations": [],
    }
    ex, invalid = validate_items(raw)
    assert invalid == 2
    assert [m.name for m in ex.methods] == ["ViT"] and ex.datasets == []
    assert [r.dataset for r in ex.results] == ["ImageNet"]


def test_missing_lists_are_empty():
    ex, invalid = validate_items({})
    assert invalid == 0 and ex.is_empty()


def test_failure_kinds_separate_plumbing_from_model_errors():
    from lacuna.extract.throughput import failure_kind

    assert failure_kind("ChainExhaustedError: [{'outcome': 'exceeds_tpm'}]") == "skipped: over TPM"
    assert (
        failure_kind("ChainExhaustedError: ... 400 - max completion tokens reached") == "output cap reached"
    )
    assert failure_kind("ValidationError: 1 validation error") == "invalid JSON/schema"
