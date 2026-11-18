import jsonschema
import pytest

from lacuna.extract.schemas import Extraction, extraction_json_schema

VALID = {
    "methods": [
        {"name": "ViT", "variant": "B/16", "role": "proposed", "extends": None, "chunk_id": "W1:abs",
         "confidence": 0.9}
    ],
    "datasets": [{"name": "ImageNet", "chunk_id": "W1:abs", "confidence": 0.8}],
    "tasks": [{"name": "image classification", "chunk_id": "W1:abs", "confidence": 0.8}],
    "metrics": [
        {"name": "top-1 accuracy", "higher_is_better": True, "chunk_id": "W1:abs", "confidence": 0.7}
    ],
    "results": [
        {"method": "ViT", "dataset": "ImageNet", "metric": "top-1 accuracy", "value": 88.55, "setting": None,
         "chunk_id": "W1:abs", "confidence": 0.9}
    ],
    "limitations": [{"text": "needs large data", "chunk_id": "W1:abs"}],
}  # fmt: skip

FORBIDDEN = {"minimum", "maximum", "format", "default", "minLength", "maxLength", "pattern", "$ref", "anyOf"}


def _walk(node, path=""):
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from _walk(v, f"{path}/{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")


def test_schema_is_strict_shaped():
    schema = extraction_json_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    n_objects = 0
    for path, node in _walk(schema):
        if node.get("type") == "object":
            n_objects += 1
            assert node["additionalProperties"] is False, path
            assert set(node["required"]) == set(node["properties"]), path
        assert not (FORBIDDEN & set(node)) or path.endswith("/properties"), path
    assert n_objects == 7  # root + six item types
    method = schema["properties"]["methods"]["items"]["properties"]
    assert method["variant"]["type"] == ["string", "null"]
    assert method["extends"]["type"] == ["string", "null"]
    assert schema["properties"]["metrics"]["items"]["properties"]["higher_is_better"]["type"] == [
        "boolean",
        "null",
    ]
    assert schema["properties"]["results"]["items"]["properties"]["value"] == {"type": "number"}


def test_valid_payload_passes_both_validators():
    jsonschema.validate(VALID, extraction_json_schema())
    ex = Extraction.model_validate(VALID)
    assert ex.counts() == {
        "methods": 1,
        "datasets": 1,
        "tasks": 1,
        "metrics": 1,
        "results": 1,
        "limitations": 1,
    }
    assert ex.cited_chunk_ids() == {"W1:abs"}


def test_missing_property_fails_strict_schema():
    bad = {**VALID, "methods": [{k: v for k, v in VALID["methods"][0].items() if k != "extends"}]}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, extraction_json_schema())


def test_confidence_is_clamped_not_rejected():
    ex = Extraction.model_validate({"datasets": [{"name": "X", "chunk_id": "c", "confidence": 1.7}]})
    assert ex.datasets[0].confidence == 1.0


def test_bad_role_and_non_numeric_value_rejected():
    with pytest.raises(ValueError):
        Extraction.model_validate(
            {"methods": [{"name": "A", "role": "invented", "chunk_id": "c", "confidence": 1}]}
        )
    with pytest.raises(ValueError):
        Extraction.model_validate(
            {"results": [{"method": "A", "dataset": "B", "metric": "C", "value": "high", "chunk_id": "c",
                          "confidence": 1}]}
        )  # fmt: skip
