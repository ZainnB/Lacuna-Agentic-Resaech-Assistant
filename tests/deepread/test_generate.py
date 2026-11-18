import pytest

from lacuna.deepread import generate
from lacuna.llm import ChainExhaustedError, LLMResult

CFG = {"model": "groq/openai/gpt-oss-120b", "max_tokens": 256}


def test_call_json_maps_result_to_dict(monkeypatch):
    seen = {}

    def fake(role, system, user, schema, **kw):
        seen.update(role=role, **kw)
        return LLMResult(
            parsed={"answer": "42", "citations": []},
            text="{}",
            provider="groq",
            model="openai/gpt-oss-120b",
            input_tokens=10,
            output_tokens=3,
            cached=True,
            stop_reason="stop",
        )

    monkeypatch.setattr(generate, "complete_json", fake)
    out = generate.call_json("sys", "user", generate.ANSWER_SCHEMA, CFG)
    assert out == {
        "parsed": {"answer": "42", "citations": []},
        "model": "openai/gpt-oss-120b",
        "provider": "groq",
        "stop_reason": "stop",
        "input_tokens": 10,
        "output_tokens": 3,
        "cached": True,
    }
    assert seen["chain"] == [CFG["model"]]  # one-element chain: no fallback
    assert seen["max_tokens"] == 256


def test_llm_error_becomes_parsed_none(monkeypatch):
    def boom(*a, **kw):
        raise ChainExhaustedError([{"spec": CFG["model"], "error": "429"}])

    monkeypatch.setattr(generate, "complete_json", boom)
    out = generate.call_json("sys", "user", generate.ANSWER_SCHEMA, CFG)
    assert out["parsed"] is None
    assert out["stop_reason"] == "error"
    assert out["provider"] == "groq" and out["model"] == "openai/gpt-oss-120b"
    answer = generate.generate("q?", [], CFG)
    assert answer["answer"] is None and answer["citations"] == []


def test_non_llm_errors_propagate(monkeypatch):
    monkeypatch.setattr(generate, "complete_json", lambda *a, **kw: (_ for _ in ()).throw(KeyError("x")))
    with pytest.raises(KeyError):
        generate.call_json("s", "u", {}, CFG)
