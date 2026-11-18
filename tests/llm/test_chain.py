"""Provider-layer tests. No network: openai talks to an httpx2 MockTransport."""

import json

import httpx2
import openai
import pytest

from lacuna.cache import MemoryCache
from lacuna.llm import ChainExhaustedError, SpendStageError
from lacuna.llm.chain import Chain
from lacuna.llm.config import LLMConfig
from lacuna.llm.providers import split_spec

SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
G = "groq/openai/gpt-oss-120b"
GEM = "gemini/gemini-3.5-flash-lite"
CER = "cerebras/gpt-oss-120b"


def ok_body(content='{"n": 1}', pt=10, ct=5):
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}
        ],
        "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
    }


class FakeNet:
    """Scripted responses keyed by request host; records every request."""

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.requests = []

    def handler(self, request):
        self.requests.append(request)
        queue = self.script[request.url.host]
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, httpx2.Response):
            return item
        return httpx2.Response(200, json=ok_body(item))

    def factory(self, spec, settings):
        return openai.OpenAI(
            base_url=spec.base(settings),
            api_key=spec.api_key(settings) or "none",
            http_client=httpx2.Client(transport=httpx2.MockTransport(self.handler)),
            max_retries=0,
        )


class Clock:
    def __init__(self):
        self.t = 1000.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s


def make(settings, net, *, keys=("gemini", "groq", "cerebras"), stage=0, conn=None, cache=None, limits=None):
    s = settings.model_copy(update={f"{k}_api_key": "fake" for k in keys} | {"spend_stage": stage})
    clock = Clock()
    cfg = LLMConfig(chains={"default": [GEM, G, CER]}, limits=limits or {})
    c = Chain(
        s, cache if cache is not None else MemoryCache(), conn, net.factory, clock.sleep, clock.now, cfg
    )
    return c, clock


def run(c, chain=None, **kw):
    return c.complete_json("default", "sys", "user", SCHEMA, chain=chain, **kw)


def rl(retry_after=None):
    h = {"retry-after": str(retry_after)} if retry_after is not None else {}
    return httpx2.Response(429, json={"error": {"message": "rate limit"}}, headers=h)


def test_split_spec_first_slash_only():
    assert split_spec(G) == ("groq", "openai/gpt-oss-120b")
    assert split_spec("ollama/qwen3:4b") == ("ollama", "qwen3:4b")
    assert split_spec("openrouter/openai/gpt-oss-20b:free") == ("openrouter", "openai/gpt-oss-20b:free")
    with pytest.raises(ValueError):
        split_spec("nomodel")


def test_success_records_tokens_and_caches(settings):
    net = FakeNet({"api.groq.com": ['{"n": 7}']})
    c, _ = make(settings, net)
    r = run(c, [G])
    assert r.parsed == {"n": 7} and not r.cached and r.spec == G
    assert (r.input_tokens, r.output_tokens) == (10, 5)
    led = c.ledger[("groq", "openai/gpt-oss-120b")]
    assert led["calls"] == 1 and led["input_tokens"] == 10 and led["output_tokens"] == 5
    r2 = run(c, [G])
    assert r2.cached and r2.parsed == {"n": 7}
    assert len(net.requests) == 1  # cache hit made no HTTP call
    assert c.ledger[("groq", "openai/gpt-oss-120b")]["calls"] == 1


def test_request_shape_uses_json_schema_strict_on_groq_gpt_oss(settings):
    net = FakeNet({"api.groq.com": ['{"n": 1}']})
    c, _ = make(settings, net)
    run(c, [G], max_tokens=33, temperature=0.2)
    body = json.loads(net.requests[0].content)
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["max_tokens"] == 33 and body["temperature"] == 0.2 and body["model"] == "openai/gpt-oss-120b"


def test_cache_key_separates_parameters(settings):
    net = FakeNet({"api.groq.com": ['{"n": 1}']})
    c, _ = make(settings, net)
    run(c, [G], max_tokens=10)
    run(c, [G], max_tokens=11)
    assert len(net.requests) == 2


def test_use_cache_false_always_calls(settings):
    net = FakeNet({"api.groq.com": ['{"n": 1}']})
    c, _ = make(settings, net)
    run(c, [G], use_cache=False)
    run(c, [G], use_cache=False)
    assert len(net.requests) == 2


def test_paid_spec_raises_at_stage0_even_with_key(settings):
    net = FakeNet({})
    c, _ = make(settings, net)
    c.settings = c.settings.model_copy(update={"openai_api_key": "sk-fake"})
    for spec in ("anthropic/claude-x", "openai/gpt-x"):
        with pytest.raises(SpendStageError):
            run(c, [spec])
    with pytest.raises(SpendStageError):  # a paid spec later in the chain still trips the gate
        run(c, [G, "anthropic/claude-x"])
    assert net.requests == []


def test_paid_spec_not_gated_at_stage2_but_no_key_means_exhausted(settings):
    net = FakeNet({})
    c, _ = make(settings, net, stage=2)
    with pytest.raises(ChainExhaustedError) as ei:
        run(c, ["anthropic/claude-x"])
    assert ei.value.attempts[0]["outcome"] == "no_key"


def test_missing_key_skipped(settings):
    net = FakeNet({"api.groq.com": ['{"n": 2}']})
    c, _ = make(settings, net, keys=("groq",))
    r = run(c, [GEM, G])
    assert r.provider == "groq"
    assert r.attempts[0] == {"spec": GEM, "outcome": "no_key"}
    assert all(req.url.host == "api.groq.com" for req in net.requests)


def test_429_falls_back_and_bumps_ledger(settings):
    net = FakeNet({"generativelanguage.googleapis.com": [rl()], "api.groq.com": ['{"n": 3}']})
    c, _ = make(settings, net)
    r = run(c, [GEM, G])
    assert r.provider == "groq" and r.parsed == {"n": 3}
    assert [a["outcome"] for a in r.attempts] == ["rate_limited", "ok"]
    gem = c.ledger[("gemini", "gemini-3.5-flash-lite")]
    assert gem["rate_limited"] == 1 and gem["fallbacks"] == 1


def test_5xx_falls_back_and_counts_error(settings):
    net = FakeNet(
        {"api.groq.com": [httpx2.Response(503, json={"error": "down"})], "api.cerebras.ai": ['{"n": 4}']}
    )
    c, _ = make(settings, net)
    r = run(c, [G, CER])
    assert r.provider == "cerebras"
    assert c.ledger[("groq", "openai/gpt-oss-120b")]["errors"] == 1


def test_quota_429_falls_back_without_waiting(settings):
    quota = httpx2.Response(
        429, json={"error": {"message": "You exceeded your current quota"}}, headers={"retry-after": "5"}
    )
    net = FakeNet({"api.groq.com": [quota], "api.cerebras.ai": ['{"n": 4}']})
    c, clock = make(settings, net)
    r = run(c, [G, CER])
    assert r.attempts[0]["outcome"] == "quota" and clock.sleeps == []
    assert c.ledger[("groq", "openai/gpt-oss-120b")]["errors"] == 1


def test_retry_after_honoured_via_injected_sleep(settings):
    net = FakeNet({"api.groq.com": [rl(7), '{"n": 5}']})
    c, clock = make(settings, net)
    r = run(c, [G])
    assert r.parsed == {"n": 5}
    assert 7 in clock.sleeps
    assert c.ledger[("groq", "openai/gpt-oss-120b")]["rate_limited"] == 1


def test_long_retry_after_falls_back_instead_of_sleeping(settings):
    net = FakeNet({"api.groq.com": [rl(600)], "api.cerebras.ai": ['{"n": 4}']})
    c, clock = make(settings, net)
    r = run(c, [G, CER])
    assert r.provider == "cerebras" and 600 not in clock.sleeps


def test_invalid_json_retries_once_then_falls_back(settings):
    net = FakeNet({"api.groq.com": ["not json"], "api.cerebras.ai": ['{"n": 9}']})
    c, _ = make(settings, net)
    r = run(c, [G, CER])
    assert r.provider == "cerebras"
    assert len([q for q in net.requests if q.url.host == "api.groq.com"]) == 2
    led = c.ledger[("groq", "openai/gpt-oss-120b")]
    assert led["schema_failures"] == 2 and led["fallbacks"] == 1


def test_schema_violation_retries_then_succeeds(settings):
    net = FakeNet({"api.groq.com": ['{"n": "x"}', '{"n": 6}']})
    c, _ = make(settings, net)
    r = run(c, [G])
    assert r.parsed == {"n": 6} and len(net.requests) == 2
    assert c.ledger[("groq", "openai/gpt-oss-120b")]["schema_failures"] == 1


def test_code_fenced_json_accepted(settings):
    net = FakeNet({"api.groq.com": ['```json\n{"n": 1}\n```']})
    c, _ = make(settings, net)
    assert run(c, [G]).parsed == {"n": 1}


def test_one_element_chain_never_falls_back(settings):
    net = FakeNet({"api.groq.com": [httpx2.Response(500, json={})]})
    c, _ = make(settings, net)
    with pytest.raises(ChainExhaustedError) as ei:
        run(c, [G])
    assert [a["spec"] for a in ei.value.attempts] == [G]
    assert c.ledger[("groq", "openai/gpt-oss-120b")]["fallbacks"] == 0
    assert all(req.url.host == "api.groq.com" for req in net.requests)


def test_failed_responses_are_not_cached(settings):
    net = FakeNet({"api.groq.com": ["bad", "bad", '{"n": 1}']})
    c, _ = make(settings, net)
    with pytest.raises(ChainExhaustedError):
        run(c, [G])
    assert run(c, [G]).parsed == {"n": 1}


def test_rpd_exhaustion_skips_provider(settings):
    net = FakeNet({"api.groq.com": ['{"n": 1}'], "api.cerebras.ai": ['{"n": 2}']})
    c, _ = make(settings, net, limits={"groq": {"rpd": 1}})
    run(c, [G, CER], use_cache=False)
    r = run(c, [G, CER], use_cache=False)
    assert r.provider == "cerebras"
    assert r.attempts[0]["outcome"] == "rpd_exhausted"
    assert len([q for q in net.requests if q.url.host == "api.groq.com"]) == 1


def test_tpd_exhaustion_skips_provider(settings):
    net = FakeNet({"api.groq.com": ['{"n": 1}'], "api.cerebras.ai": ['{"n": 2}']})
    c, _ = make(settings, net, limits={"groq": {"tpd": 15}})
    run(c, [G, CER], use_cache=False)  # 15 tokens used
    assert run(c, [G, CER], use_cache=False).attempts[0]["outcome"] == "tpd_exhausted"


def test_rpm_limiter_spaces_calls(settings):
    net = FakeNet({"api.groq.com": ['{"n": 1}']})
    c, clock = make(settings, net, limits={"groq": {"rpm": 6}})
    run(c, [G], use_cache=False)
    run(c, [G], use_cache=False)
    assert sum(clock.sleeps) == pytest.approx(10.0)


def test_unknown_provider_is_an_error(settings):
    c, _ = make(settings, FakeNet({}))
    from lacuna.llm import LLMError

    with pytest.raises(LLMError):
        run(c, ["nope/model"])


def test_role_chain_from_config(settings):
    net = FakeNet({"generativelanguage.googleapis.com": ['{"n": 1}']})
    c, _ = make(settings, net)
    assert run(c).provider == "gemini"


def test_shipped_config_loads_and_parses():
    from lacuna.llm.config import load_config

    cfg = load_config()
    for role in ("default", "generator", "judge", "screener"):
        assert [split_spec(s)[0] for s in cfg.chain_for(role)] == [
            "gemini",
            "groq",
            "cerebras",
            "openrouter",
            "ollama",
        ]
    # The extractor skips Cerebras (trial credit needs a card) and adds a second Groq model instead.
    providers = [split_spec(s)[0] for s in cfg.chain_for("extractor")]
    assert providers[0] == "gemini" and providers[-1] == "ollama" and "cerebras" not in providers
    # Blueprint section 0: no paid provider may appear in any shipped chain.
    for role in ("default", "generator", "judge", "extractor", "screener"):
        assert not {split_spec(s)[0] for s in cfg.chain_for(role)} & {"anthropic", "openai"}


def test_module_level_complete_json_gates_paid_before_postgres(settings):
    from lacuna.llm import complete_json

    with pytest.raises(SpendStageError):
        complete_json("default", "s", "u", SCHEMA, chain=["anthropic/claude-x"])


@pytest.mark.live
def test_status_cli_against_postgres(pg_conn, capsys):
    from lacuna.cli import main

    main(["llm", "status"])
    out = capsys.readouterr().out
    assert "groq" in out and "anthropic" in out and "spend_stage=" in out
    assert "usage unavailable" not in out


@pytest.mark.live
def test_postgres_ledger_roundtrip(settings, pg_conn):
    from lacuna.db import usage_today

    model = "ledger-test-model"
    pg_conn.execute("DELETE FROM provider_usage WHERE provider = 'groq' AND model = %s", (model,))
    net = FakeNet({"api.groq.com": ['{"n": 1}']})
    c, _ = make(settings, net, conn=pg_conn)
    run(c, [f"groq/{model}"])
    u = usage_today(pg_conn, "provider_usage", provider="groq", model=model)
    assert u["calls"] == 1 and u["input_tokens"] == 10
    pg_conn.execute("DELETE FROM provider_usage WHERE provider = 'groq' AND model = %s", (model,))
