"""Fallback chain: spend gate, cache, quotas, rate limits, schema validation, ledger."""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict, deque

import jsonschema

from lacuna.cache import Cache, cache_key
from lacuna.db import bump_usage, quota_day, usage_today
from lacuna.llm import ChainExhaustedError, LLMError, LLMResult, SpendStageError
from lacuna.llm.config import LLMConfig, load_config
from lacuna.llm.providers import PROVIDERS, ProviderSpec, split_spec, with_overrides
from lacuna.llm.transport import RATE_LIMITED, TransportError, chat_json, default_client_factory
from lacuna.ratelimit import RateLimiter
from lacuna.settings import Settings

MAX_RETRY_AFTER_WAIT = 30.0  # seconds; a longer Retry-After falls back instead of blocking
SCHEMA_TRIES = 2  # first try + one retry on the same provider
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_LEDGER_FIELDS = (
    "calls",
    "input_tokens",
    "output_tokens",
    "errors",
    "rate_limited",
    "fallbacks",
    "schema_failures",
)


def parse_json(text: str | None):
    """Parse model output, tolerating <think> blocks and code fences. Raises ValueError."""
    if not text or not text.strip():
        raise ValueError("empty response")
    t = _THINK.sub("", text).strip()
    m = _FENCE.match(t)
    if m:
        t = m.group(1)
    try:
        return json.loads(t)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON: {e}") from e


class Chain:
    def __init__(
        self,
        settings: Settings,
        cache: Cache,
        conn=None,
        client_factory=default_client_factory,
        sleep=time.sleep,
        clock=time.monotonic,
        config: LLMConfig | None = None,
    ):
        self.settings = settings
        self.cache = cache
        self.conn = conn
        self.client_factory = client_factory
        self.sleep = sleep
        self.clock = clock
        self.config = config or load_config()
        # In-memory ledger when there is no Postgres: {(provider, model): {field: n}}
        self.ledger: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: dict.fromkeys(_LEDGER_FIELDS, 0)
        )
        self._limiters: dict[str, RateLimiter] = {}
        self._tpm_window: dict[str, deque] = defaultdict(deque)
        self._clients: dict[str, object] = {}

    # ------------------------------------------------------------------ ledger

    def _usage(self, provider: str, model: str) -> dict:
        if self.conn is None:
            return dict(self.ledger[(provider, model)])
        return usage_today(self.conn, "provider_usage", provider=provider, model=model)

    def _bump(self, provider: str, model: str, **inc) -> None:
        inc = {k: v for k, v in inc.items() if v}
        if not inc:
            return
        if self.conn is None:
            for k, v in inc.items():
                self.ledger[(provider, model)][k] += v
        else:
            bump_usage(
                self.conn,
                "provider_usage",
                {"day": quota_day(provider), "provider": provider, "model": model},
                **inc,
            )

    # ------------------------------------------------------------------ helpers

    def _spec(self, name: str) -> ProviderSpec:
        if name not in PROVIDERS:
            raise LLMError(f"unknown provider {name!r}; known: {sorted(PROVIDERS)}")
        return with_overrides(PROVIDERS[name], self.config.limits.get(name))

    def _limiter(self, spec: ProviderSpec) -> RateLimiter | None:
        if not spec.rpm:
            return None
        if spec.name not in self._limiters:
            self._limiters[spec.name] = RateLimiter(
                spec.rpm / 60.0, burst=1, clock=self.clock, sleep=self.sleep
            )
        return self._limiters[spec.name]

    def _wait_tpm(self, spec: ProviderSpec, estimate: int) -> None:
        """Block until the last-60s token total plus `estimate` fits under tpm."""
        win = self._tpm_window[spec.name]
        while True:
            now = self.clock()
            while win and now - win[0][0] >= 60:
                win.popleft()
            if sum(t for _, t in win) + estimate <= spec.tpm or not win:
                return
            self.sleep(max(0.1, 60 - (now - win[0][0])))

    def _client(self, spec: ProviderSpec):
        if spec.name not in self._clients:
            self._clients[spec.name] = self.client_factory(spec, self.settings)
        return self._clients[spec.name]

    # ------------------------------------------------------------------ main entry

    def complete_json(
        self,
        role: str,
        system: str,
        user: str,
        schema: dict,
        *,
        chain: list[str] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        use_cache: bool = True,
    ) -> LLMResult:
        specs = list(chain) if chain else self.config.chain_for(role)
        parsed_specs = [split_spec(s) for s in specs]
        pspecs = [self._spec(p) for p, _ in parsed_specs]

        # 1. Spend gate: before the cache, before any client exists, for every spec in the chain.
        for ps, s in zip(pspecs, specs, strict=True):
            if ps.paid and self.settings.spend_stage < 2:
                raise SpendStageError(f"{s} is a paid provider; spend_stage={self.settings.spend_stage} < 2")

        keys = [
            cache_key("llm", p, m, system, user, schema, max_tokens, temperature) for p, m in parsed_specs
        ]

        # 2. Cache: any spec in the chain with a stored answer wins, in chain order. No network.
        if use_cache:
            for (provider, model), key in zip(parsed_specs, keys, strict=True):
                hit = self.cache.get(key)
                if hit is not None:
                    return LLMResult(
                        parsed=hit.get("parsed"),
                        text=hit.get("text"),
                        provider=provider,
                        model=model,
                        input_tokens=hit.get("input_tokens", 0),
                        output_tokens=hit.get("output_tokens", 0),
                        cached=True,
                        stop_reason=hit.get("stop_reason"),
                        latency_ms=hit.get("latency_ms"),
                        attempts=[{"spec": f"{provider}/{model}", "outcome": "cache"}],
                    )

        attempts: list[dict] = []
        n = len(specs)
        for i, ((provider, model), ps, key) in enumerate(zip(parsed_specs, pspecs, keys, strict=True)):
            more = i < n - 1
            outcome = self._try_provider(
                ps, provider, model, system, user, schema, max_tokens, temperature, attempts
            )
            if isinstance(outcome, LLMResult):
                outcome.attempts = attempts
                if use_cache:
                    self.cache.set(
                        key,
                        "llm",
                        {
                            "parsed": outcome.parsed,
                            "text": outcome.text,
                            "input_tokens": outcome.input_tokens,
                            "output_tokens": outcome.output_tokens,
                            "stop_reason": outcome.stop_reason,
                            "latency_ms": outcome.latency_ms,
                        },
                    )
                return outcome
            if more and outcome != "no_key":
                self._bump(provider, model, fallbacks=1)
        raise ChainExhaustedError(attempts)

    def _try_provider(self, ps, provider, model, system, user, schema, max_tokens, temperature, attempts):
        """Returns an LLMResult, or the failure reason string (also appended to `attempts`)."""
        spec = f"{provider}/{model}"

        def fail(outcome: str, **extra) -> str:
            attempts.append({"spec": spec, "outcome": outcome, **extra})
            return outcome

        if not ps.available(self.settings):
            return fail("no_key")
        use = self._usage(provider, model)
        if ps.rpd and use["calls"] >= ps.rpd:
            return fail("rpd_exhausted", used=use["calls"], limit=ps.rpd)
        if ps.tpd and use["input_tokens"] + use["output_tokens"] >= ps.tpd:
            return fail("tpd_exhausted", limit=ps.tpd)
        estimate = (len(system) + len(user)) // 3 + max_tokens
        if ps.tpm and estimate > ps.tpm:
            return fail("exceeds_tpm", estimate=estimate, limit=ps.tpm)

        client = self._client(ps)
        limiter = self._limiter(ps)
        schema_tries = 0
        rate_retried = False
        last_detail = ""
        while True:
            if limiter:
                limiter.acquire()
            if ps.tpm:
                self._wait_tpm(ps, estimate)
            try:
                raw = chat_json(
                    client, ps, model, system, user, schema, max_tokens=max_tokens, temperature=temperature
                )
            except TransportError as e:
                self._bump(provider, model, calls=1)
                if e.kind == RATE_LIMITED:
                    self._bump(provider, model, rate_limited=1)
                    wait = e.retry_after
                    if limiter:
                        limiter.penalise(wait or 1.0)
                    if wait is not None and wait <= MAX_RETRY_AFTER_WAIT and not rate_retried:
                        rate_retried = True
                        self.sleep(wait)
                        continue
                else:
                    self._bump(provider, model, errors=1)
                return fail(e.kind, status=e.status, detail=str(e)[:200])
            self._bump(
                provider, model, calls=1, input_tokens=raw.input_tokens, output_tokens=raw.output_tokens
            )
            if ps.tpm:
                self._tpm_window[ps.name].append((self.clock(), raw.input_tokens + raw.output_tokens))
            try:
                parsed = parse_json(raw.text)
                jsonschema.validate(parsed, schema)
            except (ValueError, jsonschema.ValidationError) as e:
                self._bump(provider, model, schema_failures=1)
                last_detail = str(getattr(e, "message", e))[:200]
                schema_tries += 1
                if schema_tries < SCHEMA_TRIES:
                    continue
                return fail("schema_failure", detail=last_detail)
            attempts.append({"spec": spec, "outcome": "ok"})
            return LLMResult(
                parsed=parsed,
                text=raw.text,
                provider=provider,
                model=model,
                input_tokens=raw.input_tokens,
                output_tokens=raw.output_tokens,
                stop_reason=raw.finish_reason,
                latency_ms=raw.latency_ms,
            )


# ---------------------------------------------------------------------- module-level entry point

_default: Chain | None = None


def reset_default_chain() -> None:
    global _default
    _default = None


def get_default_chain() -> Chain:
    global _default
    if _default is None:
        from lacuna.cache import PostgresCache
        from lacuna.db import connect, migrate
        from lacuna.settings import get_settings

        settings = get_settings()
        try:
            conn = connect()
            migrate(conn)
        except Exception as e:  # noqa: BLE001
            raise LLMError(
                f"Postgres is unreachable ({settings.postgres_dsn.rsplit('@', 1)[-1]}): the LLM cache and "
                f"usage ledger need it. Run `make services` and `uv run lacuna db migrate`. ({e})"
            ) from e
        _default = Chain(settings, PostgresCache(conn, "llm_cache"), conn)
    return _default


def complete_json(
    role, system, user, schema, *, chain=None, max_tokens=1024, temperature=0.0, use_cache=True
):
    # Spend gate first so a paid spec fails the same way whether or not Postgres is up.
    from lacuna.settings import get_settings

    settings = get_settings()
    for s in chain or []:
        p, _ = split_spec(s)
        if p in PROVIDERS and PROVIDERS[p].paid and settings.spend_stage < 2:
            raise SpendStageError(f"{s} is a paid provider; spend_stage={settings.spend_stage} < 2")
    return get_default_chain().complete_json(
        role,
        system,
        user,
        schema,
        chain=chain,
        max_tokens=max_tokens,
        temperature=temperature,
        use_cache=use_cache,
    )
