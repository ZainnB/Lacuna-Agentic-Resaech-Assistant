"""Provider layer: one call signature in front of a free-tier fallback chain.

Contract (stable; other packages code against this):

    result = complete_json(
        role="generator",                 # picks the default chain from llm config
        system="...", user="...",
        schema={...},                     # JSON schema; the parsed object is validated against it
        chain=["groq/openai/gpt-oss-120b"],   # optional explicit chain; overrides the role default
        max_tokens=1024, temperature=0.0,
        use_cache=True,
    )

Model specs are "<provider>/<model id>", split on the first slash only, e.g.
"gemini/gemini-3.5-flash-lite", "groq/openai/gpt-oss-120b", "ollama/qwen3:4b".

Guarantees:
  * Providers marked paid raise `SpendStageError` while settings.spend_stage < 2. Never bypassed.
  * Every response is cached by (provider, model, prompt hash) in `llm_cache`; a cache hit makes
    no network call and sets `result.cached = True`.
  * On 429, quota exhaustion or a provider error the next spec in the chain is tried; each try is
    appended to `result.attempts`. A one-element chain never falls back.
  * Invalid JSON or schema failure retries once on the same provider, then falls back.
  * Each call bumps `provider_usage` (calls, tokens, errors, rate_limited, fallbacks).
"""

from __future__ import annotations

from dataclasses import dataclass, field


class LLMError(Exception):
    """Base class for provider-layer failures."""


class SpendStageError(LLMError):
    """A paid provider was requested below spend stage 2."""


class ChainExhaustedError(LLMError):
    """Every provider in the chain failed; see `attempts`."""

    def __init__(self, attempts: list[dict]):
        super().__init__(f"all providers failed: {attempts}")
        self.attempts = attempts


@dataclass
class LLMResult:
    parsed: dict | None
    text: str | None
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    stop_reason: str | None = None
    latency_ms: int | None = None
    attempts: list[dict] = field(default_factory=list)

    @property
    def spec(self) -> str:
        return f"{self.provider}/{self.model}"


def complete_json(
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
    # Implemented in lacuna.llm.chain; imported lazily so this module stays import-light.
    from lacuna.llm.chain import complete_json as _impl

    return _impl(
        role,
        system,
        user,
        schema,
        chain=chain,
        max_tokens=max_tokens,
        temperature=temperature,
        use_cache=use_cache,
    )


__all__ = ["ChainExhaustedError", "LLMError", "LLMResult", "SpendStageError", "complete_json"]
