"""Provider table: base URLs, key lookup, paid flag and default free-tier limits.

Limits are enforced in code (blueprint section 0). A limit of 0 means "not enforced here".
Free-tier numbers drift; override them in config/llm.toml [limits.<provider>] when they change.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from lacuna.settings import Settings


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    base_url: str
    key_attr: str | None  # Settings attribute holding the key; None = no key needed
    paid: bool = False
    rpm: int = 0
    rpd: int = 0
    tpm: int = 0
    tpd: int = 0
    strict_prefixes: tuple[str, ...] = ()  # model-id prefixes where json_schema strict=True is safe

    def api_key(self, settings: Settings) -> str | None:
        return getattr(settings, self.key_attr, None) if self.key_attr else None

    def available(self, settings: Settings) -> bool:
        if self.paid:  # placeholders: no Settings key exists yet, so never callable (Stage 2 work)
            return False
        return self.key_attr is None or bool(self.api_key(settings))

    def base(self, settings: Settings) -> str:
        return settings.ollama_base_url if self.name == "ollama" else self.base_url

    def strict_for(self, model: str) -> bool:
        return any(model.startswith(p) for p in self.strict_prefixes)


# Default limits, with sources (checked 2026-11-14):
#   groq: 30 rpm / 1000 rpd / 8000 tpm / 200000 tpd for openai/gpt-oss-120b and -20b. Source
#         console.groq.com/docs/rate-limits, but the page labels this table "Developer plan" and
#         gives no Free-plan table, so the free-plan values are UNVERIFIED. Per-org limits are in the
#         Groq console. Kept as conservative defaults.
#   gemini: ~15 rpm / 1000 rpd for Flash-Lite-class free tier. The public rate-limit docs
#         (aistudio.google.com/docs/rate-limits) no longer list per-model free numbers; they point
#         to the AI Studio quota page (aistudio.google.com/rate-limit). UNVERIFIED.
#   cerebras: free trial gpt-oss-120b is 5 rpm / 30K uncached tpm / 1M tokens per hour and per day
#         (inference-docs.cerebras.ai/support/rate-limits, 2026-11-14). VERIFIED. No daily request
#         cap is published, so rpd is left unenforced. (The older 30 rpm / 14400 rpd is gone.)
#   openrouter: 20 rpm / 50 rpd on ":free" models without credit (OpenRouter docs, as remembered
#         2026-11-13; UNVERIFIED). 1000 rpd after a $10 top-up; raise it in llm.toml if that applies.
#   ollama: local, unlimited.
PROVIDERS: dict[str, ProviderSpec] = {
    p.name: p
    for p in (
        ProviderSpec(
            "gemini",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
            "gemini_api_key",
            rpm=15,
            rpd=1000,
        ),
        ProviderSpec(
            "groq",
            "https://api.groq.com/openai/v1",
            "groq_api_key",
            rpm=30,
            rpd=1000,
            tpm=8000,
            tpd=200000,
            strict_prefixes=("openai/gpt-oss",),
        ),
        ProviderSpec(
            "cerebras",
            "https://api.cerebras.ai/v1",
            "cerebras_api_key",
            rpm=5,
            tpm=30000,
            tpd=1000000,
        ),
        ProviderSpec("openrouter", "https://openrouter.ai/api/v1", "openrouter_api_key", rpm=20, rpd=50),
        ProviderSpec("ollama", "http://localhost:11434/v1", None),
        # Paid placeholders: the spend gate only. No client is ever built for these before Stage 2.
        ProviderSpec("anthropic", "https://api.anthropic.com/v1/", None, paid=True),
        ProviderSpec("openai", "https://api.openai.com/v1", None, paid=True),
    )
}


def split_spec(spec: str) -> tuple[str, str]:
    """'groq/openai/gpt-oss-120b' -> ('groq', 'openai/gpt-oss-120b'). Splits on the first slash."""
    provider, sep, model = spec.partition("/")
    if not sep or not provider or not model:
        raise ValueError(f"model spec must be '<provider>/<model>': {spec!r}")
    return provider, model


def with_overrides(spec: ProviderSpec, overrides: dict | None) -> ProviderSpec:
    if not overrides:
        return spec
    allowed = {k: int(v) for k, v in overrides.items() if k in ("rpm", "rpd", "tpm", "tpd")}
    return replace(spec, **allowed)
