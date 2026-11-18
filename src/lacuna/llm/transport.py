"""One chat-completions call over the `openai` SDK, with errors classified for the chain.

openai 3.x runs on `httpx2` (not `httpx`): pass `http_client=httpx2.Client(...)`, or
`openai.DefaultHttpxClient()` for the default. SDK retries are off; retry policy lives in the chain.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import openai

from lacuna.llm.providers import ProviderSpec
from lacuna.settings import Settings

RATE_LIMITED = "rate_limited"
QUOTA = "quota"
AUTH = "auth"
OTHER = "other"


class TransportError(Exception):
    def __init__(self, kind: str, message: str, status: int | None = None, retry_after: float | None = None):
        super().__init__(f"{kind}: {message}")
        self.kind = kind
        self.status = status
        self.retry_after = retry_after


@dataclass
class RawResponse:
    text: str | None
    input_tokens: int
    output_tokens: int
    finish_reason: str | None
    latency_ms: int


ClientFactory = Callable[[ProviderSpec, Settings], openai.OpenAI]


def default_client_factory(spec: ProviderSpec, settings: Settings) -> openai.OpenAI:
    return openai.OpenAI(
        base_url=spec.base(settings),
        api_key=spec.api_key(settings) or "none",  # SDK insists on a non-empty key; Ollama ignores it
        http_client=openai.DefaultHttpxClient(),
        max_retries=0,
        timeout=90.0,
    )


def _retry_after(e: openai.APIStatusError) -> float | None:
    raw = e.response.headers.get("retry-after")
    try:
        return max(0.0, float(raw)) if raw is not None else None
    except ValueError:
        return None


def classify(e: Exception) -> TransportError:
    if isinstance(e, openai.APIStatusError):
        status = e.status_code
        msg = str(e)[:300]
        low = msg.lower()
        if status == 429:
            if "insufficient_quota" in low or "exceeded your current quota" in low or "per day" in low:
                return TransportError(QUOTA, msg, status)
            return TransportError(RATE_LIMITED, msg, status, _retry_after(e))
        if status in (401, 403):
            return TransportError(AUTH, msg, status)
        return TransportError(OTHER, msg, status)
    return TransportError(OTHER, f"{type(e).__name__}: {e}"[:300])  # connection errors, timeouts


def chat_json(
    client: openai.OpenAI,
    spec: ProviderSpec,
    model: str,
    system: str,
    user: str,
    schema: dict,
    *,
    max_tokens: int,
    temperature: float,
) -> RawResponse:
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "response", "schema": schema, "strict": spec.strict_for(model)},
    }
    t0 = time.monotonic()
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format=response_format,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as e:  # noqa: BLE001
        raise classify(e) from e
    latency = int((time.monotonic() - t0) * 1000)
    choice = r.choices[0] if r.choices else None
    usage = r.usage
    return RawResponse(
        text=choice.message.content if choice else None,
        input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        finish_reason=choice.finish_reason if choice else None,
        latency_ms=latency,
    )
