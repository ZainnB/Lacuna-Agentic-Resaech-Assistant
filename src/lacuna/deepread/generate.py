"""Frozen generator: same model, prompt template and settings for every phase.

Only retrieval should change between phases, so treat PROMPT_VERSION, SYSTEM_PROMPT and the
[generator] config as frozen. Both feed the run's config hash. Re-frozen once (v2) to a
free-tier model before the first baseline; see config/deepread.toml and docs/decisions.md.
"""

from lacuna.llm import LLMError, complete_json

PROMPT_VERSION = "v2"

SYSTEM_PROMPT = """You answer questions about research papers.

Answer only from the provided context excerpts. Do not use outside knowledge.
Give the shortest answer that fully answers the question: a number, a name or a short phrase. \
For numeric answers, give the number exactly as it appears in the paper, without units.
Cite every excerpt you used as [paper_id, page].
If the context does not contain the answer, set the answer to exactly "not found" and cite nothing."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"paper_id": {"type": "string"}, "page": {"type": "integer"}},
                "required": ["paper_id", "page"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["answer", "citations"],
    "additionalProperties": False,
}


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{c['paper_id']}, {c['page']}]\n{c['text']}" for c in chunks)


def call_json(system: str, user: str, schema: dict, cfg: dict, role: str = "generator") -> dict:
    """One structured-output call through the provider layer.

    Returns the parsed JSON plus usage and model metadata: parsed, model, provider, stop_reason,
    input_tokens, output_tokens, cached. The chain has exactly one element and there is no
    fallback on purpose: a silent switch to another model would break the frozen-generator
    comparison, so a provider failure is recorded as parsed=None with stop_reason="error".
    """
    spec = cfg["model"]
    try:
        r = complete_json(
            role,
            system,
            user,
            schema,
            chain=[spec],
            max_tokens=cfg["max_tokens"],
            temperature=0.0,
        )
    except LLMError as e:
        provider, _, model = spec.partition("/")
        return {
            "parsed": None,
            "model": model,
            "provider": provider,
            "stop_reason": "error",
            "input_tokens": 0,
            "output_tokens": 0,
            "cached": False,
            "error": f"{type(e).__name__}: {e}",
        }
    return {
        "parsed": r.parsed,
        "model": r.model,
        "provider": r.provider,
        "stop_reason": r.stop_reason,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "cached": r.cached,
    }


def generate(question: str, chunks: list[dict], cfg: dict) -> dict:
    user = f"<context>\n{format_context(chunks)}\n</context>\n\nQuestion: {question}"
    result = call_json(SYSTEM_PROMPT, user, ANSWER_SCHEMA, cfg)
    parsed = result.pop("parsed") or {}
    return {"answer": parsed.get("answer"), "citations": parsed.get("citations", []), **result}
