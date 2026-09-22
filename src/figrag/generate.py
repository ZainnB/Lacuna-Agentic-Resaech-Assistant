"""Fixed generator: same model, prompt template and settings for every phase.

Only retrieval should change between phases, so treat PROMPT_VERSION, SYSTEM_PROMPT
and the [generator] config as frozen. Both feed the run's config hash.
"""
import json
from functools import lru_cache

import anthropic

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You answer questions about computer vision research papers.

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


@lru_cache(maxsize=1)
def client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def format_context(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{c['paper_id']}, {c['page']}]\n{c['text']}" for c in chunks)


def call_json(system: str, user: str, schema: dict, cfg: dict) -> dict:
    """One structured-output call. Returns the parsed JSON plus usage and model metadata.

    No refusal fallback on purpose: a silent switch to another model would break
    the fixed-generator comparison, so refusals are recorded instead.
    """
    params = {
        "model": cfg["model"],
        "max_tokens": cfg["max_tokens"],
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "output_config": {"format": {"type": "json_schema", "schema": schema}},
    }
    if "effort" in cfg:
        params["output_config"]["effort"] = cfg["effort"]
    if "temperature" in cfg:  # only for models that still accept sampling params
        params["temperature"] = cfg["temperature"]
    response = client().messages.create(**params)
    meta = {
        "model": response.model,
        "stop_reason": response.stop_reason,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    text = next((b.text for b in response.content if b.type == "text"), None)
    if response.stop_reason != "end_turn" or text is None:
        return {"parsed": None, **meta}
    return {"parsed": json.loads(text), **meta}


def generate(question: str, chunks: list[dict], cfg: dict) -> dict:
    user = f"<context>\n{format_context(chunks)}\n</context>\n\nQuestion: {question}"
    result = call_json(SYSTEM_PROMPT, user, ANSWER_SCHEMA, cfg)
    parsed = result.pop("parsed") or {}
    return {"answer": parsed.get("answer"), "citations": parsed.get("citations", []), **result}
