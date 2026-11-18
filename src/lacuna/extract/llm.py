"""Server-driven extraction (Mode B): paper_chunks -> batches -> complete_json -> validated Extraction.

Guarantees enforced in code, not in the prompt:
  * a model's item is kept only if its chunk_id is one of the chunks in the batch it was shown
    (`rejected_citations` counts the rest);
  * the merged result is cached in `extraction_cache` keyed by (paper, chunk content, prompt version,
    extractor), so a re-run spends nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import get_args

from pydantic import ValidationError

from lacuna.extract import prompts, store
from lacuna.extract.schemas import Extraction, extraction_json_schema
from lacuna.llm import LLMError, SpendStageError, complete_json

INPUT_TOKEN_BUDGET = 6000
# gpt-oss counts reasoning tokens against max_tokens: 4096 ran out live on a 6-chunk batch. The chain
# books input/3 + max_tokens against Groq's 8k TPM: 4 full-text chunks + 5500 came to ~8.3k and Groq
# skipped every full-text batch. 3 chunks + 5000 is ~7.3k.
OUTPUT_TOKENS = 5000


class NoChunksError(LookupError):
    """The paper has no rows in paper_chunks."""


@dataclass
class ExtractResult:
    extraction: Extraction
    extracted_by: str
    cached: bool = False
    rejected_citations: int = 0
    calls: int = 0  # LLM calls attempted (cache hits inside the LLM layer included)
    tokens: dict = field(default_factory=lambda: {"input": 0, "output": 0})
    failed_calls: int = 0  # batches with no valid answer (strict=False only)
    invalid_items: int = 0  # items dropped one by one for failing the Pydantic model
    failures: list[str] = field(default_factory=list)  # why each failed batch failed (strict=False)
    latencies_ms: list[int] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)


def validate_items(raw: dict) -> tuple[Extraction, int]:
    """Validate item by item and drop the invalid ones.

    Strict JSON schema can't express rules like non-empty strings, so a schema-valid answer can
    still hold an item Pydantic rejects (seen live: a result with dataset ""). One bad item must
    not discard the rest of the paper.
    """
    if not isinstance(raw, dict):
        raise ValueError("extraction is not a JSON object")
    kept: dict[str, list] = {}
    invalid = 0
    for name, field_info in Extraction.model_fields.items():
        (item_model,) = get_args(field_info.annotation)
        kept[name] = []
        for item in raw.get(name) or []:
            try:
                kept[name].append(item_model.model_validate(item))
            except ValidationError:
                invalid += 1
    return Extraction(**kept), invalid


def estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1


def make_batches(
    chunks: list[dict], *, max_chunks: int, token_budget: int = INPUT_TOKEN_BUDGET
) -> list[list[dict]]:
    """Order-preserving batches of at most `max_chunks` chunks and ~`token_budget` input tokens.

    A single chunk larger than the budget still gets its own batch.
    """
    batches: list[list[dict]] = []
    cur: list[dict] = []
    used = 0
    for c in chunks:
        t = estimate_tokens(c["text"]) + 20
        if cur and (len(cur) >= max_chunks or used + t > token_budget):
            batches.append(cur)
            cur, used = [], 0
        cur.append(c)
        used += t
    if cur:
        batches.append(cur)
    return batches


def filter_citations(extraction: Extraction, allowed: set[str]) -> tuple[Extraction, int]:
    """Drop items citing a chunk outside `allowed`. Returns (clean extraction, number dropped)."""
    kept, dropped = {}, 0
    for k in Extraction.LISTS:
        items = getattr(extraction, k)
        good = [i for i in items if i.chunk_id in allowed]
        dropped += len(items) - len(good)
        kept[k] = good
    return Extraction(**kept), dropped


def merge(parts: list[Extraction]) -> Extraction:
    """Concatenate batches, dropping exact duplicates."""
    out = {}
    for k in Extraction.LISTS:
        seen, items = set(), []
        for part in parts:
            for i in getattr(part, k):
                key = i.model_dump_json()
                if key not in seen:
                    seen.add(key)
                    items.append(i)
        out[k] = items
    return Extraction(**out)


def extract_paper(
    conn,
    paper_id: str,
    *,
    chain: list[str] | None = None,
    max_chunks_per_call: int = 3,
    use_cache: bool = True,
    strict: bool = True,
) -> ExtractResult:
    """Extract one paper from its stored chunks.

    `use_cache=False` bypasses the extraction cache and the LLM response cache (throughput runs) and
    writes nothing to the extraction cache. `strict=False` records a failed batch instead of
    raising `LLMError` (SpendStageError always raises); a partial result is never cached.
    """
    chunks = store.load_chunks(conn, paper_id)
    if not chunks:
        raise NoChunksError(f"no paper_chunks for {paper_id}")
    chash = store.content_hash(chunks)
    chunk_ids = [c["chunk_id"] for c in chunks]

    if use_cache:
        names = [f"api:{s}" for s in chain] if chain else None
        hit = store.cache_get(conn, paper_id, chash, names)
        if hit is not None and (names is not None or hit[1].startswith("api:")):
            return ExtractResult(hit[0], hit[1], cached=True, chunk_ids=chunk_ids)

    schema = extraction_json_schema()
    res = ExtractResult(Extraction(), "", chunk_ids=chunk_ids)
    parts: list[Extraction] = []
    specs: list[str] = []
    for batch in make_batches(chunks, max_chunks=max_chunks_per_call):
        allowed = {c["chunk_id"] for c in batch}
        res.calls += 1
        try:
            r = complete_json(
                role="extractor",
                system=prompts.SYSTEM,
                user=prompts.build_user(batch),
                schema=schema,
                chain=chain,
                max_tokens=OUTPUT_TOKENS,
                use_cache=use_cache,
            )
            parsed, invalid = validate_items(r.parsed)
        except SpendStageError:
            raise
        except (LLMError, ValidationError, ValueError) as e:
            if strict:
                raise
            res.failed_calls += 1
            res.failures.append(f"{type(e).__name__}: {e}"[:300])
            continue
        res.invalid_items += invalid
        clean, dropped = filter_citations(parsed, allowed)
        res.rejected_citations += dropped
        parts.append(clean)
        if r.spec not in specs:
            specs.append(r.spec)
        res.tokens["input"] += r.input_tokens
        res.tokens["output"] += r.output_tokens
        if r.latency_ms is not None and not r.cached:
            res.latencies_ms.append(r.latency_ms)

    res.extraction = merge(parts)
    if specs:
        res.extracted_by = "api:" + "+".join(specs)
    elif chain:
        res.extracted_by = f"api:{chain[0]}"
    else:
        res.extracted_by = "api:unknown"
    if use_cache and res.failed_calls == 0 and specs:
        store.cache_put(conn, paper_id, chash, res.extracted_by, res.extraction)
    return res
