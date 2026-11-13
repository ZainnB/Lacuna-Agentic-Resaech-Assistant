# Decisions

Format: date — decision. **Why.** **Given up.**

## 2026-11-13 — First usable phase = blueprint Phase 0 + Phase 1

Deliverable: `lacuna scout "<question>"` returns a ranked, cited paper set built by keyword search
plus depth-1 citation snowballing, stored in Postgres and Neo4j, and **E1** reports recall@50/100/200
against the reference lists of five CV surveys for the two $0 baselines. Phase 0's foundation
(provider layer, spend gate, caches, deepread moved in and re-frozen, compose, CI) ships underneath.
**Why.** Phase 0 alone is plumbing nobody can use; Phase 1 is the first thing that answers a real
question and the first number. **Given up.** LLM-based screening, extraction and the MCP server
(Phase 2).

## 2026-11-13 — Phase 0 and Phase 1 components built in parallel on disjoint packages

`llm/`, `deepread/`, `ingest/`, `graph/` and the E1 dataset have no dependencies on each other and
were built by separate agents at the same time. Phase 0's *Done when* is verified before the Phase 1
integration (`scout/`, E1 runner) is wired on top. **Why.** Wall-clock. **Given up.** Nothing; the
phase gate is about verified foundations, not about serial typing.

## 2026-11-13 — Provider layer on the `openai` SDK with a base URL per provider, not LiteLLM

Gemini (OpenAI-compatible endpoint), Groq, Cerebras, OpenRouter and Ollama all speak OpenAI chat
completions with `response_format: json_schema`. The cache, quotas, spend gate and fallback are our
code either way. **Why.** One light dependency, full control over retries and structured-output
handling, trivially testable with a mock transport. **Given up.** LiteLLM's breadth of providers
and its name on the CV. The interface is provider-agnostic, so LiteLLM can be slotted in as a
transport later if Mode B agents need it.

## 2026-11-13 — Postgres database renamed `figrag` → `lacuna`; volume recreated

**Why.** The project is no longer FAMM-RAG, and the old volume held only the deepread chunk index,
which rebuilds in minutes from `data/chunks.jsonl` and cached embeddings. **Given up.** Nothing.

## 2026-11-13 — Deepread generator re-frozen to `groq/openai/gpt-oss-120b`, single-provider chain

Judge: `gemini/gemini-3.5-flash-lite` (different vendor, so the judge never grades its own output;
the free quota on the Flash-Lite line is reported to be far larger than on Flash). The generator
chain has one element: a fallback to another model would silently break the frozen-generator
comparison, so a failure is recorded instead. The system prompt's "computer vision research papers"
became "research papers" in the same re-freeze (the corpus now includes Llama 2 and GPT-3).
**Why.** No paid keys (user decision, 2026-11-04). gpt-oss-120b has open weights, so the frozen
baseline stays reproducible if Groq retires it; Cohere's free tier was the best technical fit but
its terms forbid benchmarking use and its weights are closed. **Given up.** Opus-level answer
quality; retrieval metrics are unaffected and the judge is calibrated by `spotcheck`.

## 2026-11-13 — `paper_id` = OpenAlex work ID; other sources resolved via free ID lookups

Semantic Scholar and arXiv hits are `SearchHit`s until an OpenAlex DOI/arXiv lookup (free, $0)
gives them a `W…` ID. **Why.** One key across graph and Postgres (blueprint 5), and OpenAlex has
the reference lists E1 needs. **Given up.** Papers OpenAlex doesn't know are dropped from the set.

## 2026-11-13 — OpenAlex daily spend capped in code at $0.90

Counted in `api_usage.cost_usd` per day using OpenAlex's published prices (ID lookups $0, list
and filter $0.0001, search $0.001); a request that would cross the cap raises before it is sent.
**Why.** The free allowance is $1/day and resets at midnight UTC; exceeding it is impossible by
construction. **Given up.** The last 10% of the allowance, kept as margin for price drift.

## 2026-11-13 — Synchronous code until Phase 2

**Why.** No server yet; sync code is simpler to test and debug. **Given up.** Concurrency in
snowballing; the rate limiters would cap it anyway.

## 2026-11-13 — Dropped the `anthropic` and `requests` dependencies

**Why.** No paid providers before Stage 2; `httpx` serves every client. **Given up.** Nothing.
