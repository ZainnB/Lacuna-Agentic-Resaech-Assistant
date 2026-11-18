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

## 2026-11-14 — Provider-layer behaviours not fixed by the contract

- **Spend gate covers the whole chain, before the cache.** A paid spec anywhere in the chain raises
  `SpendStageError` at stage < 2, even with a key and even if an earlier spec would have answered.
  Paid providers are placeholders with no key setting, so they are never callable yet.
- **Cache pre-pass.** A cached answer for any spec in the chain is returned before any network call,
  so a re-run never re-spends quota on an earlier provider that was failing. Only schema-valid answers
  are cached.
- **Limits are per (provider, model)**, matching the `provider_usage` primary key and how Groq/Gemini
  meter. `calls` is bumped for every network attempt (a 429 still burns provider quota); tokens only on
  responses. rpd/tpd skip a provider from today's ledger; rpm is a token bucket; tpm waits on a 60 s
  window and skips requests whose estimate (chars/3 + max_tokens) exceeds tpm outright.
- **429 handling.** A `Retry-After` of at most 30 s is slept (injectable) and retried once on the same
  provider; longer, absent, or quota-style 429s fall back. Auth and 5xx errors fall back and count as
  `errors`. SDK-level retries are off.
- **`strict: true` only for model ids starting `openai/gpt-oss` on Groq**; elsewhere `strict: false`.
- **Free-tier limits other than Groq's are unverified defaults** (see providers.py); override in
  `config/llm.toml [limits.<provider>]`. **Given up:** per-key accounting across multiple keys.

## Deepread CLI, judge role and retrieval-only baseline

**Decision.** `lacuna deepread {download,ingest,chunk,index,build,ask,check-questions,eval,spotcheck,agreement}`
lives in `lacuna.deepread.cli` (`register(sub)`). `deepread.store.connect()` keeps its own tuple-row psycopg
connection (+ `register_vector`) on `settings.postgres_dsn`; it is not `lacuna.db.connect` (dict rows). The `[db]`
section of the old config is gone. Judge calls go through `generate.call_json(..., role="judge")` with the explicit
one-element chain from `[judge].model`. `eval --retrieval-only` (`retrieval_baseline(cfg, conn)`) makes no
generator/judge/API call, writes `evals/results/e6_retrieval_<index_hash>.{json,md}` and records eval
`e6_retrieval`. `ingest.py` uses httpx (streamed, 60s timeout) instead of requests; `test_eval_subset` skips unless
both GROQ and GEMINI keys are set.

**Why.** Retrieval is the only part of E6 measurable with no keys, so it is the first number. Keeping the tuple-row
connection avoids touching `retrieve.py`/`store.py` row handling.

**Given up.** The retrieval-only run excludes unanswerable questions (no gold location), so n=34 of 41. A
`# noqa: E501` sits on the frozen faithfulness prompt rather than rewrapping it, to keep the prompt hash unchanged.

## 2026-11-14 — Ingest clients: shared cached/rate-limited HTTP layer; unauthenticated OpenAlex budget

`ingest/http.py` `CachedClient` does cache lookup, budget guard, limiter, retry (max 4, `Retry-After`
honoured, 429 penalises the limiter), then ledger and cache write. Cache hits touch neither the
network nor the ledger, and cost is booked only on a 2xx response (429s count as calls and
`rate_limited`). API keys and `mailto` are sent but kept out of the cache key, so adding a key never
invalidates the cache. Verified live (2026-11-13): the key is the `api_key` query parameter (a
bearer header also works), `per_page` max is 200, a pipe-OR filter takes at most 100 IDs, and
without a key OpenAlex still answers but shares a $0.10/day per-IP allowance. So with no key the
code caps spend at min(`OPENALEX_DAILY_BUDGET_USD`, $0.09). Our `PRICES` table is used for the
guard; responses also report `meta.cost_usd`, which we do not read. Single-work lookups follow
redirects (a merged work such as W2963403868 comes back as W2626778328). Semantic Scholar without
a key returned HTTP 429 on the first request, so S2 is best-effort until a key exists.
**Given up.** Reconciling our ledger with OpenAlex's own `x-ratelimit-remaining-usd`.

## 2026-11-14 — E1 gold excludes references published after the freeze date

The freeze is the earlier of the arXiv v1 and journal dates, but OpenAlex stores the journal
version's reference list, which can cite later work (50 of 254 for the 3DGS survey). Retrieval is
barred from post-freeze papers, so those references are unreachable by construction. They are
dropped from gold and reported per survey (`dropped`). **Why.** Otherwise recall is capped below 1
for reasons no method can affect. **Given up.** Nothing measurable; undated references are kept.

## 2026-11-14 — E1 queries are fixed before results and never edited after

The video-survey query scored 0 for both methods: OpenAlex search drifted to humanities papers on
humour and emotion. The query stays as written. **Why.** Rewriting a query after seeing its score is
tuning on the eval set. The failure is the finding: keyword search can't handle plain-language
questions, which Phase 2 query planning must fix and E1 must show. **Given up.** A prettier E1 table.

## 2026-11-14 — Scout ranking: seed links weighted 3×, log citations 0.5×

A paper linked to several seeds must outrank a classic linked once (log1p(50k)·0.5 ≈ 5.4 < 2 links).
Even so, ImageNet, ResNet and BERT rank high because almost every seed cites them. **Why.** No
relevance signal exists in Phase 1 without an LLM or embeddings; this is the honest $0 baseline.
**Given up.** Better ranking now. It comes from Phase 2 screening, measured on E1.

## 2026-11-14 — Sub-agents run on Sonnet

The first wave of agents (on the main model) hit the usage limit mid-task. The re-run used Sonnet
with briefs that point at what predecessors left on disk. **Why.** The user asked for lower token
use; well-specified, test-gated package work doesn't need the largest model. **Given up.** Some
judgement on open-ended choices, which is why the main session reviews and verifies each report.

## 2026-11-14 — MCP SDK 2.x, server package `lacuna.mcp_server`

`mcp>=2.3,<3`: in 2.x `FastMCP` became `mcp.server.mcpserver.MCPServer`. The package is
`lacuna.mcp_server`, not `lacuna.mcp`, so `import mcp` inside it can never be confused with the SDK.
**Why.** Current SDK; a name that can't shadow a dependency. **Given up.** The blueprint's `mcp/` path.

## 2026-11-14 — Topic-paper chunks live in `paper_chunks`, not deepread's `chunks`

Deepread's `chunks` table and index hash are frozen for E6. Topic papers get their own table with
the same embedding model (bge-base, 768-d) and the same chunking parameters. **Why.** Adding topic
papers to `chunks` would change E6's corpus. **Given up.** One shared table, as blueprint 7.2 drew it.

## 2026-11-14 — Full text from arXiv only; the abstract is always chunk 0

A paper with no arXiv id gets abstract-level extraction only. **Why.** arXiv PDFs are reliably
fetchable and permitted for research (blueprint 15); publisher PDFs are not. **Given up.** Full text
for journal-only papers.

## 2026-11-14 — E3 gold labels are written by the user, never generated

Templates are pre-filled with title and abstract only, never with model output. **Why.** Pre-filled
model labels would anchor the labeller toward the extractor being evaluated. **Given up.** Labelling
speed. Until labels exist, E3 reports only the per-provider schema-validity rate.

## 2026-11-14 — Topic full text reuses deepread's extract/chunk code, renumbered per paper

`lacuna.papers` imports `extract_pages` (references dropped, appendices kept, control chars stripped)
and `chunk_pages` (400/80 tokens, bge tokenizer) unchanged, then renumbers chunks `<paper_id>:<n>`
from 1 in reading order; `page` and `section` come from deepread's chunker. Re-ingest upserts and
deletes fulltext chunks beyond the new count. The approval gate is enforced inside `fetch_fulltext`
("not_approved" if no topic row has `approved`), not in the CLI. Abstract chunks skip embedding when
the sha256 of title+abstract is unchanged. **Why.** Same chunking as the frozen E6 pipeline, stable
citable ids, no way to bypass the gate. **Given up.** Per-page chunk ids (`p3:c1`), and full text for
papers without an arXiv id (only 1 of the top 3 papers of the first live topic had one).

## 2026-11-14 — E3 harness: split file, scoring conventions, verified model ids

`evals/datasets/e3/split.json` is the source of truth for tuning/reporting; new papers go to the smaller
half in hash order, so a batch is balanced and existing assignments never move. Scoring is exact match
after normalisation (the resolver's `normalise` when importable, else lowercase, no punctuation, variant
suffix stripped); precision/recall/F1 are 0 when their denominator is 0, and macro skips types absent from
both gold and prediction. A paper whose extraction raises scores as empty and is listed in `failed`.
`--scope fulltext` only selects fulltext labels; what extract_paper reads depends on the paper's chunks.
Model ids re-checked 2026-11-14: openrouter `openai/gpt-oss-20b:free` no longer exists (now
`google/gemma-4-31b-it:free`); Groq `qwen/qwen3.8-27b` is preview-only so it is in no chain; Cerebras free
trial is 5 rpm / 1M tokens per day (limits updated); Groq and Gemini free-plan numbers could not be
verified from the public pages and stay UNVERIFIED. **Given up.** Exact-match scoring misses near-synonyms
the resolver does not know.

## 2026-11-14 — Extraction: citations enforced in code; Mode A and Mode B share one schema

`lacuna.extract` validates every extraction with one Pydantic `Extraction`, and hands providers a
hand-built strict JSON schema (all properties required, nullable via `["type","null"]`, no
min/max/format). Confidence is clamped to 0-1 in Pydantic, not in the schema. An item whose
`chunk_id` is not in the batch the model saw is dropped and counted (`rejected_citations`); a host
submission citing any chunk that was not served to its session, or that belongs to another paper,
is rejected whole. The cache key is (paper, hash of concatenated chunk hashes, PROMPT_VERSION,
extracted_by). **Why.** No citation, no edge (blueprint 7.1) must hold without trusting a prompt.
**Given up.** Partial acceptance of a host submission: one bad id costs the whole payload.

## 2026-11-14 — Entity resolution threshold 0.90 is untuned

Order: alias table, exact normalised match, bge-base cosine >= `Resolver(threshold=0.90)` against
same-type entities, else create. 0.90 is a guess; it stays untuned until E3 gold labels exist, and
then it is tuned on the tuning half only. Method names are split into family + variant first
(`ViT-B/16` -> `ViT` + `B/16`; `ResNet-50` -> `ResNet` + `50`), because bge scores `ResNet-50` and
`ResNet-101` as near-identical and a similarity merge would hide the difference; datasets, tasks and
metrics are never split (`CIFAR-10` is not `CIFAR` + `10`). **Given up.** Precision of merges until
labels exist. The first live run showed the limit: "image classification tasks" did not merge into
"image classification" (a seed alias would fix that one; embeddings alone did not at 0.90).

## 2026-11-14 — Graph: one extra edge `USES_METHOD`, latest extraction replaces the previous

Blueprint 7.1 has `PROPOSES` only. Methods the paper uses or compares against need a home or
`topic_entities` would count only proposed methods, so they get `USES_METHOD {role, variant, ...}`.
`REPORTS` is keyed by (method, variant, dataset, setting) and carries `value` as data, so two values
for the same key overwrite; a metric named without a number becomes a `REPORTS` edge with a null
value. `EVALUATES_ON` is also derived from every (method, dataset) a result ties together. Writes
first delete the paper's previous extraction edges and Limitation nodes in the same transaction, so
the graph holds one extractor's view per paper, never a mix; the full history stays in
`extraction_cache`. Entity nodes in Neo4j carry no `aliases[]`; the alias list lives in Postgres
`entities`. **Given up.** Keeping two extractors' edges side by side for the same paper.

## 2026-11-14 — MCP server: lazy long-lived connections, session-scoped served chunks, caps in code

One Postgres connection and one GraphRepo per server process, opened lazily and re-opened when a
`SELECT 1` / `verify_connectivity` fails; a lock serialises tool calls because the SDK runs sync tools
on worker threads. The session id (`mcp-<uuid>`) is per process, so `served_chunks` and
`validate_citations` mean "served since this server started". `get_chunks` marks served only the
chunks it returns (own insert, with an `offset` for paging past the 60-chunk cap) so a host can never
cite a chunk it did not receive. Tools return `{"error": ...}` instead of raising; rejected host
extractions also come back as `error` with the validator's `errors`. Caps: papers 100, seeds 30, chunks
60, k 20, search 25, full text 20 per approve call, validate 100. `approve_papers` is a gate by
description only (the host is told to ask the user); the code-level gate is that `fetch_fulltext`
refuses unapproved papers. Startup failures of Postgres or Neo4j are logged, not fatal, so the tool
list still works. **Given up.** Per-user sessions (one process = one session) and tool-level
human-in-the-loop enforcement (MCP elicitation could add it later).

## 2026-11-14 — Ledger days follow each provider's quota clock

`db.quota_day(source)`: UTC by default, America/Los_Angeles for Gemini. The ledgers had used local
time (UTC+5), so counters reset hours before the providers' did, which could let a run exceed a real
daily quota. Six misdated rows were merged into the correct day. **Why.** A spend cap that resets
early is not a cap. **Given up.** Nothing.

## 2026-11-14 — Extraction validates item by item

Strict JSON schema can't express rules such as non-empty strings, so a schema-valid answer can hold
an item Pydantic rejects. Seen live: a result with `dataset: ""` failed a whole paper and stopped a
topic run. Now invalid items are dropped and counted (`invalid_items`), and a failing paper is
recorded without stopping the topic. **Why.** One bad item shouldn't discard a paper's other 40.
**Given up.** Nothing; the count is reported.

## 2026-11-14 — Extraction batches of 4 chunks, 5,500 output tokens

gpt-oss counts reasoning tokens against `max_tokens`, and 4,096 ran out on a 6-chunk full-text
batch. The chain books input + `max_tokens` against Groq's 8k TPM, so a larger cap alone would make
Groq skip every call. **Why.** It fits both constraints. **Given up.** More calls per paper.

## 2026-11-14 — E3 matches canonical names, deterministically

E3 compares names after the same variant split (methods only) and seed-alias mapping the extractor
uses, without the database or embeddings. **Why.** A score must not depend on what `entities` holds
at run time; the first version compared raw spellings, so "ViT-B/16" never matched a gold "ViT".
**Given up.** Credit for similarity merges that only the embedding step would make.
