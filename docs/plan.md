# Plan — first usable phase (blueprint Phase 0 + Phase 1)

Status as of 2026-11-14: **Phase 1 done; Phase 0 done except the deepread answer baseline, which is blocked on two free keys.** `[~]` = partly done. The vision is `blueprint.md`; this is what is being built right now.

## What "done" means

A person with no paid keys can run:

```
make setup && make services && uv run lacuna db migrate
uv run lacuna scout "efficient vision transformers for edge devices" --max-papers 60
uv run lacuna evals e1
uv run lacuna evals e6 --retrieval-only
```

and get: a ranked paper set with seed/snowball provenance in Postgres and `Paper`/`Author`/`CITES`
in Neo4j; an E1 table of recall@50/100/200 for keyword search vs citation snowballing on five CV
surveys; and the deepread retrieval baseline. `make test` and `make lint` pass; a paid-provider
call raises `SpendStageError`.

## Phase 0 — Foundation. Done when

- [x] `make test` passes (unit tests, no network); `make lint` passes.
- [x] `complete_json(chain=["anthropic/..."])` raises `SpendStageError` at `SPEND_STAGE=0`.
- [x] Deepread lives in `src/lacuna/deepread/`, its tests pass, `lacuna deepread build` rebuilds
      the index into the new database.
- [~] Deepread generator re-frozen (decision logged) and the **retrieval** baseline written to
      `evals/results/`. The free-tier **answer** baseline runs as soon as `GROQ_API_KEY` and
      `GEMINI_API_KEY` exist in `.env` (blocked on the user until then).
- [x] `docker compose up -d --wait` brings up Postgres+pgvector and Neo4j; CI runs lint + unit tests.

## Phase 1 — Ingestion, citation graph, E1. Done when

- [x] OpenAlex, Semantic Scholar and arXiv clients: cached, rate-limited, OpenAlex spend ledger.
- [x] `lacuna scout` builds a paper set: seed search → depth-1 snowball (references + citations)
      → ranked and capped → Postgres (`topics`, `topic_papers`, `papers`, `paper_references`)
      → Neo4j (`Paper`, `Author`, `AUTHORED`, `CITES` between set members only).
- [x] E1 dataset frozen in `evals/datasets/e1/` (five surveys with reference lists and freeze dates).
- [x] `lacuna evals e1` writes recall@50/100/200 for both $0 baselines, with the leakage rules
      (survey excluded, nothing published after the survey).

## Work split

| Agent | Owns | Delivers |
|---|---|---|
| research-e1 | `evals/datasets/e1/`, `docs/api-notes.md` | Five surveys chosen by stated criteria, reference lists downloaded, API facts verified against official docs |
| llm-provider | `src/lacuna/llm/`, `tests/llm/` | Chain, cache, quotas, spend gate, structured outputs, `lacuna llm status` |
| ingest-clients | `src/lacuna/ingest/`, `tests/ingest/`, `tests/fixtures/ingest/` | Three clients behind one cache and limiter; OpenAlex cost ledger |
| graph-repo | `src/lacuna/graph/`, `tests/graph/` | Neo4j repository, constraints, MERGE-only upserts, Cypher templates, `lacuna graph stats` |
| deepread-move | `src/lacuna/deepread/`, `tests/deepread/`, `config/deepread.toml`, `evals/datasets/e6/` | FAMM-RAG moved, re-frozen, on the provider layer; retrieval baseline |
| integration (main session) | `src/lacuna/scout/`, `src/lacuna/evals/`, docs | Snowball orchestration, E1 runner, end-to-end verification |

Shared foundation (`settings`, `models`, `db`, `cache`, `ratelimit`, `util`, `cli`, the `llm`
contract) was written first so every agent codes against the same interfaces. Agents do not edit
`pyproject.toml` or `uv.lock`; a needed dependency is reported back and added centrally.

## What the user provides

Free keys in `.env` (see `.env.example`): `OPENALEX_API_KEY` (needed for scout and E1 at full
speed; without it OpenAlex allows $0.10/day), `S2_API_KEY` (optional; arrives by email), and
`GROQ_API_KEY` + `GEMINI_API_KEY` (needed only for the deepread answer baseline). Optional: Ollama
as a local fallback.

## Known limits of this phase

- No LLM anywhere in scout or E1. Screening and extraction arrive in Phase 2.
- E1 uses five surveys; treat the table as directional until more are added.
- Semantic Scholar refuses unauthenticated requests (429 on the first call), so scout uses OpenAlex only until `S2_API_KEY` exists.

## Results (2026-11-14)

- **E1** (`evals/results/e1_07cb6e06.md`): recall@200 keyword 0.036, snowball 0.170 (macro, 5 surveys).
- **E6 retrieval** (`evals/results/e6_retrieval_ce40123b.md`): recall@5 0.735 over 34 questions.
- Live end-to-end scout: 30 papers, 141 authors, 97 in-set CITES in 44 s for $0.0032; a re-run hits
  the cache (0.5 s, zero API calls). Full E1 run: 140 OpenAlex calls, $0.025, 8 minutes.
- Tests: 92 passed, 1 skipped (needs keys); ruff clean.

## Next (Phase 2 candidates, each measured against E1)

1. Query planning: turn the question into several keyword queries, restricted to the CS field. Fixes
   the video-survey drift.
2. Relevance screening of snowball candidates: local bge similarity to the question ($0), then
   free-tier LLM screening on abstracts. Fixes the "generic classics" ranking.
3. Extraction + MCP server, so the tool is usable from Claude Code (personal-use trigger).

---

# Phase 2 — Extraction + MCP server → personal use (started 2026-11-14)

Blueprint 13, Phase 2. The goal is the **personal MVP**: run a real topic from Claude Code through
MCP and see papers, extracted entities and answers with citations. E3 reports F1 per entity type.

## Done when

- [x] Abstracts of every topic paper are chunks (`<paper_id>:abs`), embedded, searchable.
- [x] Approval gate: full text (arXiv PDF → chunks → embeddings) is fetched only for approved papers.
- [x] Extraction over chunks through the free-tier chain **and** through the host (`save_extraction`),
      validated against a Pydantic schema; every edge carries `chunk_id`, `confidence`, `extracted_by`;
      a host extraction citing a chunk it was never served is rejected.
- [~] Entity resolution: normalisation, a seed alias table and bge similarity, so ViT / Vision
      Transformer / ViT-B/16 (variant as a property) and ImageNet-1k / ILSVRC-2012 merge.
- [x] Graph: `Method`, `Dataset`, `Task`, `Metric`, `Limitation` and the six extracted edge types, MERGE-only.
- [x] Throughput and schema-validity per provider measured on 10 papers before scaling.
- [~] MCP server (stdio, `mcp` 2.x) with tools `search_papers`, `scout_topic`, `approve_papers`,
      `get_chunks`, `save_extraction`, `ask`, `related`, `validate_citations`, and the workflow
      prompts `scout_topic`, `draft_review`, `deep_read`; connected from Claude Code.
- [x] `usage_log` records every MCP tool call (Stage 2 trigger 1).
- [~] **E3** reports P/R/F1 per entity type on hand labels, plus schema-validity rate per provider.
      **Hand labels come from the user** (blueprint 11: 30 hand-labelled papers, split into tuning
      and reporting halves). The harness and the labelling templates are built here.

`find_gaps` and the rule-based router are Phase 3.

## Work split (Sonnet agents, disjoint paths)

| Agent | Owns | Delivers |
|---|---|---|
| papers | `src/lacuna/papers/`, `tests/papers/` | abstract chunks, approval gate, arXiv full text → chunks, `get_chunks` (records served), `search_chunks` |
| extract | `src/lacuna/extract/`, `src/lacuna/graph/entities.py`, `config/aliases.toml`, `tests/extract/` | schema, LLM + host extraction, resolution, graph writes, throughput command |
| e3 | `src/lacuna/evals/e3.py`, `evals/datasets/e3/`, `config/llm.toml`, `src/lacuna/llm/providers.py`, `tests/evals/test_e3.py` | E3 scoring + labelling templates; verified model ids and limits |
| mcp (wave 2) | `src/lacuna/mcp_server/`, `tests/mcp_server/` | the MCP server over the three packages above |
| main session | `db.py`, docs, integration | schema, review, live end-to-end verification |

## Phase 2 status (2026-11-14)

Built and verified live; two items wait on the user.

- **Entity resolution** `[~]`: aliases, variant split and similarity work (ViT-B/16 → ViT; ILSVRC-2012 →
  ImageNet-1k). The 0.90 similarity threshold is untuned, and live output shows misclassifications
  (BERT Large and ALBERT xxlarge filed as datasets; 27 PROPOSES edges for one paper). E3 will measure them.
- **MCP server** `[~]`: a full Mode A session ran over real stdio: scout, get_chunks, a host
  extraction rejected for citing another paper's chunk, a valid one saved to the graph, ask, and
  validate_citations catching a fake id. Claude Code reads `.mcp.json`, but the server is **pending the
  user's approval** in Claude Code.
- **E3** `[~]`: harness, guidelines and 24 templates (12 tuning / 12 reporting) are ready. **F1 needs
  the user's hand labels.**

Results:

- **Extraction throughput** (`evals/results/extract_throughput_2026-11-14.md`, 10 papers, caches off):
  gpt-oss-120b 20/20 batches, 100% schema-valid, 253 items; gemini-3.5-flash-lite 18/20 (two Google
  503s), 94.7%, 172 items; gpt-oss-20b 17/20 (output cap), 100%, 164 items. The chain keeps Gemini first
  for its larger free quota, with gpt-oss-120b as the fallback.
- **E6 answers** (`evals/results/2026-11-14_0a70d1e9.md`, closes Phase 0): accuracy 0.63 overall; text
  0.64, table 0.67, **figure 0.00**, unanswerable 1.00 (no hallucinated answers). The figure split is
  the gap the deep-read module exists to close (Phase 3+). Faithfulness reads 1.00 but the judge is
  uncalibrated.

Bugs found by live testing and fixed: ledgers dated by local time instead of each provider's quota
clock; one invalid item failing a whole paper; Groq silently skipping every full-text batch because
batch + output cap exceeded its TPM; a throughput metric that blamed models for skipped calls; E3
comparing raw spellings instead of canonical names.

## Next

1. **User:** approve the `lacuna` server in Claude Code and run a real topic (`/mcp`, then the
   `scout_topic` prompt). That is the Phase 2 personal-MVP check and Stage 2 trigger 1.
2. **User:** label the 24 templates in `evals/datasets/e3/todo/` (about an hour), then
   `uv run lacuna evals e3 --split tuning` to tune the resolver threshold and prompt, and
   `--split reporting` once for the number.
3. Phase 3: gap detectors + E2, rule-based router, deep-read integration.
