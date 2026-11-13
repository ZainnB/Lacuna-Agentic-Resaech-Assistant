# Plan — first usable phase (blueprint Phase 0 + Phase 1)

Status as of 2026-11-13. The vision is `blueprint.md`; this is what is being built right now.

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

- [ ] `make test` passes (unit tests, no network); `make lint` passes.
- [ ] `complete_json(chain=["anthropic/..."])` raises `SpendStageError` at `SPEND_STAGE=0`.
- [ ] Deepread lives in `src/lacuna/deepread/`, its tests pass, `lacuna deepread build` rebuilds
      the index into the new database.
- [ ] Deepread generator re-frozen (decision logged) and the **retrieval** baseline written to
      `evals/results/`. The free-tier **answer** baseline runs as soon as `GROQ_API_KEY` and
      `GEMINI_API_KEY` exist in `.env` (blocked on the user until then).
- [ ] `docker compose up -d --wait` brings up Postgres+pgvector and Neo4j; CI runs lint + unit tests.

## Phase 1 — Ingestion, citation graph, E1. Done when

- [ ] OpenAlex, Semantic Scholar and arXiv clients: cached, rate-limited, OpenAlex spend ledger.
- [ ] `lacuna scout` builds a paper set: seed search → depth-1 snowball (references + citations)
      → ranked and capped → Postgres (`topics`, `topic_papers`, `papers`, `paper_references`)
      → Neo4j (`Paper`, `Author`, `AUTHORED`, `CITES` between set members only).
- [ ] E1 dataset frozen in `evals/datasets/e1/` (five surveys with reference lists and freeze dates).
- [ ] `lacuna evals e1` writes recall@50/100/200 for both $0 baselines, with the leakage rules
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
- Semantic Scholar without a key is slow; the snowball still completes because results are cached.
