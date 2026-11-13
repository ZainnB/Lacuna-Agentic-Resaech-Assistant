# Lacuna — build rules

Agentic literature-review assistant over a scholarly knowledge graph. Vision and phases:
`docs/blueprint.md`. Current execution plan: `docs/plan.md`. Decision log: `docs/decisions.md`.

## Rules (blueprint section 0)

- Build **phase by phase**. A phase's *Done when* must hold before the next phase is declared started.
- **Write the eval before the feature.** Every phase ends with a number; no number, no claim.
- **Paid providers stay disabled until Stage 2.** `lacuna.llm` raises `SpendStageError` for paid
  specs while `SPEND_STAGE < 2`. Never work around this, not even in tests.
- **Every LLM and paper-API response is cached.** Re-running anything must not re-spend quota.
- **Limits are enforced in code, never in prompts**: rate limiters, daily budgets, caps on papers,
  depth and tokens.
- Record every non-obvious decision in `docs/decisions.md` (decision, why, what was given up).
- No new heavy dependency without a decision entry.
- `deepread/` (formerly FAMM-RAG) has a **frozen generator, prompt and chunking config**. It was
  re-frozen once to a free-tier model before its first baseline. Do not change them again.
- Only public paper text goes to free-tier APIs. Private notes go through MCP (Mode A) only.
- `paper_id` is always the OpenAlex work ID. Graph `Paper` nodes exist only for papers in a topic
  set; outside references are IDs in `paper_references`.
- Secrets live in `.env` only. Never hardcode an email address or key; `CONTACT_EMAIL` is optional.

## Layout

```
src/lacuna/
  settings.py models.py db.py cache.py ratelimit.py util.py cli.py   shared foundation
  llm/        provider layer (free chain, cache, quotas, spend gate)
  ingest/     openalex, semantic_scholar, arxiv clients
  graph/      neo4j repository, cypher templates
  scout/      seed search -> snowball -> ranked paper set
  deepread/   FAMM-RAG (full-text / table / figure QA)
  evals/      e1 ... e6 runners; datasets in evals/datasets/, results in evals/results/
tests/        unit tests run with no network and no Docker; `live` tests skip when services are absent
```

## Commands

```
make setup                 uv sync --extra dev
make services              docker compose up -d --wait   (Postgres+pgvector :5433, Neo4j :7474/:7687)
uv run lacuna db migrate   idempotent schema
uv run lacuna --help       every subcommand
make test / make lint
```

## Conventions

- Python 3.12, synchronous code until the API server (Phase 2). Pydantic v2 models at boundaries.
- Tests: `pytest`; unit tests use `MemoryCache`, `respx`/`httpx.MockTransport` and recorded fixtures.
  Mark anything that needs Docker or keys `@pytest.mark.live`.
- Match the surrounding code's style; `ruff` (line length 110) is the formatter and linter.
