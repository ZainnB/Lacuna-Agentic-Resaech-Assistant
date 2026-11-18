# Lacuna

An agentic research assistant for literature reviews and research gaps in computer vision and ML.
It builds a scholarly knowledge graph from papers, and **every claim it makes is measured**. The
full design is in [docs/blueprint.md](docs/blueprint.md); what is being built now is in
[docs/plan.md](docs/plan.md). It costs $0 to run: free-tier APIs only, with paid models disabled in code.

## What works today (Phases 0–2)

```bash
make setup && make services && uv run lacuna db migrate
uv run lacuna scout "efficient vision transformers for edge devices" --max-papers 60
uv run lacuna graph topic <topic_id>        # printed by scout
uv run lacuna evals e1                      # survey recall
uv run lacuna deepread eval --retrieval-only
uv run lacuna papers index <topic_id> && uv run lacuna extract topic <topic_id>   # Phase 2
```

**From Claude Code (Mode A, $0 to the project):** the repo ships `.mcp.json`, so opening Claude Code
here offers the `lacuna` MCP server. Approve it, then use the `scout_topic`, `draft_review` or
`deep_read` prompts. Claude plans and writes; Lacuna does search, snowballing, chunk retrieval,
extraction storage and citation checks. Claude may cite only chunks the tools returned in the
session, and an extraction citing anything else is rejected in code.

`scout` turns a question into a ranked paper set: OpenAlex keyword search for seeds, then depth-1
citation snowballing (references and citing works), ranked by how many seeds link to each
candidate. Papers, references and provenance go to Postgres; `Paper`, `Author`, `CITES` and `Topic`
go to Neo4j. Every API response is cached, so re-running a scout is free and instant.

## Results

### E1 — survey recall (Phase 1, no LLM)

Five CV/ML surveys from 2025–26. Each is queried with a plain-language question about its topic, and
scored by how much of its reference list a method retrieves. The survey itself is excluded, and
nothing published after the survey may be retrieved. Macro average over the five surveys:

| method | recall@50 | recall@100 | recall@200 |
|---|---:|---:|---:|
| keyword search | 0.029 | 0.032 | 0.036 |
| + citation snowball | **0.064** | **0.103** | **0.170** |

Citation snowballing finds 4–5× more of what a survey author cited than keyword search does. The
absolute numbers are low, and two failure modes explain most of it:

- **Plain-language queries drift.** For the video-understanding survey, the query about "emotions,
  humour and intent" pulled humanities papers about comedy. Recall is 0 for both methods, and the
  snowball inherits the off-topic seeds.
- **Snowballing pulls in generic classics.** ImageNet, ResNet and BERT are cited by nearly every
  seed, so they outrank on-topic papers.

Both point at Phase 2: LLM query planning and relevance screening, measured against this table.
Per-survey numbers are in `evals/results/e1_*.md`; the dataset and its selection criteria are in
[evals/datasets/e1/](evals/datasets/e1/README.md).

### Extraction throughput (Phase 2, free tier)

10 papers per provider, caches off. Schema validity is measured over responses actually received.

| model | batches OK | schema-valid | entities found | mean latency |
|---|---:|---:|---:|---:|
| groq/openai/gpt-oss-120b | 20/20 | 100% | 253 | 3.5 s |
| gemini/gemini-3.5-flash-lite | 18/20 | 94.7% | 172 | 2.5 s |
| groq/openai/gpt-oss-20b | 17/20 | 100% | 164 | 2.6 s |

E3 (extraction precision/recall per entity type) is waiting on hand labels; templates are in
`evals/datasets/e3/todo/`.

### E6 — deep read (formerly FAMM-RAG)

Full-text question answering over 7 papers (246 pages, 773 chunks), 34 answerable questions:

| split | n | recall@5 | MRR@10 |
|---|---:|---:|---:|
| text | 14 | 0.571 | 0.426 |
| table | 15 | 0.867 | 0.606 |
| figure | 5 | 0.800 | 0.550 |
| all | 34 | 0.735 | 0.524 |

Answers on the frozen free-tier generator (`groq/openai/gpt-oss-120b`, judged by
`gemini-3.5-flash-lite`): accuracy text 0.64, table 0.67, **figure 0.00**, unanswerable 1.00 (no
hallucinated answers). Text-only retrieval can't read plots; closing that gap is what the
deep-read module is for. The split design, figure-split check and labelling rules are in
[docs/archive/famm-plan.md](docs/archive/famm-plan.md).

## Setup

Needs [uv](https://docs.astral.sh/uv/) and Docker. Copy `.env.example` to `.env`. Every key is free
and optional, but each one unlocks more:

| key | without it | with it |
|---|---|---|
| `OPENALEX_API_KEY` | $0.09/day of OpenAlex (≈ 3 E1 runs) | $0.90/day |
| `S2_API_KEY` | Semantic Scholar refuses requests | 1 request/s |
| `GROQ_API_KEY`, `GEMINI_API_KEY` | no LLM calls | deep-read answer baseline; Phase 2 extraction |

```bash
make setup      # uv sync --extra dev
make services   # Postgres + pgvector on :5433, Neo4j on :7474 / :7687
make test       # unit tests run with no network; live tests skip when services are absent
make lint
```

## Layout

```
src/lacuna/   settings, db, cache, ratelimit, models, cli   shared foundation
  llm/        free-tier provider chain: cache, quotas, spend gate (paid calls raise below Stage 2)
  ingest/     OpenAlex, Semantic Scholar, arXiv clients
  graph/      Neo4j repository; all Cypher parameterised
  scout/      seed search -> snowball -> ranked paper set
  papers/     abstract + approved full-text chunks, embeddings, served-chunk tracking
  extract/    schema, free-tier and host extraction, entity resolution
  mcp_server/ MCP tools and workflow prompts for Claude Code / Desktop
  deepread/   full-text, table and figure QA (formerly FAMM-RAG)
  evals/      E1 (survey recall), E3 (extraction), E6 (deep read)
docs/         blueprint, plan, decisions log, API notes
```
