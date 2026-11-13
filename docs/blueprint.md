# Lacuna — Blueprint & Build Plan (v2: zero-cost first)

*Working name ("lacuna" = a gap). Rename freely.*

An agentic research assistant for finding ideas, research gaps and doing literature reviews in
computer vision / ML. It builds a **scholarly knowledge graph** from papers, runs a **literature-review
workflow** over it, and backs every claimed gap with graph evidence. **Everything it claims is
measured.**

Merges two guide projects — *Knowledge Graph RAG* and *Multi-Agent Research Assistant* — and absorbs
**FAMM-RAG** as its deep-read module (full-text, table and figure QA over the papers it finds).

**v2 change:** the build costs **$0 in API spend** until two triggers are met (section 2). Reasoning
runs either in the user's own MCP host (Claude, Claude Code, ChatGPT — paid by their subscription,
not by this project) or on free-tier providers. Paid models come later, behind the same interface.

---

## 0. How Claude Code should use this file

- Build **phase by phase** (section 13). Don't start a phase until the previous one's *Done when* holds.
- **Write the eval before the feature.** Every phase has a number; no number, no claim.
- **Paid providers stay disabled until Stage 2.** The provider layer refuses paid calls while
  `SPEND_STAGE < 2`. Never work around this.
- **Every LLM response is cached** by (provider, model, prompt hash). Re-running anything must not
  re-spend free-tier quota.
- Record every non-obvious decision in `docs/decisions.md` (decision, why, what was given up).
- FAMM-RAG's generator gets **re-frozen once** to a free-tier model before its first baseline run
  (no results exist yet, so nothing is invalidated). After that, never change its frozen generator,
  prompt or chunking config.
- No new heavy dependency without a decision entry. Limits are enforced in code, never in prompts.

---

## 1. Goals and non-goals

**Goals**
1. Turn a research question into a curated, cited paper set via search plus citation snowballing.
2. Find **evidence-backed** research gaps through deterministic graph queries.
3. Write literature-review drafts where every citation resolves to a retrieved paper or chunk.
4. Answer deep questions about specific papers (text, tables, figures) through the FAMM-RAG module.
5. Report measured results for each of the above.
6. Be Zain's own research tool for MSc planning, then a tool other researchers use.
7. **Cost $0 to run** until it has proven useful.

**Non-goals (v1)**
- Beating Elicit, Undermind or Semantic Scholar on coverage. We win on evidence and measurement.
- Billing or paid plans.
- Fields outside CV/ML. The design stays domain-agnostic, but evals target CV/ML only.

**Positioning in one line:** existing gap-finders output generated synthesis claims; Lacuna outputs
gaps as graph queries with evidence, and measures how often its gaps were real.

---

## 2. Spending stages

| Stage | When | Who does the reasoning | API spend |
|---|---|---|---|
| **0 — Build** | Now | Free-tier providers (server-side) and the developer's MCP host | $0, enforced in code |
| **1 — Personal use** | MCP server works locally | Zain's own Claude / Claude Code via MCP | $0 |
| **2 — Paid** | **Both triggers met** | Paid models for server-side agents and full eval runs | Per-run and monthly caps |

**Triggers for Stage 2** (logged in the `usage_log` table, not judged by feel; adjust numbers in
`docs/decisions.md`):
1. **Personal use:** at least 3 real topics run end to end, and at least one result used in Zain's
   actual MSc research planning.
2. **External use:** deployed, with at least 10 external users or 25 runs by others in a month.

Until then: no paid model calls, no paid eval runs.

---

## 3. Two reasoning modes

The same tools serve both modes, so nothing is built twice.

### Mode A — Host-driven via MCP (default for personal use; free to the project)

The MCP host (Claude Desktop, Claude Code, ChatGPT developer mode, Cursor — any MCP client) is the
planner, writer and reviewer. Lacuna is the tool server: search, snowball, ingest, graph queries, gap
detectors, retrieval, citation validation.

- **Workflow prompts** shipped as MCP prompts: `scout_topic`, `find_gaps`, `draft_review`,
  `deep_read`. They tell the host the steps and the rules (cite only tool results, stop at budgets).
- **Host-side extraction:** the host reads chunks through `get_chunks(paper_id)` and submits
  `save_extraction(paper_id, json)`. The server validates the schema, stores provenance
  (`extracted_by = host:<model>`), and rejects anything citing a chunk it didn't return.
- **Private ideas stay private:** Zain's own notes and draft ideas go through Mode A only, never to
  free-tier APIs (several free tiers may train on prompts).

### Mode B — Server-driven agents (web UI, scheduled runs, evals)

LangGraph agents call LLMs through a **provider layer** (LiteLLM) with a free-tier fallback chain:

```
Gemini Flash (Google AI Studio) → Groq → Cerebras → OpenRouter :free models → local Ollama
```

- Each provider has its own rate limiter and daily quota counter; on 429 or quota exhaustion the
  layer falls back to the next provider.
- Structured outputs are validated with Pydantic; invalid JSON retries once, then falls back.
- Every response is cached; every call is logged with provider, model, tokens and latency.
- In Stage 2 a paid provider is added to the front of the chain for chosen roles, with caps.

---

## 4. Core workflows

| Workflow | User does | System does |
|---|---|---|
| **Scout a topic** | Asks a question ("efficient ViTs for edge devices") with a budget | Plans sub-questions, searches, snowballs citations, screens, asks the user to approve the paper set, ingests it |
| **Find gaps** | Opens gaps for a topic | Runs gap detectors over the graph; each gap shows type, evidence papers and a counter-evidence check |
| **Draft a review** | Asks for a review of a topic or sub-question | Draft from findings; every claim cited; one review round |
| **Ask** | Asks "which methods were tested on COCO and ADE20K?" | Router picks graph, vector or deep-read path and answers with citations |

---

## 5. Architecture

```
   MCP hosts (Claude, Claude Code, ChatGPT)            Nuxt 3 web app (Phase 4)
          │  stdio locally / HTTPS + OAuth when deployed     │ REST + SSE
┌─────────▼──────────────────────────────────────────────────▼────────────────┐
│ FastAPI + MCP server (same service layer)                                    │
│  tools: search · snowball · ingest · get_chunks · save_extraction ·          │
│         graph_query · find_gaps · ask · deep_read · validate_citations       │
└──────┬──────────────────────────────┬──────────────────────────────────────┘
       │ enqueue (Mode B)             │ read
┌──────▼──────── worker (arq) ────────┴──────────────────────────────────────┐
│ LangGraph agents: Planner → Scout ⇄ approval gate → Analyst → Writer ⇄ Reviewer│
│ Provider layer (LiteLLM): free-tier chain, cache, quotas, Stage gate          │
└──┬──────────┬───────────────┬─────────────────┬──────────────────────────────┘
 Redis      Neo4j        Postgres + pgvector    Paper APIs: OpenAlex · Semantic
 run state  knowledge    chunks, caches, runs,  Scholar · arXiv PDFs
 + queue    graph        traces, usage, evals   Local: bge-base embeddings, reranker
```

**Shared key:** graph nodes and Postgres chunks share `paper_id` and `chunk_id`, so the system can
jump from a graph path to the exact sentence and back.

---

## 6. Stack and key decisions

| Layer | Choice | Why |
|---|---|---|
| Language / tooling | Python 3.12, uv, ruff, pytest | Matches FAMM-RAG |
| API | FastAPI, Pydantic v2, SSE | Async, typed, Swagger docs |
| MCP | Official MCP Python SDK; stdio locally, streamable HTTP + OAuth when deployed | One service layer behind both API and MCP |
| Jobs / state | arq + Redis | Long runs off the request path; resume after crash |
| Graph | Neo4j Community in Docker (dev); AuraDB Free (deploy) | Multi-hop and gap queries in Cypher; free hosted tier |
| Vectors + relational | Postgres + pgvector (from FAMM-RAG); Neon free tier when deployed | Already used for FAMM and Riley |
| Embeddings / reranker | bge-base (local), small local cross-encoder | $0, already measured |
| LLM | LiteLLM provider layer, free-tier chain (section 3) | One interface for free now and paid later |
| Agents | LangGraph (hand-rolled state machine acceptable; log the decision) | Named skill in postings |
| PDFs | PyMuPDF (from FAMM-RAG) | Already set up |
| Paper sources | OpenAlex, Semantic Scholar, arXiv | Free; limits below |
| Frontend | Nuxt 3, Vue 3, TypeScript, PrimeVue, Tailwind, Cytoscape.js, Chart.js or ECharts | Zain's production stack |
| Tracing | OpenTelemetry → Jaeger (compose) + `steps` table | Industry standard plus a queryable run log |
| Hosting (Phase 5) | Render free web service (as for Riley), Neon, AuraDB Free | $0; verify a free option for the worker |

**Free limits to design around (as reported mid-2026; verify at build time):**
- **Gemini (AI Studio):** most generous free tier, roughly 1,500 requests/day on Flash; free-tier
  prompts may be used for training.
- **Groq:** about 30 requests/minute free.
- **Cerebras:** about 1M tokens/day but only ~5 requests/minute.
- **OpenRouter:** many `:free` models behind one key; limits vary per model.
- **OpenAlex:** API key required; about $1/day of free credits per key; ID/DOI lookups free.
- **Semantic Scholar:** free; about 1 request/second with a key.
- **AuraDB Free:** 200,000 nodes, 400,000 relationships, 1 GB. **Design rule:** create `Paper`
  nodes only for ingested papers; store outside references as IDs in Postgres, not graph stubs.

---

## 7. Data model

### 7.1 Knowledge graph (Neo4j)

| Node | Key properties |
|---|---|
| `Paper` | `paper_id` (OpenAlex ID), doi, arxiv_id, title, year, venue, citation_count |
| `Author` | `author_id`, name |
| `Method` / `Dataset` / `Task` | `canonical_name`, aliases[] |
| `Metric` | `canonical_name`, higher_is_better |
| `Limitation` | text, paper_id, chunk_id |

| Edge | From → To | Properties |
|---|---|---|
| `CITES` | Paper → Paper | source, context (if available) |
| `AUTHORED` | Author → Paper | position |
| `PROPOSES` | Paper → Method | chunk_id, confidence, extracted_by |
| `EXTENDS` | Method → Method | paper_id, chunk_id, confidence, extracted_by |
| `EVALUATES_ON` | Paper → Dataset | method, chunk_id, confidence, extracted_by |
| `ADDRESSES` | Paper → Task | chunk_id, confidence, extracted_by |
| `REPORTS` | Paper → Metric | method, dataset, value, setting, chunk_id, confidence, extracted_by |
| `STATES_LIMITATION` | Paper → Limitation | chunk_id |

- Every extracted edge carries `chunk_id`, `confidence` and `extracted_by` (`host:<model>` or
  `api:<provider>/<model>`). No citation, no edge.
- `MERGE`, never `CREATE`, so re-ingestion is idempotent.
- **Entity resolution:** normalise, then embedding similarity above a tuned threshold, then aliases.
  Get right: ViT / Vision Transformer / ViT-B/16 (variant as property); ImageNet-1k / ILSVRC-2012.

### 7.2 Postgres

| Table | Purpose |
|---|---|
| `papers`, `chunks` | Metadata mirror, ingest status; chunks with pgvector embeddings |
| `api_cache`, `llm_cache` | Paper-API responses; LLM responses keyed by (provider, model, prompt hash) |
| `extraction_cache` | (paper_id, content_hash, prompt_version) → validated extraction |
| `topics`, `runs`, `steps` | Workspaces, runs with budgets, step-level trace |
| `findings`, `gaps` | Claims with citations; gaps with evidence and counter-evidence |
| `provider_usage` | Calls, tokens, 429s and fallbacks per provider per day |
| `usage_log` | Real-use events for the Stage 2 triggers (topic, mode, user, outcome) |
| `eval_results` | Eval name, config hash (includes model), metrics JSON |

---

## 8. Agents (Mode B) and workflow prompts (Mode A)

Each agent is a prompt + tool set + Pydantic schemas. The same steps ship as MCP prompts for Mode A.

| Agent | Tools | Output |
|---|---|---|
| **Planner** | none | `Plan`: sub-questions, search queries, seed hints |
| **Scout** | `search_openalex`, `search_s2`, `snowball`, `screen`, `ingest` | Candidates with screening reasons; status `ok` / `no_results` / `rate_limited` / `budget_hit` |
| **Analyst** | `find_gaps(type)`, `graph_query(template, params)`, `ask` | `Gap[]` with evidence |
| **Writer** | `get_chunks`, `deep_read` | `Report` with per-claim citations |
| **Reviewer** | `validate_citations`, `get_chunks` | Approve, or one round of fixes |

**Rules enforced in code:** caps on sub-questions, papers, snowball depth, tokens, wall-clock and
(Stage 2) dollars; reviewer sends back once; **human approval gate** before full-text extraction;
state saved after every node; search failures are states, not crashes.

**Leakage rule:** only papers returned by tools in this run may be cited. The model's memory of
papers is never a source. This also protects the evals.

---

## 9. Gap detection

Deterministic graph queries; the LLM only words the gap and runs the counter-evidence search.
These cost no LLM calls, so they run freely in every stage.

| Gap type | Query idea |
|---|---|
| **Coverage** | Popular methods × popular datasets/tasks with no `EVALUATES_ON` link |
| **Unaddressed limitation** | Limitation clusters with no later citing paper addressing them |
| **Conflicting results** | Same method + dataset + metric, reported values far apart |
| **Emerging combination** | Two rising methods/tasks that rarely co-occur |

**Counter-evidence check:** search once more for a filling paper; if found, mark `possibly_filled`.
**Scoring:** popularity × recency × evidence strength, simple and logged.

---

## 10. QA router, API, UI, MCP

**Router:** returns `graph` / `vector` / `deep_read` / `both`. Starts **rule-based** (keywords and
detected entities) so it costs nothing; an LLM router is a Stage 2 option, measured against the rules.
Graph path uses **parameterised Cypher templates only** — the model fills parameters, never writes
Cypher. Answers require a citation per claim that resolves to a retrieved `chunk_id`.

**API:** `/topics`, `/runs` (+ `/events` SSE, `/approve`, `/report`, `/trace`), `/topics/{id}/gaps`,
`/ask`, `/graph/neighbourhood`, `/papers/{id}`, `/evals`, `/usage`.

**MCP tools (Phase 2, read + extraction):** `search_papers`, `scout_topic`, `get_chunks`,
`save_extraction`, `find_gaps`, `ask`, `deep_read`, `related(paper_id)`, `validate_citations`.
Plus the four workflow prompts from section 3.

**Nuxt screens (Phase 4):** topics; new run with budget; run view with approval gate and trace;
graph explorer; gap matrix (method × dataset heatmap with evidence drawer); report with clickable
citations; evals; provider usage.

---

## 11. Evaluation plan

Split by cost. **Free evals run now. Paid evals wait for Stage 2.** Config hash includes the model,
so free-tier results stay separate from paid-model results.

| # | Eval | Runs in | Ground truth → metric | Baselines |
|---|---|---|---|---|
| E1 | **Survey recall** | Free | Reference lists of 5 CV surveys → recall@50/100/200 | Keyword search; snowball without LLM (both $0); agent on free tier |
| E2 | **Temporal gap validation** | Free (graph-only) | Freeze corpus at date T → % of top-N gaps filled after T | **Random empty cells** of equal popularity |
| E3 | **Extraction quality** | Free | 30 hand-labelled papers → P/R/F1 per entity type | Abstract vs full text; **per-provider schema-validity rate** |
| E4 | **Citation faithfulness** | Labels free; judge calibration **paid** | ~100 hand-labelled claim–citation pairs → Cohen's kappa | Collect labels now, calibrate a judge in Stage 2 |
| E5 | **Multi-hop QA** | Retrieval free; answers free tier | 50–100 questions by hop count → evidence recall@k and accuracy | Vector-only |
| E6 | **Deep-read** (FAMM) | Free | FAMM's 41 questions → retrieval metrics, then answers on re-frozen free generator | Text-only |

**Stage 2 additions:** re-run E1, E3, E5, E6 with paid models and publish a **free vs paid** table
(quality, latency, cost per run). That trade-off table is itself a strong interview artifact.

**Leakage traps:** E1 excludes the survey and anything after it; agents cite tool results only;
prefer surveys after the model's training cutoff. E2 gaps come from graph queries over the frozen
corpus only. E3/E4 split labels into tuning and reporting halves.

**Rate-limit-aware eval runner:** resumable, reads `llm_cache` first, spreads calls across providers,
records which provider answered. With small sets, CI gates on counts of regressed items.

---

## 12. Repo layout

```
lacuna/
├── api/            FastAPI app, routers, SSE
├── mcp/            MCP server: tools + workflow prompts (Phase 2)
├── agents/         planner, scout, analyst, writer, reviewer, supervisor graph, budgets
├── llm/            provider layer: LiteLLM config, fallback chain, cache, quotas, stage gate
├── ingest/         openalex, semantic_scholar, arxiv clients; cache; rate limiter; pdf
├── extract/        schemas, prompts, entity resolution, host-extraction validation
├── graph/          neo4j repository, cypher templates
├── retrieval/      vector search, rule router, answer assembly, citation validation
├── deepread/       FAMM-RAG, moved in (generator re-frozen once to a free model)
├── gaps/           detectors, scoring, counter-evidence
├── evals/          e1 … e6, datasets/, results/
├── web/            Nuxt 3 app (Phase 4)
├── docs/           blueprint, decisions.md, guide-references.md
├── docker-compose.yml   postgres+pgvector, neo4j, redis, jaeger, ollama (optional), api, worker, web
├── Makefile        setup, ingest, eval, reproduce
└── CLAUDE.md       build rules (section 0)
```

---

## 13. Phases (near full-time: personal use ≈ 2 weeks, full ≈ 5–6 weeks)

**v2 reorder:** the MCP server moves early (Phase 2) so personal use — trigger 1 — starts in week 2.
The web UI moves to Phase 4, before deployment — trigger 2.

### Phase 0 — Foundation (days 1–2)
- Restructure; move FAMM-RAG into `deepread/` with tests passing.
- Provider layer: free chain, `llm_cache`, quotas, `SPEND_STAGE` gate (paid calls raise).
- Re-freeze FAMM's generator to a free-tier model (decision entry); run its **retrieval** baseline
  ($0, no LLM) and its free-tier answer baseline.
- Docker Compose; CI with lint and tests.
- **Done when:** `make test` passes, a paid-provider call fails loudly, and FAMM baselines exist.

### Phase 1 — Ingestion, citation graph, E1 baselines (days 3–6)
- Cached, rate-limited OpenAlex / Semantic Scholar / arXiv clients.
- Seed query → papers → snowball (depth 1) → Neo4j `Paper`, `Author`, `CITES`.
- **E1** with the two $0 baselines.
- **Done when:** recall@k table for keyword search vs snowballing on 5 surveys.

### Phase 2 — Extraction + MCP server → personal use starts (days 7–12)
- Extraction on abstracts, then arXiv full text for approved papers, via free-tier chain **and** via
  host (`save_extraction`). Measure throughput per provider on 10 papers before scaling.
- Entity resolution with aliases.
- MCP server (stdio): tools + workflow prompts; connect from Claude Desktop/Code.
- **E3** with hand labels; per-provider schema-validity rate.
- Start `usage_log`.
- **Done when (personal MVP):** Zain runs a real topic from Claude through MCP, sees papers, entities
  and answers with citations, and E3 reports F1 per entity type.

### Phase 3 — Gaps, router, deep-read (days 13–18)
- Four gap detectors, scoring, counter-evidence; `find_gaps` MCP tool.
- **E2** against the random-cell baseline.
- Rule-based router, Cypher templates, deep-read integration, citation validation.
- **E5** retrieval metrics (and free-tier answers).
- **Done when:** E2 and E5 tables exist and every gap shows evidence.

### Phase 4 — Server-side agents + web UI (days 19–26)
- LangGraph agents on the free chain with budgets, Redis state, resume after a killed worker,
  approval gate, `/trace`, SSE.
- Nuxt app: topics, runs, approval gate, trace, graph explorer, gap matrix, report, usage.
- **E1** agent row on free tier.
- **Done when:** a question run from the web UI produces a cited report on free models, a killed
  worker resumes, and every run has a trace.

### Phase 5 — Deploy for others, still $0 (days 27–36)
- Deploy API + Nuxt (Render), Neon, AuraDB Free. Remote MCP over HTTPS with OAuth so other
  researchers connect from **their own** Claude or ChatGPT — their subscription pays for reasoning.
- Web UI uses the free chain with per-user rate limits.
- OpenTelemetry → Jaeger; provider-usage dashboard; CI eval gates on free evals.
- Collect E4 human labels.
- README opens with results tables; 90-second demo video; "what didn't work" section.
- **Done when:** an outside user completes a run through remote MCP or the web app, and
  `usage_log` tracks both triggers.

### Stage 2 — Paid (only when both triggers are met)
- Add a paid provider at the front of the chain for chosen roles, with per-run and monthly caps.
- Calibrate the E4 judge; re-run evals; publish the free vs paid table.

---

## 14. Cost and rate controls

- `SPEND_STAGE` gate in the provider layer; paid calls impossible before Stage 2.
- Per-run caps: papers, tokens, wall-clock (and dollars in Stage 2).
- Cache everything: paper APIs, LLM responses, extractions. Re-runs are free.
- Abstract-level extraction for screening; full text only after the approval gate.
- OpenAlex: prefer free ID/DOI lookups; batch; ingest per topic, never per question.
- Spread load across providers; record 429s and fallbacks in `provider_usage`.

---

## 15. Risks

| Risk | Mitigation |
|---|---|
| Free-tier limits or terms change | Fallback chain across 5 providers + local Ollama; provider-agnostic layer |
| Free models produce invalid JSON | Pydantic validation, one retry, fallback; per-provider validity measured in E3 |
| Free tiers may train on prompts | Only public paper text goes to free APIs; private notes use Mode A |
| Free-tier quality understates the system | Report results per model; Stage 2 free vs paid table |
| Rate limits make evals slow | Resumable runner, cache-first, small CI subsets |
| AuraDB Free size limits | Paper nodes only for ingested papers; references as IDs in Postgres |
| Scope explosion | Phase gates; personal MVP at Phase 2 |
| LLM memory leaks into evals | Tool-only citations, graph-only gaps, post-cutoff surveys |
| Full-text licensing | arXiv PDFs for personal research; store derived data; don't redistribute full text |

---

## 16. Resume lines (use only once the numbers exist)

- `AIE` `FS` Built Lacuna, an agentic literature-review system over a scholarly knowledge graph of
  N papers; recovered X% of human-curated survey references vs Y% for keyword search, with every
  research-gap claim backed by graph evidence.
- `MLCV` `AIE` Designed a temporal-holdout evaluation for research-gap detection: X% of top-N
  graph-derived gaps were filled by later papers vs Y% for random candidates.
- `AIE` `FDE` Exposed the system as an MCP server so researchers run it from their own Claude or
  ChatGPT; the server does retrieval and graph work, keeping hosting cost at $0.
- `FS` `AIE` Built a provider-agnostic LLM layer with a free-tier fallback chain, response caching,
  per-provider quotas and a spend gate; measured schema validity per provider.
- `FS` FastAPI async job service with durable Redis run state, run caps, OpenTelemetry traces, a
  Nuxt 3 UI and eval-gated CI.

Skills unlocked once built: Neo4j/Cypher, LangGraph, LiteLLM, Redis, arq, OpenTelemetry, knowledge
graphs, entity resolution, MCP server development (local and remote with OAuth), eval-gated CI.

---

## 17. Open decisions

1. Final name.
2. Which 5 surveys for E1 (recent CV surveys with arXiv versions and good OpenAlex reference data).
3. Freeze date T for E2.
4. Which free model to re-freeze FAMM's generator to.
5. Stage 2 trigger numbers (defaults in section 2).
6. LangGraph vs hand-rolled state machine (default LangGraph).
7. Free hosting for the worker (verify Render or alternatives before Phase 5).
