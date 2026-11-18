"""MCPServer assembly: tool descriptions, prompts, instructions."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from lacuna.mcp_server.state import State
from lacuna.mcp_server.tools import Tools

INSTRUCTIONS = """\
Lacuna is a literature-review tool server over a scholarly knowledge graph. You plan and write;
Lacuna finds papers, stores evidence chunks and checks citations.
Rules: cite ONLY chunk_ids returned by Lacuna tools in this session (never from memory); ask the
user before approve_papers; run validate_citations before presenting a draft; respect the caps."""

CITE_RULES = """\
Rules for this workflow:
- Cite only chunk_ids that Lacuna tools returned in this session. Never cite a paper or claim from memory.
- Ask the user before approving full-text download (approve_papers). Do not approve on your own.
- Run validate_citations on every chunk_id in a draft before presenting it; fix or drop any that fail.
- Stop at the caps (papers <= 100, chunks per call <= 60, k <= 20). Do not loop to get around them."""


def _schema_summary() -> str:
    try:
        from lacuna.extract.schemas import extraction_json_schema

        schema = extraction_json_schema()
        defs = schema.get("$defs", {})
        parts = []
        for name, prop in schema.get("properties", {}).items():
            node = prop.get("items", prop)
            if "$ref" in node:
                node = defs.get(node["$ref"].split("/")[-1], node)
            fields = ", ".join(node.get("properties", {}))
            parts.append(f"{name}: [{{{fields}}}]" if fields else name)
        return "; ".join(parts)
    except Exception:
        return "see lacuna.extract.schemas.extraction_json_schema()"


def build_server(state: State | None = None) -> MCPServer:
    state = state or State()
    tools = Tools(state)
    server = MCPServer("lacuna", instructions=INSTRUCTIONS)
    server.lacuna_state = state  # type: ignore[attr-defined]
    server.lacuna_tools = tools  # type: ignore[attr-defined]

    def reg(fn, description: str) -> None:
        server.tool(name=fn.__name__, description=description)(fn)

    reg(
        tools.search_papers,
        "Search OpenAlex for papers matching a query. Read-only: nothing is stored. Use it to probe a "
        "topic before scout_topic. Returns paper_id (OpenAlex id), title, year, citations, has_arxiv and "
        "a short abstract. limit <= 25.",
    )
    reg(
        tools.scout_topic,
        "Build a topic: search OpenAlex for seeds, snowball one level of citations, rank, then store the "
        "papers (Postgres + Neo4j) with abstract chunks. Returns topic_id, status and the ranked papers "
        "(rank, role seed/snowball_ref/snowball_cite, paper_id, title, year, citations, has_arxiv). "
        "max_papers <= 100, seeds <= 30. Start here for a literature review.",
    )
    reg(
        tools.approve_papers,
        "HUMAN APPROVAL GATE for full-text download. ASK THE USER which papers to approve before calling "
        "this; never approve on your own initiative. Marks papers approved (paper_ids, or the best `top`) "
        "and, when fetch_fulltext is true, downloads and chunks the arXiv full text (at most 20 per call). "
        "Returns a per-paper status; papers without an arXiv id stay abstract-only.",
    )
    reg(
        tools.get_chunks,
        "Get a paper's chunks (kinds: 'abstract', 'fulltext'; default both) with chunk_ids. Everything "
        "returned is marked as served to this session, which is what allows save_extraction and citations "
        "to use those chunk_ids. At most 60 chunks per call; use offset to page.",
    )
    reg(
        tools.save_extraction,
        "Save YOUR extraction of one paper (Mode A). First call get_chunks for the paper. Every item must "
        "cite a chunk_id returned by get_chunks in this session; the whole payload is rejected otherwise. "
        "`model` is your model name (stored as host:<model>). Extraction shape: "
        f"{_schema_summary()}. Method role is the schema's role enum (see the extraction schema); "
        "confidence is 0-1; result value is a number. Returns {ok, errors, counts}.",
    )
    reg(
        tools.extract_free_tier,
        "Server-side extraction of one paper with the free-tier LLM chain (public paper text only; cached; "
        "rate limited). Use it when you do not want to spend your own tokens on extraction. Citations are "
        "enforced in code. Returns counts and graph writes.",
    )
    reg(
        tools.ask,
        "Semantic search over a topic's chunks. Returns the k best evidence chunks (chunk_id, paper_id, "
        "title, score, text; k <= 20) plus the topic's top entities. YOU write the answer from these chunks "
        "and must cite their chunk_ids. Chunks returned here count as served to this session.",
    )
    reg(
        tools.related,
        "Graph neighbourhood of one paper: what it cites, what cites it (in the stored graph), authors, "
        "and its extracted methods/datasets/tasks/results/limitations. Lists are capped at 25.",
    )
    reg(
        tools.topic_overview,
        "Overview of a topic: paper count, in-set citation count, most-cited papers, approved full-text "
        "count, and the most frequent extracted methods, datasets, tasks and metrics.",
    )
    reg(
        tools.validate_citations,
        "Check chunk_ids before presenting a draft. For each id: exists, paper_id, in_topic (when topic_id "
        "is given) and served_this_session. ok is true only if every id passes. Up to 100 ids. Remove or "
        "re-fetch any citation that fails.",
    )

    @server.prompt(name="scout_topic", description="Workflow: scout a research question into a topic.")
    def scout_prompt(question: str) -> str:
        return f"""Scout a topic for this research question: {question}

Steps:
1. Call search_papers once or twice to check the question returns relevant work.
2. Call scout_topic(question) (max_papers 40, seeds 15 unless the user asks otherwise).
3. Call topic_overview(topic_id) and summarise the ranked papers for the user.
4. Ask the user which papers (if any) they want full text for; only then call approve_papers.
{CITE_RULES}"""

    @server.prompt(name="draft_review", description="Workflow: draft a cited literature review for a topic.")
    def draft_review(topic_id: str) -> str:
        return f"""Draft a literature review for topic {topic_id}.

Steps:
1. topic_overview({topic_id!r}) for the shape of the topic.
2. Plan 3-6 sections. For each, call ask({topic_id!r}, <section question>, k=8) and read the evidence.
3. If evidence is thin, call get_chunks for key papers, or ask the user before approving full text.
4. Write each section using only the returned chunks; cite as [chunk_id].
5. Call validate_citations with every chunk_id in the draft (topic_id={topic_id!r}). Fix failures.
6. Present the draft with a list of papers used. Say what the evidence did not cover.
{CITE_RULES}"""

    @server.prompt(name="deep_read", description="Workflow: read one paper closely and save an extraction.")
    def deep_read(paper_id: str) -> str:
        return f"""Deep-read paper {paper_id}.

Steps:
1. get_chunks({paper_id!r}). If only the abstract exists, tell the user and offer approve_papers.
2. Extract methods, datasets, tasks, metrics, results and limitations, each citing a chunk_id from step 1.
3. save_extraction({paper_id!r}, <extraction>, <your model name>). If rejected, fix the cited chunk_ids.
4. related({paper_id!r}) for context, then summarise for the user with [chunk_id] citations.
{CITE_RULES}"""

    return server
