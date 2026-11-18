"""extract -> resolve -> graph, for one paper or a whole topic."""

from __future__ import annotations

from lacuna.extract.llm import ExtractResult, NoChunksError, extract_paper
from lacuna.extract.resolve import Resolver
from lacuna.extract.schemas import Extraction
from lacuna.graph import entities
from lacuna.llm import LLMError, SpendStageError


def resolve_extraction(resolver: Resolver, extraction: Extraction) -> dict:
    """Map every surface name to a canonical entity id. Returns the dict `graph.entities` writes."""
    r = resolver.resolve
    methods = []
    for m in extraction.methods:
        eid, name, variant = r("method", m.name, m.variant)
        ext_id = ext_name = None
        if m.extends:
            ext_id, ext_name, _ = r("method", m.extends)
        methods.append(
            {
                "entity_id": eid,
                "canonical_name": name,
                "variant": variant,
                "role": m.role,
                "extends_id": ext_id,
                "extends_name": ext_name,
                "chunk_id": m.chunk_id,
                "confidence": m.confidence,
            }
        )

    def simple(items, type_):
        out = []
        for i in items:
            eid, name, _ = r(type_, i.name)
            out.append(
                {"entity_id": eid, "canonical_name": name, "chunk_id": i.chunk_id, "confidence": i.confidence}
            )
        return out

    metrics = []
    for m in extraction.metrics:
        eid, name, _ = r("metric", m.name)
        metrics.append(
            {
                "entity_id": eid,
                "canonical_name": name,
                "higher_is_better": m.higher_is_better,
                "chunk_id": m.chunk_id,
                "confidence": m.confidence,
            }
        )
    results = []
    for x in extraction.results:
        mid, mname, mvar = r("method", x.method)
        did, dname, _ = r("dataset", x.dataset)
        kid, kname, _ = r("metric", x.metric)
        results.append(
            {
                "method_id": mid,
                "method": mname,
                "variant": mvar,
                "dataset_id": did,
                "dataset": dname,
                "metric_id": kid,
                "metric": kname,
                "value": x.value,
                "setting": x.setting,
                "chunk_id": x.chunk_id,
                "confidence": x.confidence,
            }
        )
    return {
        "methods": methods,
        "datasets": simple(extraction.datasets, "dataset"),
        "tasks": simple(extraction.tasks, "task"),
        "metrics": metrics,
        "results": results,
        "limitations": [{"text": x.text, "chunk_id": x.chunk_id} for x in extraction.limitations],
    }


def store_extraction(
    conn,
    repo,
    paper_id: str,
    extraction: Extraction,
    extracted_by: str,
    *,
    resolver: Resolver | None = None,
    extra_props: dict | None = None,
) -> dict:
    """Resolve a validated extraction and write it to the graph. Marks the paper `extracted`."""
    resolver = resolver or Resolver(conn)
    resolved = resolve_extraction(resolver, extraction)
    graph = entities.write_extraction(
        repo, paper_id, resolved, extracted_by=extracted_by, extra_props=extra_props
    )
    if not extraction.is_empty():
        conn.execute("UPDATE papers SET ingest_status = 'extracted' WHERE paper_id = %s", (paper_id,))
    return {"resolution": dict(resolver.stats), "graph": graph}


def extract_and_store(
    conn, repo, paper_id: str, *, chain: list[str] | None = None, resolver: Resolver | None = None
) -> dict:
    res: ExtractResult = extract_paper(conn, paper_id, chain=chain)
    stored = store_extraction(conn, repo, paper_id, res.extraction, res.extracted_by, resolver=resolver)
    return {
        "paper_id": paper_id,
        "extracted_by": res.extracted_by,
        "cached": res.cached,
        "calls": res.calls,
        "tokens": res.tokens,
        "rejected_citations": res.rejected_citations,
        "invalid_items": res.invalid_items,
        "counts": res.extraction.counts(),
        **stored,
    }


def topic_paper_ids(conn, topic_id: str) -> list[str]:
    """Topic papers that have chunks, best rank first."""
    rows = conn.execute(
        "SELECT tp.paper_id FROM topic_papers tp "
        "WHERE tp.topic_id = %s AND EXISTS (SELECT 1 FROM paper_chunks c WHERE c.paper_id = tp.paper_id) "
        "ORDER BY tp.rank NULLS LAST, tp.paper_id",
        (topic_id,),
    ).fetchall()
    return [r["paper_id"] for r in rows]


def extract_topic(
    conn, repo, topic_id: str, *, limit: int | None = None, chain: list[str] | None = None
) -> dict:
    """Extract every chunked paper of a topic. A paper failing on LLM errors is recorded, not fatal;
    a spend-gate refusal stops the run."""
    ids = topic_paper_ids(conn, topic_id)
    if limit is not None:
        ids = ids[:limit]
    resolver = Resolver(conn)
    done, failed = [], []
    for pid in ids:
        try:
            done.append(extract_and_store(conn, repo, pid, chain=chain, resolver=resolver))
        except SpendStageError:
            raise
        except (LLMError, NoChunksError, ValueError) as e:  # ValidationError is a ValueError
            failed.append({"paper_id": pid, "error": f"{type(e).__name__}: {e}"[:300]})
    return {
        "topic_id": topic_id,
        "papers": len(ids),
        "extracted": len(done),
        "failed": failed,
        "cached": sum(d["cached"] for d in done),
        "rejected_citations": sum(d["rejected_citations"] for d in done),
        "invalid_items": sum(d.get("invalid_items", 0) for d in done),
        "calls": sum(d["calls"] for d in done),
        "resolution": dict(resolver.stats),
        "results": done,
    }
