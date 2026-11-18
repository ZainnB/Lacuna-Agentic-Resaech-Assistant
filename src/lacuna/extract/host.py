"""Mode A: the MCP host extracts and submits; the server validates and refuses unserved citations."""

from __future__ import annotations

from pydantic import ValidationError

from lacuna.extract import store
from lacuna.extract.schemas import Extraction


def _fail(*errors: str) -> dict:
    return {"ok": False, "errors": list(errors), "counts": {}}


def save_host_extraction(
    conn,
    paper_id: str,
    payload: dict,
    *,
    extracted_by: str,
    session: str = "default",
    repo=None,
    resolver=None,
    write_graph: bool = True,
) -> dict:
    """Validate and store a host-side extraction. Never raises on bad input; returns
    {ok, errors, counts}.

    Rejected whole (nothing cached or written) when: `extracted_by` is not "host:<model>"; the
    payload fails the schema; or any cited chunk_id was not served to this `session`, or does not
    belong to `paper_id`. With `write_graph`, resolves entities and writes the graph (a `repo` is
    opened from settings when none is given).
    """
    if not (isinstance(extracted_by, str) and extracted_by.startswith("host:") and len(extracted_by) > 5):
        return _fail('extracted_by must be "host:<model>"')
    try:
        extraction = Extraction.model_validate(payload)
    except ValidationError as e:
        return _fail(*[f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors()][:20])

    cited = extraction.cited_chunk_ids()
    chunks = store.load_chunks(conn, paper_id)
    if not chunks:
        return _fail(f"no chunks stored for paper {paper_id}")
    own = {c["chunk_id"] for c in chunks}
    foreign = sorted(cited - own)
    served = {
        r["chunk_id"]
        for r in conn.execute(
            "/* served_for_session */ SELECT chunk_id FROM served_chunks "
            "WHERE session = %s AND chunk_id = ANY(%s)",
            (session, sorted(cited & own)),
        ).fetchall()
    }
    unserved = sorted((cited & own) - served)
    errors = []
    if foreign:
        errors.append(f"chunk_ids not belonging to paper {paper_id}: {foreign}")
    if unserved:
        errors.append(f"chunk_ids never served to session {session!r}: {unserved}")
    if errors:
        return _fail(*errors)

    store.cache_put(conn, paper_id, store.content_hash(chunks), extracted_by, extraction)
    counts = extraction.counts()
    out = {"ok": True, "errors": [], "counts": counts}
    if write_graph:
        from lacuna.extract.pipeline import store_extraction

        if repo is None:
            from lacuna.graph.repo import GraphRepo

            with GraphRepo.from_settings() as own_repo:
                out["stored"] = store_extraction(
                    conn, own_repo, paper_id, extraction, extracted_by, resolver=resolver
                )
        else:
            out["stored"] = store_extraction(
                conn, repo, paper_id, extraction, extracted_by, resolver=resolver
            )
    return out
