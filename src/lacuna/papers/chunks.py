"""Abstract and full-text chunks in `paper_chunks`, approval gate, served-chunk ledger, search.

Reuses deepread's PDF extraction, chunking and embedding code unchanged (frozen config in
config/deepread.toml). Every function takes a psycopg connection from `lacuna.db.connect()`.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
from pgvector.psycopg import register_vector

from lacuna.deepread.config import load_config
from lacuna.settings import get_settings

DEFAULT_KINDS = ("abstract", "fulltext")


def _clean(s: str) -> str:
    return s.replace("\x00", "")


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def embed_texts(texts: list[str], *, query: bool = False) -> np.ndarray:
    """bge-base, normalised. The query instruction prefix goes on queries only. Tests patch this."""
    from lacuna.deepread.embed import get_model

    cfg = load_config()["embedding"]
    model = get_model(cfg["model"], cfg["device"])
    if query:
        texts = [cfg["query_prefix"] + t for t in texts]
    return model.encode(
        texts, batch_size=cfg["batch_size"], normalize_embeddings=True, show_progress_bar=False
    ).astype(np.float32)


def _prep(conn) -> None:
    register_vector(conn)


# ------------------------------------------------------------------ abstracts


def ensure_abstract_chunks(conn, paper_ids: list[str]) -> int:
    """Write `<paper_id>:abs` (title + abstract) for each paper. Unchanged text is skipped."""
    _prep(conn)
    if not paper_ids:
        return 0
    papers = conn.execute(
        "SELECT paper_id, title, abstract FROM papers WHERE paper_id = ANY(%s)", (list(paper_ids),)
    ).fetchall()
    existing = {
        r["chunk_id"]: r["content_hash"]
        for r in conn.execute(
            "SELECT chunk_id, content_hash FROM paper_chunks WHERE chunk_id = ANY(%s)",
            ([f"{p['paper_id']}:abs" for p in papers],),
        )
    }
    todo = []
    for p in papers:
        title, abstract = _clean(p["title"] or ""), _clean(p["abstract"] or "").strip()
        text = f"{title}\n\n{abstract}" if abstract else title
        cid = f"{p['paper_id']}:abs"
        h = _hash(text)
        if existing.get(cid) != h and text.strip():
            todo.append((cid, p["paper_id"], text, h))
    if not todo:
        return 0
    vectors = embed_texts([t[2] for t in todo])
    for (cid, pid, text, h), vec in zip(todo, vectors, strict=True):
        conn.execute(
            "INSERT INTO paper_chunks (chunk_id, paper_id, kind, ord, text, content_hash, embedding) "
            "VALUES (%s, %s, 'abstract', 0, %s, %s, %s) "
            "ON CONFLICT (chunk_id) DO UPDATE SET text = EXCLUDED.text, "
            "content_hash = EXCLUDED.content_hash, embedding = EXCLUDED.embedding",
            (cid, pid, text, h, vec),
        )
    return len(todo)


# ------------------------------------------------------------------ approval gate


def approve(conn, topic_id: str, paper_ids: list[str] | None = None, *, top: int | None = None) -> list[str]:
    """Mark papers approved for full-text download. `top=N` approves the N best-ranked."""
    if paper_ids is None:
        if top is None:
            raise ValueError("approve needs paper_ids or top")
        rows = conn.execute(
            "SELECT paper_id FROM topic_papers WHERE topic_id = %s "
            "ORDER BY rank NULLS LAST, score DESC NULLS LAST, paper_id LIMIT %s",
            (topic_id, top),
        ).fetchall()
        paper_ids = [r["paper_id"] for r in rows]
    rows = conn.execute(
        "UPDATE topic_papers SET approved = true WHERE topic_id = %s AND paper_id = ANY(%s) "
        "RETURNING paper_id",
        (topic_id, list(paper_ids)),
    ).fetchall()
    done = {r["paper_id"] for r in rows}
    return [p for p in paper_ids if p in done]


def approved_papers(conn, topic_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT paper_id FROM topic_papers WHERE topic_id = %s AND approved "
        "ORDER BY rank NULLS LAST, paper_id",
        (topic_id,),
    ).fetchall()
    return [r["paper_id"] for r in rows]


# ------------------------------------------------------------------ full text


def fetch_fulltext(conn, paper_id: str, *, arxiv=None, pdf_dir: Path | None = None) -> str:
    """Download, chunk and embed one approved paper's arXiv PDF.

    Returns "chunked" | "already" | "not_approved" | "no_arxiv" | "failed:<reason>".
    """
    _prep(conn)
    ok = conn.execute(
        "SELECT 1 FROM topic_papers WHERE paper_id = %s AND approved LIMIT 1", (paper_id,)
    ).fetchone()
    if not ok:
        return "not_approved"
    paper = conn.execute(
        "SELECT arxiv_id, ingest_status FROM papers WHERE paper_id = %s", (paper_id,)
    ).fetchone()
    if paper is None:
        return "failed:unknown paper"
    have = conn.execute(
        "SELECT 1 FROM paper_chunks WHERE paper_id = %s AND kind = 'fulltext' LIMIT 1", (paper_id,)
    ).fetchone()
    if have and paper["ingest_status"] in ("chunked", "extracted"):
        return "already"
    from lacuna.ingest.arxiv import normalise_arxiv_id

    aid = normalise_arxiv_id(paper["arxiv_id"] or "")
    if not aid:
        return "no_arxiv"
    try:
        if arxiv is None:
            from lacuna.ingest import make_clients

            arxiv = make_clients(conn=conn).arxiv
        pdf_dir = Path(pdf_dir) if pdf_dir else get_settings().data_dir / "pdfs" / "topics"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf = arxiv.download_pdf(aid, pdf_dir)
        if not Path(pdf).read_bytes()[:5].startswith(b"%PDF"):
            return "failed:not a pdf"
        from lacuna.deepread.chunk import chunk_pages
        from lacuna.deepread.ingest import extract_pages

        pages = extract_pages(Path(pdf))
        for p in pages:
            p["paper_id"] = paper_id
        chunks = chunk_pages(pages, load_config()["chunking"])
        if not chunks:
            return "failed:no text extracted"
        texts = [_clean(c["text"]) for c in chunks]
        vectors = embed_texts(texts)
    except Exception as e:  # noqa: BLE001 - report, never crash a batch
        return f"failed:{type(e).__name__}: {e}"[:300]
    keep = []
    for n, (c, text, vec) in enumerate(zip(chunks, texts, vectors, strict=True), start=1):
        cid = f"{paper_id}:{n}"
        keep.append(cid)
        conn.execute(
            "INSERT INTO paper_chunks (chunk_id, paper_id, kind, ord, page, section, text, content_hash, "
            "embedding) VALUES (%s, %s, 'fulltext', %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (chunk_id) DO UPDATE SET kind = 'fulltext', ord = EXCLUDED.ord, "
            "page = EXCLUDED.page, section = EXCLUDED.section, text = EXCLUDED.text, "
            "content_hash = EXCLUDED.content_hash, embedding = EXCLUDED.embedding",
            (cid, paper_id, n, c["page"], c.get("section"), text, _hash(text), vec),
        )
    conn.execute(
        "DELETE FROM paper_chunks WHERE paper_id = %s AND kind = 'fulltext' AND NOT (chunk_id = ANY(%s))",
        (paper_id, keep),
    )
    conn.execute(
        "UPDATE papers SET ingest_status = 'chunked', updated_at = now() WHERE paper_id = %s", (paper_id,)
    )
    return "chunked"


# ------------------------------------------------------------------ reading and search

_COLS = "c.chunk_id, c.paper_id, c.kind, c.ord, c.page, c.section, c.text"


def _record_served(conn, chunk_ids: list[str], session: str) -> None:
    for cid in chunk_ids:
        conn.execute(
            "INSERT INTO served_chunks (chunk_id, session) VALUES (%s, %s) "
            "ON CONFLICT (chunk_id, session) DO UPDATE SET served_at = now()",
            (cid, session),
        )


def get_chunks(
    conn,
    paper_id: str,
    *,
    kinds: tuple[str, ...] = DEFAULT_KINDS,
    session: str = "default",
    record_served: bool = True,
) -> list[dict]:
    rows = conn.execute(
        f"SELECT {_COLS} FROM paper_chunks c WHERE c.paper_id = %s AND c.kind = ANY(%s) ORDER BY c.ord",
        (paper_id, list(kinds)),
    ).fetchall()
    out = [dict(r) for r in rows]
    if record_served:
        _record_served(conn, [r["chunk_id"] for r in out], session)
    return out


def search_chunks(
    conn,
    query: str,
    *,
    topic_id: str | None = None,
    paper_ids: list[str] | None = None,
    k: int = 8,
    session: str = "default",
    record_served: bool = True,
) -> list[dict]:
    """Cosine search over chunks of a topic's papers and/or the given paper ids."""
    _prep(conn)
    qvec = embed_texts([query], query=True)[0]
    where, params = ["c.embedding IS NOT NULL"], []
    if topic_id is not None:
        where.append("c.paper_id IN (SELECT paper_id FROM topic_papers WHERE topic_id = %s)")
        params.append(topic_id)
    if paper_ids is not None:
        where.append("c.paper_id = ANY(%s)")
        params.append(list(paper_ids))
    rows = conn.execute(
        f"SELECT {_COLS}, p.title, 1 - (c.embedding <=> %s) AS score "
        f"FROM paper_chunks c LEFT JOIN papers p ON p.paper_id = c.paper_id "
        f"WHERE {' AND '.join(where)} ORDER BY c.embedding <=> %s LIMIT %s",
        (qvec, *params, qvec, k),
    ).fetchall()
    out = [dict(r) for r in rows]
    if record_served:
        _record_served(conn, [r["chunk_id"] for r in out], session)
    return out


# ------------------------------------------------------------------ topic pipeline


def index_topic(conn, topic_id: str, *, fulltext: bool = False) -> dict:
    """Abstract chunks for every topic paper; full text for approved ones when `fulltext`."""
    ids = [
        r["paper_id"]
        for r in conn.execute(
            "SELECT paper_id FROM topic_papers WHERE topic_id = %s ORDER BY rank NULLS LAST", (topic_id,)
        )
    ]
    result: dict = {"papers": len(ids), "abstract_chunks_written": ensure_abstract_chunks(conn, ids)}
    if fulltext:
        from lacuna.ingest import make_clients

        arxiv = make_clients(conn=conn).arxiv  # its limiter spaces downloads 3 s apart
        statuses = Counter()
        for pid in approved_papers(conn, topic_id):
            s = fetch_fulltext(conn, pid, arxiv=arxiv)
            statuses[s.split(":")[0] if s.startswith("failed") else s] += 1
            if s.startswith("failed"):
                result.setdefault("failures", {})[pid] = s
        result["fulltext"] = dict(statuses)
    return result
