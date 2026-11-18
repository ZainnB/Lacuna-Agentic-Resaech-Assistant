"""Tool implementations. Each returns a compact JSON-able dict and never raises: failures come
back as {"error": "..."}. Every call writes a usage_log row (event "mcp:<tool>", mode "host").

Every cap is enforced here in code, not in the tool descriptions.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from lacuna import db, papers
from lacuna.mcp_server.state import State

log = logging.getLogger("lacuna.mcp")

MAX_SEARCH = 25
MAX_PAPERS = 100
MAX_SEEDS = 30
MAX_CHUNKS = 60
MAX_K = 20
MAX_FULLTEXT_PER_CALL = 20
MAX_VALIDATE = 100
MAX_RELATED = 25
KINDS = ("abstract", "fulltext")
ABSTRACT_CHARS = 600
TITLE_CHARS = 140


def clamp(value, lo: int, hi: int, default: int) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError):
        v = default
    return max(lo, min(hi, v))


def trunc(s: str | None, n: int) -> str | None:
    if s is None:
        return None
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _paper_row(rec, **extra) -> dict:
    return {
        "paper_id": rec.paper_id,
        "title": trunc(rec.title, TITLE_CHARS),
        "year": rec.year,
        "citations": rec.citation_count,
        "has_arxiv": bool(rec.arxiv_id),
        **extra,
    }


def _slim(items: list[dict] | None, cap: int = MAX_RELATED) -> list[dict]:
    out = []
    for x in (items or [])[:cap]:
        x = dict(x)
        if "title" in x:
            x["title"] = trunc(x["title"], TITLE_CHARS)
        out.append(x)
    return out


class Tools:
    def __init__(self, state: State) -> None:
        self.state = state

    # -- plumbing ----------------------------------------------------------------------------
    def _run(self, name: str, fn: Callable[[Any, dict], dict], topic_id: str | None = None) -> dict:
        """Run `fn(conn, detail)` under the state lock; convert exceptions; log usage."""
        detail: dict = {}
        outcome = "ok"
        conn = None
        with self.state.lock:
            try:
                conn = self.state.conn()
                result = fn(conn, detail)
                if isinstance(result, dict) and result.get("error"):
                    outcome = "error"
            except Exception as e:
                log.exception("tool %s failed", name)
                outcome = "error"
                result = {"error": f"{type(e).__name__}: {trunc(str(e), 300)}"}
                detail["error"] = result["error"]
                try:
                    if conn is not None and not conn.closed:
                        conn.rollback()
                except Exception:
                    pass
            if outcome == "error" and "error" not in detail and isinstance(result, dict):
                detail["error"] = trunc(str(result.get("error")), 200)
            topic = detail.pop("topic_id", topic_id)
            try:
                db.log_usage(
                    self.state.conn(), f"mcp:{name}", topic_id=topic, mode="host", outcome=outcome,
                    detail=detail,
                )  # fmt: skip
            except Exception as e:
                log.warning("usage_log write failed: %s", e)
        return result

    # -- discovery ---------------------------------------------------------------------------
    def search_papers(
        self, query: str, limit: int = 10, from_year: int | None = None, to_year: int | None = None
    ) -> dict:
        def work(conn, d):
            from datetime import date

            from lacuna.ingest import make_clients

            n = clamp(limit, 1, MAX_SEARCH, 10)
            d.update(query=trunc(query, 100), limit=n)
            clients = make_clients(conn=conn)
            try:
                recs = clients.openalex.search_works(
                    query,
                    from_date=date(int(from_year), 1, 1) if from_year else None,
                    to_date=date(int(to_year), 12, 31) if to_year else None,
                    max_results=n,
                )
            finally:
                clients.close()
            d["n"] = len(recs)
            return {
                "papers": [
                    _paper_row(r, abstract=trunc(r.abstract, ABSTRACT_CHARS), venue=trunc(r.venue, 80))
                    for r in recs[:n]
                ]
            }

        return self._run("search_papers", work)

    def scout_topic(self, question: str, max_papers: int = 40, seeds: int = 15) -> dict:
        def work(conn, d):
            from lacuna import scout as scout_pkg
            from lacuna.ingest import make_clients
            from lacuna.scout import store

            cfg = scout_pkg.ScoutConfig(
                max_papers=clamp(max_papers, 1, MAX_PAPERS, 40), seeds=clamp(seeds, 1, MAX_SEEDS, 15)
            )
            d.update(max_papers=cfg.max_papers, seeds=cfg.seeds)
            clients = make_clients(conn=conn)
            try:
                result = scout_pkg.scout(question, clients.openalex, cfg)
            finally:
                clients.close()
            d["status"] = result.status
            if not result.papers:
                return {"topic_id": None, "status": result.status, "papers": [], "notes": result.notes[:5]}
            repo = self.state.repo()
            topic_id = store.persist(conn, result, cfg, repo)
            d["topic_id"] = topic_id
            ids = [r.record.paper_id for r in result.papers]
            chunks = papers.ensure_abstract_chunks(conn, ids)
            d["n_papers"] = len(ids)
            return {
                "topic_id": topic_id,
                "status": result.status,
                "n_papers": len(ids),
                "abstract_chunks_written": chunks,
                "notes": result.notes[:5],
                "papers": [
                    _paper_row(r.record, rank=r.rank, role=r.role) for r in result.papers[:MAX_PAPERS]
                ],
            }

        return self._run("scout_topic", work)

    def approve_papers(
        self,
        topic_id: str,
        paper_ids: list[str] | None = None,
        top: int | None = None,
        fetch_fulltext: bool = True,
    ) -> dict:
        def work(conn, d):
            d["topic_id"] = topic_id
            if not paper_ids and not top:
                return {"error": "give paper_ids or top"}
            ids = list(paper_ids)[:MAX_PAPERS] if paper_ids else None
            top_n = clamp(top, 1, MAX_PAPERS, 10) if top else None
            approved = papers.approve(conn, topic_id, ids, top=top_n)
            d["approved"] = len(approved)
            out: dict = {"topic_id": topic_id, "approved": approved, "fulltext": {}}
            if not fetch_fulltext or not approved:
                return out
            from lacuna.ingest import make_clients

            clients = make_clients(conn=conn)
            try:
                for i, pid in enumerate(approved):
                    if i >= MAX_FULLTEXT_PER_CALL:
                        out["fulltext"][pid] = "skipped_cap"
                        continue
                    try:
                        out["fulltext"][pid] = str(papers.fetch_fulltext(conn, pid, arxiv=clients.arxiv))
                    except Exception as e:  # one bad paper must not lose the rest
                        out["fulltext"][pid] = f"failed: {trunc(str(e), 120)}"
            finally:
                clients.close()
            d["fulltext"] = len(out["fulltext"])
            return out

        return self._run("approve_papers", work, topic_id)

    # -- evidence ----------------------------------------------------------------------------
    def get_chunks(self, paper_id: str, kinds: list[str] | None = None, offset: int = 0) -> dict:
        def work(conn, d):
            ks = tuple(k for k in (kinds or KINDS) if k in KINDS)
            if not ks:
                return {"error": f"kinds must be a subset of {list(KINDS)}"}
            rows = papers.get_chunks(
                conn, paper_id, kinds=ks, session=self.state.session, record_served=False
            )
            start = clamp(offset, 0, len(rows), 0)
            page = rows[start : start + MAX_CHUNKS]
            self.state.mark_served(conn, [r["chunk_id"] for r in page])
            d.update(paper_id=paper_id, n=len(page), total=len(rows))
            if not rows:
                return {"paper_id": paper_id, "chunks": [], "total": 0, "note": "no chunks stored"}
            return {
                "paper_id": paper_id,
                "total": len(rows),
                "offset": start,
                "truncated": start + len(page) < len(rows),
                "chunks": [
                    {k: r.get(k) for k in ("chunk_id", "kind", "page", "section", "text")} for r in page
                ],
            }

        return self._run("get_chunks", work)

    def save_extraction(self, paper_id: str, extraction: dict, model: str) -> dict:
        def work(conn, d):
            from lacuna.extract.host import save_host_extraction

            model_s = str(model or "").strip()
            if not model_s:
                return {"error": "model is required (your own model name), e.g. 'claude-sonnet-5'"}
            d.update(paper_id=paper_id, model=model_s)
            res = save_host_extraction(
                conn,
                paper_id,
                extraction,
                extracted_by=f"host:{model_s}",
                session=self.state.session,
                repo=self.state.repo(),
            )
            d["ok"] = bool(res.get("ok"))
            if not res.get("ok"):
                return {"error": "extraction rejected", **res}
            return res

        return self._run("save_extraction", work)

    def extract_free_tier(self, paper_id: str) -> dict:
        def work(conn, d):
            from lacuna.extract import pipeline

            d["paper_id"] = paper_id
            res = pipeline.extract_and_store(conn, self.state.repo(), paper_id, chain=None)
            return {k: res.get(k) for k in ("paper_id", "extracted_by", "cached", "calls", "counts")} | {
                "rejected_citations": res.get("rejected_citations"),
                "graph": res.get("graph"),
            }

        return self._run("extract_free_tier", work)

    def ask(self, topic_id: str, question: str, k: int = 8) -> dict:
        def work(conn, d):
            n = clamp(k, 1, MAX_K, 8)
            d.update(k=n, question=trunc(question, 100), topic_id=topic_id)
            rows = papers.search_chunks(conn, question, topic_id=topic_id, k=n, session=self.state.session)[
                :n
            ]
            out: dict = {
                "topic_id": topic_id,
                "evidence": [
                    {
                        "chunk_id": r["chunk_id"],
                        "paper_id": r["paper_id"],
                        "title": trunc(r.get("title"), TITLE_CHARS),
                        "score": round(float(r.get("score") or 0), 4),
                        "text": r["text"],
                    }
                    for r in rows
                ],
                "instructions": "Answer only from these chunks and cite their chunk_ids.",
            }
            try:
                from lacuna.graph import entities

                out["top_entities"] = entities.topic_entities(self.state.repo(), topic_id, top=8)
            except Exception as e:
                out["top_entities"] = None
                out["entities_error"] = trunc(str(e), 120)
            d["n"] = len(rows)
            return out

        return self._run("ask", work, topic_id)

    def related(self, paper_id: str) -> dict:
        def work(conn, d):
            from lacuna.graph import entities

            d["paper_id"] = paper_id
            repo = self.state.repo()
            hood = repo.neighbourhood(paper_id)
            if hood is None:
                return {"error": f"paper {paper_id} is not in the graph (scout a topic first)"}
            p = dict(hood["paper"])
            p["title"] = trunc(p.get("title"), TITLE_CHARS)
            p.pop("abstract", None)
            return {
                "paper": {k: p.get(k) for k in ("paper_id", "title", "year", "citation_count")},
                "cites": _slim(hood["cites"]),
                "cited_by": _slim(hood["cited_by"]),
                "n_cites": len(hood["cites"] or []),
                "n_cited_by": len(hood["cited_by"] or []),
                "authors": (hood["authors"] or [])[:10],
                "entities": entities.paper_entities(repo, paper_id),
            }

        return self._run("related", work)

    def topic_overview(self, topic_id: str) -> dict:
        def work(conn, d):
            from lacuna.graph import entities

            d["topic_id"] = topic_id
            repo = self.state.repo()
            summary = repo.topic_summary(topic_id, top_n=10)
            if summary is None:
                return {"error": f"unknown topic {topic_id}"}
            ents = entities.topic_entities(repo, topic_id, top=10)
            counts = conn.execute(
                "SELECT count(*) AS papers, count(*) FILTER (WHERE approved) AS approved "
                "FROM topic_papers WHERE topic_id = %s",
                (topic_id,),
            ).fetchone()
            return {
                "topic_id": topic_id,
                "question": trunc(summary["question"], 300),
                "n_papers": summary["n_papers"],
                "n_cites_within": summary["n_cites_within"],
                "n_approved_fulltext": (counts or {}).get("approved", 0),
                "most_cited": _slim(summary["top_cited"], 10),
                "methods": ents["methods"],
                "datasets": ents["datasets"],
                "tasks": ents["tasks"],
                "metrics": ents["metrics"],
            }

        return self._run("topic_overview", work, topic_id)

    def validate_citations(self, chunk_ids: list[str], topic_id: str | None = None) -> dict:
        def work(conn, d):
            ids = list(dict.fromkeys(str(c) for c in (chunk_ids or [])))[:MAX_VALIDATE]
            d.update(n=len(ids), topic_id=topic_id)
            if not ids:
                return {"error": "chunk_ids is empty"}
            rows = {
                r["chunk_id"]: r["paper_id"]
                for r in conn.execute(
                    "SELECT chunk_id, paper_id FROM paper_chunks WHERE chunk_id = ANY(%s)", (ids,)
                ).fetchall()
            }
            served = {
                r["chunk_id"]
                for r in conn.execute(
                    "SELECT chunk_id FROM served_chunks WHERE session = %s AND chunk_id = ANY(%s)",
                    (self.state.session, ids),
                ).fetchall()
            }
            in_topic: set[str] = set()
            if topic_id:
                in_topic = {
                    r["paper_id"]
                    for r in conn.execute(
                        "SELECT paper_id FROM topic_papers WHERE topic_id = %s AND paper_id = ANY(%s)",
                        (topic_id, sorted(set(rows.values()))),
                    ).fetchall()
                }
            results = []
            for cid in ids:
                pid = rows.get(cid)
                results.append(
                    {
                        "chunk_id": cid,
                        "exists": pid is not None,
                        "paper_id": pid,
                        "in_topic": (pid in in_topic) if topic_id and pid else None,
                        "served_this_session": cid in served,
                    }
                )
            ok = all(
                r["exists"] and r["served_this_session"] and (r["in_topic"] is not False) for r in results
            )
            d["ok"] = ok
            return {"ok": ok, "results": results, "n_invalid": sum(1 for r in results if not _valid(r))}

        return self._run("validate_citations", work, topic_id)


def _valid(r: dict) -> bool:
    return bool(r["exists"] and r["served_this_session"] and r["in_topic"] is not False)
