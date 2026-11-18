"""`lacuna ingest ...` subcommands."""

from __future__ import annotations

import json


def _show_record(r) -> None:
    ids = " ".join(x for x in (r.doi and f"doi:{r.doi}", r.arxiv_id and f"arxiv:{r.arxiv_id}") if x)
    print(f"{r.paper_id}  {r.year or '----'}  cites={r.citation_count}  {r.title[:90]}  {ids}")


def cmd_search(args) -> None:
    from lacuna.db import connect
    from lacuna.ingest import make_clients

    with connect() as conn:
        c = make_clients(conn=conn)
        if args.source == "openalex":
            rows = c.openalex.search_works(args.query, max_results=args.limit)
            for r in rows:
                _show_record(r)
        else:
            hits = (
                c.s2.search(args.query, args.limit)
                if args.source == "s2"
                else c.arxiv.search(args.query, args.limit)
            )
            for h in hits:
                print(f"{h.source_id}  {h.year or '----'}  cites={h.citation_count}  {h.title[:90]}  "
                      f"doi:{h.doi} arxiv:{h.arxiv_id}")  # fmt: skip


def cmd_paper(args) -> None:
    from lacuna.db import connect
    from lacuna.ingest import make_clients

    with connect() as conn:
        rec = make_clients(conn=conn).openalex.work(args.id)
    if rec is None:
        raise SystemExit(f"not found: {args.id}")
    d = rec.model_dump(mode="json")
    d["referenced_ids"] = f"{len(rec.referenced_ids)} ids"
    d["abstract"] = (rec.abstract or "")[:300]
    d["authors"] = [a.name for a in rec.authors[:8]]
    print(json.dumps(d, indent=2))


def cmd_usage(args) -> None:
    from lacuna.db import connect, usage_today
    from lacuna.settings import get_settings

    cfg = get_settings()
    with connect() as conn:
        for src in ("openalex", "s2", "arxiv"):
            u = usage_today(conn, "api_usage", source=src)
            cost = float(u["cost_usd"])
            line = f"{src:9} calls={u['calls']} cost_usd={cost:.4f} rate_limited={u['rate_limited']}"
            if src == "openalex":
                key = "yes" if cfg.openalex_api_key else "no"
                line += f"  budget={cfg.openalex_daily_budget_usd:.2f} key={key}"
            print(line)


def register(sub) -> None:
    p = sub.add_parser("ingest", help="paper-API clients (OpenAlex, Semantic Scholar, arXiv)")
    isub = p.add_subparsers(dest="ingest_cmd", required=True)
    s = isub.add_parser("search", help="search one source")
    s.add_argument("query")
    s.add_argument("--source", choices=["openalex", "s2", "arxiv"], default="openalex")
    s.add_argument("--limit", type=int, default=10)
    s.set_defaults(fn=cmd_search)
    q = isub.add_parser("paper", help="free OpenAlex lookup by W id, DOI or arXiv id")
    q.add_argument("id")
    q.set_defaults(fn=cmd_paper)
    isub.add_parser("usage", help="today's API usage ledger").set_defaults(fn=cmd_usage)
