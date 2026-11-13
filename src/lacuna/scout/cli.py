"""`lacuna scout "<question>"`: build and store a paper set for a research question."""

from __future__ import annotations

import json
from datetime import date

from lacuna.db import connect, log_usage, migrate
from lacuna.scout.core import ScoutConfig, scout
from lacuna.scout.store import persist


def cmd_scout(args) -> None:
    from lacuna.ingest import make_clients

    cfg = ScoutConfig(
        max_papers=args.max_papers,
        seeds=args.seeds,
        depth=args.depth,
        from_date=date.fromisoformat(args.from_date) if args.from_date else None,
        to_date=date.fromisoformat(args.to_date) if args.to_date else None,
    )
    with connect() as conn:
        migrate(conn)
        clients = make_clients(conn=conn)
        result = scout(args.question, clients.openalex, cfg)
        if args.json:
            print(
                json.dumps(
                    {
                        "status": result.status,
                        "n_seeds": result.n_seeds,
                        "notes": result.notes,
                        "papers": [
                            {
                                "rank": r.rank,
                                "role": r.role,
                                "score": round(r.score, 3),
                                "paper_id": r.record.paper_id,
                                "year": r.record.year,
                                "citations": r.record.citation_count,
                                "title": r.record.title,
                            }
                            for r in result.papers
                        ],
                    },
                    indent=2,
                )
            )
        else:
            print(
                f"status={result.status} seeds={result.n_seeds} "
                f"ref_candidates={result.n_ref_candidates} cite_candidates={result.n_cite_candidates}"
            )
            for r in result.papers:
                print(
                    f"{r.rank:>3} {r.role:<13} {r.score:6.2f} {r.record.paper_id:<12} "
                    f"{r.record.year or '----'} {(r.record.citation_count or 0):>6}  {r.record.title[:80]}"
                )
            for n in result.notes:
                print("note:", n)
        if args.dry_run or not result.papers:
            return
        graph = None
        if not args.no_graph:
            from lacuna.graph.repo import GraphRepo

            graph = GraphRepo.from_settings()
        try:
            topic_id = persist(conn, result, cfg, graph)
        finally:
            if graph is not None:
                graph.close()
        log_usage(
            conn,
            "scout",
            topic_id=topic_id,
            mode="cli",
            outcome=result.status,
            detail={"n_papers": len(result.papers)},
        )
        print(f"\nsaved topic {topic_id} ({len(result.papers)} papers)")


def register(sub) -> None:
    p = sub.add_parser("scout", help="seed search + citation snowball -> ranked paper set")
    p.add_argument("question")
    p.add_argument("--max-papers", type=int, default=60)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--depth", type=int, default=1, choices=[0, 1])
    p.add_argument("--from-date", help="YYYY-MM-DD")
    p.add_argument("--to-date", help="YYYY-MM-DD; nothing published after this date")
    p.add_argument("--no-graph", action="store_true", help="skip Neo4j")
    p.add_argument("--dry-run", action="store_true", help="print the set, store nothing")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_scout)
