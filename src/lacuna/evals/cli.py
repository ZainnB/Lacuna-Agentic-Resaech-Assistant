"""`lacuna evals e1|e6`."""

from __future__ import annotations

from lacuna.db import connect, migrate, record_eval


def cmd_e1(args) -> None:
    from lacuna.evals.e1 import load_dataset, render_markdown, run_e1, write_results
    from lacuna.ingest import make_clients

    surveys = load_dataset()
    if args.surveys:
        wanted = set(args.surveys)
        surveys = [s for s in surveys if s.openalex_id in wanted]
    with connect() as conn:
        migrate(conn)
        clients = make_clients(conn=conn)
        result = run_e1(clients.openalex, surveys=surveys, seeds=args.seeds)
        record_eval(conn, "e1", result["config_hash"], result["summary"])
    print(render_markdown(result))
    for path in write_results(result):
        print("wrote", path)


def cmd_e6(args) -> None:
    from lacuna.deepread.cli import retrieval_baseline
    from lacuna.deepread.config import load_config
    from lacuna.deepread.store import connect as dr_connect

    cfg = load_config()
    with dr_connect() as conn:
        print(retrieval_baseline(cfg, conn))


def register(sub) -> None:
    p = sub.add_parser("evals", help="run an eval and write evals/results/")
    s = p.add_subparsers(dest="eval", required=True)
    e1 = s.add_parser("e1", help="survey recall: keyword search vs citation snowball")
    e1.add_argument("--surveys", nargs="+", help="OpenAlex ids to restrict to")
    e1.add_argument("--seeds", type=int, default=20)
    e1.set_defaults(fn=cmd_e1)
    e6 = s.add_parser("e6", help="deep-read baseline")
    e6.add_argument("--retrieval-only", action="store_true", default=True)
    e6.set_defaults(fn=cmd_e6)
