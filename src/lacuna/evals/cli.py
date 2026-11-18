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


def cmd_e3(args) -> None:
    from lacuna.evals import e3

    with connect() as conn:
        migrate(conn)
        result = e3.run_e3(conn, split=args.split, chain=args.chain or None, scope=args.scope)
        if result["scores"] is None:
            c = result["label_counts"]
            print(
                f"No scored labels for split={args.split} scope={args.scope}. Labels on disk: "
                f"{c['total']} ({c['tuning']} tuning, {c['reporting']} reporting). No scores are reported."
            )
            print("\nSchema validity today\n" + e3.validity_table(result["schema_validity_today"]))
            return
        record_eval(conn, "e3", result["config_hash"], result)
    print(e3.render_markdown(result))
    for path in e3.write_results(result):
        print("wrote", path)


def cmd_e3_template(args) -> None:
    from lacuna.evals import e3

    with connect() as conn:
        migrate(conn)
        written, split = e3.write_template(conn, args.topic_id, args.n)
    n = {s: sum(1 for v in split.values() if v == s) for s in e3.SPLITS}
    print(f"wrote {len(written)} new templates to evals/datasets/e3/todo/ (split: {n}); split.json updated")


def cmd_e3_validity(args) -> None:
    from lacuna.evals import e3

    with connect() as conn:
        migrate(conn)
        print(e3.validity_table(e3.schema_validity(conn)))


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
    e3 = s.add_parser("e3", help="extraction P/R/F1 per entity type on hand labels")
    e3.add_argument("--split", choices=["reporting", "tuning", "all"], default="reporting")
    e3.add_argument("--chain", nargs="+", help="model specs, e.g. groq/openai/gpt-oss-120b")
    e3.add_argument("--scope", choices=["abstract", "fulltext"], default="abstract")
    e3.set_defaults(fn=cmd_e3)
    t = s.add_parser("e3-template", help="write empty labelling templates for a topic's top papers")
    t.add_argument("topic_id")
    t.add_argument("--n", type=int, default=30)
    t.set_defaults(fn=cmd_e3_template)
    v = s.add_parser("e3-validity", help="schema-validity per provider/model, all days")
    v.set_defaults(fn=cmd_e3_validity)
    e6 = s.add_parser("e6", help="deep-read baseline")
    e6.add_argument("--retrieval-only", action="store_true", default=True)
    e6.set_defaults(fn=cmd_e6)
