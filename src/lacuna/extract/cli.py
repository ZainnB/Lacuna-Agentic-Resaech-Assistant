"""`lacuna extract ...`: run extraction from the shell. Output is JSON."""

import json
import sys

from lacuna.db import connect, migrate


def _dump(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_paper(args) -> None:
    from lacuna.extract.pipeline import extract_and_store
    from lacuna.graph.repo import GraphRepo

    with connect() as conn, GraphRepo.from_settings() as repo:
        migrate(conn)
        repo.ensure_constraints()
        from lacuna.graph.entities import ensure_entity_constraints

        ensure_entity_constraints(repo)
        _dump(extract_and_store(conn, repo, args.paper_id, chain=args.chain))


def cmd_topic(args) -> None:
    from lacuna.extract.pipeline import extract_topic
    from lacuna.graph.entities import ensure_entity_constraints
    from lacuna.graph.repo import GraphRepo

    with connect() as conn, GraphRepo.from_settings() as repo:
        migrate(conn)
        repo.ensure_constraints()
        ensure_entity_constraints(repo)
        out = extract_topic(conn, repo, args.topic_id, limit=args.limit, chain=args.chain)
    out.pop("results", None)
    _dump(out)


def cmd_throughput(args) -> None:
    from lacuna.extract.pipeline import topic_paper_ids
    from lacuna.extract.throughput import measure
    from lacuna.llm.config import load_config

    specs = args.specs or load_config().chain_for("extractor")
    with connect() as conn:
        migrate(conn)
        ids = topic_paper_ids(conn, args.topic_id)[: args.papers]
        if not ids:
            sys.exit(f"no chunked papers in topic {args.topic_id}")
        _dump(measure(conn, ids, specs))


def register(sub) -> None:
    p = sub.add_parser("extract", help="entity extraction over paper chunks")
    s = p.add_subparsers(dest="extract_cmd", required=True)

    paper = s.add_parser("paper", help="extract one paper into the graph")
    paper.add_argument("paper_id")
    paper.add_argument("--chain", nargs="+", metavar="SPEC", help="e.g. gemini/gemini-3.5-flash-lite")
    paper.set_defaults(fn=cmd_paper)

    topic = s.add_parser("topic", help="extract every chunked paper of a topic")
    topic.add_argument("topic_id")
    topic.add_argument("--limit", type=int)
    topic.add_argument("--chain", nargs="+", metavar="SPEC")
    topic.set_defaults(fn=cmd_topic)

    tp = s.add_parser("throughput", help="per-provider throughput and schema validity (spends quota)")
    tp.add_argument("topic_id")
    tp.add_argument("--papers", type=int, default=10)
    tp.add_argument("--specs", nargs="+", metavar="SPEC", help="default: the extractor chain")
    tp.set_defaults(fn=cmd_throughput)
