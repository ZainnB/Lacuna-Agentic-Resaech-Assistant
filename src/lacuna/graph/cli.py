"""`lacuna graph ...`: inspect the Neo4j graph from the shell. Output is JSON so it pipes to jq."""

import json
import sys

from lacuna.graph.repo import GraphRepo


def _dump(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def cmd_stats(args) -> None:
    with GraphRepo.from_settings() as repo:
        repo.ensure_constraints()
        _dump(repo.stats())


def cmd_paper(args) -> None:
    with GraphRepo.from_settings() as repo:
        hood = repo.neighbourhood(args.paper_id)
    if hood is None:
        sys.exit(f"no Paper node {args.paper_id}")
    _dump(hood)


def cmd_topic(args) -> None:
    with GraphRepo.from_settings() as repo:
        summary = repo.topic_summary(args.topic_id, top_n=args.top)
    if summary is None:
        sys.exit(f"no Topic node {args.topic_id}")
    _dump(summary)


def cmd_reset(args) -> None:
    if not args.yes:
        sys.exit("refusing to delete the graph without --yes")
    with GraphRepo.from_settings() as repo:
        n = repo.reset(confirm=True)
    print(f"deleted {n} nodes")


def register(sub) -> None:
    p = sub.add_parser("graph", help="Neo4j knowledge graph")
    s = p.add_subparsers(dest="graph_cmd", required=True)
    s.add_parser("stats", help="node and relationship counts").set_defaults(fn=cmd_stats)
    paper = s.add_parser("paper", help="depth-1 neighbourhood of a paper")
    paper.add_argument("paper_id", help="OpenAlex work ID, e.g. W2741809807")
    paper.set_defaults(fn=cmd_paper)
    topic = s.add_parser("topic", help="members and within-set citations of a topic")
    topic.add_argument("topic_id")
    topic.add_argument("--top", type=int, default=10, help="how many top-cited members to list")
    topic.set_defaults(fn=cmd_topic)
    reset = s.add_parser("reset", help="delete every node (dev only)")
    reset.add_argument("--yes", action="store_true", help="required; there is no undo")
    reset.set_defaults(fn=cmd_reset)
