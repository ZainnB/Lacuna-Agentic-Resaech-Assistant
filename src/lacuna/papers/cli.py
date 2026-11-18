"""`lacuna papers ...`: index, approve, read and search topic-paper chunks."""

from __future__ import annotations

from lacuna.db import connect, migrate
from lacuna.papers.chunks import approve, get_chunks, index_topic, search_chunks


def cmd_index(args) -> None:
    with connect() as conn:
        migrate(conn)
        print(index_topic(conn, args.topic, fulltext=args.fulltext))


def cmd_approve(args) -> None:
    with connect() as conn:
        migrate(conn)
        ids = approve(conn, args.topic, args.ids, top=args.top)
    print(f"approved {len(ids)}:")
    for i in ids:
        print(" ", i)


def cmd_chunks(args) -> None:
    with connect() as conn:
        for c in get_chunks(conn, args.paper_id, record_served=False):
            loc = f"p{c['page']}" if c["page"] else "abs"
            print(f"[{c['chunk_id']}] {c['kind']} {loc} {c['section'] or ''}\n{c['text'][:400]}\n")


def cmd_search(args) -> None:
    with connect() as conn:
        for r in search_chunks(conn, args.query, topic_id=args.topic, k=args.k, record_served=False):
            print(f"{r['score']:.3f}  [{r['chunk_id']}] {r['title']}\n    {r['text'][:200]!r}")


def register(sub) -> None:
    p = sub.add_parser("papers", help="topic paper chunks: index, approve, read, search")
    s = p.add_subparsers(dest="papers_cmd", required=True)
    a = s.add_parser("index", help="embed abstracts (and approved full text with --fulltext)")
    a.add_argument("topic")
    a.add_argument("--fulltext", action="store_true")
    a.set_defaults(fn=cmd_index)
    a = s.add_parser("approve", help="approve papers for full-text download")
    a.add_argument("topic")
    g = a.add_mutually_exclusive_group(required=True)
    g.add_argument("--top", type=int)
    g.add_argument("--ids", nargs="+")
    a.set_defaults(fn=cmd_approve)
    a = s.add_parser("chunks", help="print a paper's chunks")
    a.add_argument("paper_id")
    a.set_defaults(fn=cmd_chunks)
    a = s.add_parser("search", help="semantic search over a topic's chunks")
    a.add_argument("topic")
    a.add_argument("query")
    a.add_argument("-k", type=int, default=8)
    a.set_defaults(fn=cmd_search)
