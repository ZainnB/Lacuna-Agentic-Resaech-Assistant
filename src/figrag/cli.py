import argparse
import json
import sys
from pathlib import Path

from figrag.config import CHUNKS_PATH, PAGES_PATH, QUESTIONS_PATH, index_hash, load_config, read_jsonl


def cmd_download(cfg, args):
    from figrag.ingest import download

    download()


def cmd_ingest(cfg, args):
    from figrag.ingest import ingest

    ingest()


def cmd_chunk(cfg, args):
    from figrag.chunk import chunk

    chunk(cfg)


def cmd_index(cfg, args):
    from figrag.embed import embed_chunks
    from figrag.store import connect, load

    chunks = read_jsonl(CHUNKS_PATH)
    vectors = embed_chunks(chunks, cfg["embedding"])
    with connect(cfg["db"]["dsn"]) as conn:
        load(conn, chunks, vectors, index_hash(cfg))
    print(f"indexed {len(chunks)} chunks (index {index_hash(cfg)})")


def cmd_build(cfg, args):
    for step in (cmd_ingest, cmd_chunk, cmd_index):
        step(cfg, args)


def cmd_ask(cfg, args):
    from figrag.embed import embed_queries
    from figrag.generate import generate
    from figrag.retrieve import search
    from figrag.store import connect

    k = cfg["retrieval"]["top_k"]
    with connect(cfg["db"]["dsn"]) as conn:
        chunks = search(conn, embed_queries([args.question], cfg["embedding"])[0], k)
    for c in chunks:
        preview = " ".join(c["text"].split())[:160]
        print(f"{c['score']:.3f}  {c['id']:<24} [{c['section'] or '-'}]  {preview}")
    out = generate(args.question, chunks, cfg["generator"])
    print(f"\nanswer: {out['answer']}\ncitations: {out['citations']}")
    print(f"({out['model']}, {out['input_tokens']} in / {out['output_tokens']} out, stop={out['stop_reason']})")


def load_questions() -> list[dict]:
    from figrag.evaluate import validate_questions

    questions = read_jsonl(QUESTIONS_PATH)
    errors = validate_questions(questions)
    if errors:
        raise SystemExit("invalid questions.jsonl:\n  " + "\n  ".join(errors))
    return questions


def cmd_check_questions(cfg, args):
    from collections import Counter

    from figrag.evaluate import figure_split_violations

    questions = load_questions()
    print(f"{len(questions)} questions: {dict(Counter(q['split'] for q in questions))}")
    violations = figure_split_violations(questions, read_jsonl(PAGES_PATH))
    for v in violations:
        print("figure split:", v)
    if violations:
        sys.exit(1)


def cmd_eval(cfg, args):
    from figrag.evaluate import render_markdown, run, write_results
    from figrag.store import connect

    questions = load_questions()
    if args.ids:
        questions = [q for q in questions if q["id"] in set(args.ids)]
    if args.limit:
        questions = questions[: args.limit]
    with connect(cfg["db"]["dsn"]) as conn:
        result = run(cfg, questions, conn, judge=not args.no_judge)
    print(render_markdown(result))
    if not (args.limit or args.ids or args.no_write):
        for path in write_results(result):
            print("wrote", path)


def cmd_spotcheck(cfg, args):
    from figrag.evaluate import write_spotcheck

    chunks_by_id = {c["id"]: c for c in read_jsonl(CHUNKS_PATH)}
    print("wrote", write_spotcheck(Path(args.result), chunks_by_id, n=args.n))


def cmd_agreement(cfg, args):
    from figrag.evaluate import spotcheck_agreement

    print(json.dumps(spotcheck_agreement(args.csv)))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="figrag")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download", help="download pinned PDFs from data/papers.csv").set_defaults(fn=cmd_download)
    sub.add_parser("ingest", help="extract per-page text").set_defaults(fn=cmd_ingest)
    sub.add_parser("chunk", help="chunk pages").set_defaults(fn=cmd_chunk)
    sub.add_parser("index", help="embed chunks and load them into Postgres").set_defaults(fn=cmd_index)
    sub.add_parser("build", help="ingest + chunk + index").set_defaults(fn=cmd_build)
    p = sub.add_parser("ask", help="retrieve and answer one question (sanity check)")
    p.add_argument("question")
    p.set_defaults(fn=cmd_ask)
    sub.add_parser("check-questions", help="validate questions.jsonl and the figure split").set_defaults(fn=cmd_check_questions)
    p = sub.add_parser("eval", help="run the eval; full runs are written to eval/results/")
    p.add_argument("--limit", type=int)
    p.add_argument("--ids", nargs="+")
    p.add_argument("--no-judge", action="store_true", help="skip LLM judges (faithfulness, free-text answers)")
    p.add_argument("--no-write", action="store_true")
    p.set_defaults(fn=cmd_eval)
    p = sub.add_parser("spotcheck", help="sample faithfulness judgements for hand-labelling")
    p.add_argument("result")
    p.add_argument("-n", type=int, default=15)
    p.set_defaults(fn=cmd_spotcheck)
    p = sub.add_parser("agreement", help="judge/human agreement on a filled spot-check CSV")
    p.add_argument("csv")
    p.set_defaults(fn=cmd_agreement)
    args = parser.parse_args(argv)
    args.fn(load_config(), args)


if __name__ == "__main__":
    main()
