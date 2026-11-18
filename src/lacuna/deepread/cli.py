"""`lacuna deepread ...`: the FAMM-RAG full-text QA module."""

import argparse
import json
import sys
from pathlib import Path

from lacuna.deepread.config import (
    CHUNKS_PATH,
    PAGES_PATH,
    QUESTIONS_PATH,
    RESULTS_DIR,
    index_hash,
    load_config,
    read_jsonl,
)


def cmd_download(cfg, args):
    from lacuna.deepread.ingest import download

    download()


def cmd_ingest(cfg, args):
    from lacuna.deepread.ingest import ingest

    ingest()


def cmd_chunk(cfg, args):
    from lacuna.deepread.chunk import chunk

    chunk(cfg)


def cmd_index(cfg, args):
    from lacuna.deepread.embed import embed_chunks
    from lacuna.deepread.store import connect, load

    chunks = read_jsonl(CHUNKS_PATH)
    vectors = embed_chunks(chunks, cfg["embedding"])
    with connect() as conn:
        load(conn, chunks, vectors, index_hash(cfg))
    print(f"indexed {len(chunks)} chunks (index {index_hash(cfg)})")


def cmd_build(cfg, args):
    for step in (cmd_ingest, cmd_chunk, cmd_index):
        step(cfg, args)


def cmd_ask(cfg, args):
    from lacuna.deepread.embed import embed_queries
    from lacuna.deepread.generate import generate
    from lacuna.deepread.retrieve import search
    from lacuna.deepread.store import connect

    k = cfg["retrieval"]["top_k"]
    with connect() as conn:
        chunks = search(conn, embed_queries([args.question], cfg["embedding"])[0], k)
    for c in chunks:
        preview = " ".join(c["text"].split())[:160]
        print(f"{c['score']:.3f}  {c['id']:<24} [{c['section'] or '-'}]  {preview}")
    out = generate(args.question, chunks, cfg["generator"])
    print(f"\nanswer: {out['answer']}\ncitations: {out['citations']}")
    print(
        f"({out['model']}, {out['input_tokens']} in / {out['output_tokens']} out, stop={out['stop_reason']})"
    )


def load_questions() -> list[dict]:
    from lacuna.deepread.evaluate import validate_questions

    questions = read_jsonl(QUESTIONS_PATH)
    errors = validate_questions(questions)
    if errors:
        raise SystemExit("invalid questions.jsonl:\n  " + "\n  ".join(errors))
    return questions


def cmd_check_questions(cfg, args):
    from collections import Counter

    from lacuna.deepread.evaluate import figure_split_violations, orphan_violations

    questions = load_questions()
    pages = read_jsonl(PAGES_PATH)
    print(f"{len(questions)} questions: {dict(Counter(q['split'] for q in questions))}")
    violations = []
    for label, found in (
        ("orphaned", orphan_violations(questions, pages)),
        ("figure split", figure_split_violations(questions, pages)),
    ):
        for v in found:
            print(f"{label}: {v}")
        violations += found
    if violations:
        sys.exit(1)


def retrieval_baseline(cfg: dict, conn) -> dict:
    """Retrieval-only eval: no generator, no judge, no API. Returns per-split metrics.

    Writes evals/results/e6_retrieval_<index_hash>.{json,md} and records them as eval
    "e6_retrieval". `conn` is deepread's own tuple-row connection (store.connect()).
    """
    from lacuna import db
    from lacuna.deepread.embed import embed_queries
    from lacuna.deepread.evaluate import mean, retrieval_metrics
    from lacuna.deepread.retrieve import search
    from lacuna.deepread.store import stored_index_hash

    expected, actual = index_hash(cfg), stored_index_hash(conn)
    if actual != expected:
        raise SystemExit(
            f"index was built with config {actual}, current config is {expected}; run `lacuna deepread index`"
        )
    k, depth = cfg["retrieval"]["top_k"], max(cfg["retrieval"]["eval_depth"], cfg["retrieval"]["top_k"])
    questions = [q for q in load_questions() if q["split"] != "unanswerable"]
    vectors = embed_queries([q["question"] for q in questions], cfg["embedding"])
    records = []
    for q, v in zip(questions, vectors, strict=True):
        m = retrieval_metrics(search(conn, v, depth), q, k)
        records.append({"id": q["id"], "split": q["split"], **m})

    metrics: dict = {}
    for split in ("text", "table", "figure", "all"):
        rs = records if split == "all" else [r for r in records if r["split"] == split]
        if rs:
            metrics[split] = {
                "n": len(rs),
                f"recall_at_{k}": mean(r["recall_at_k"] for r in rs),
                "mrr": mean(r["rr"] for r in rs),
            }
    result = {"index_hash": expected, "k": k, "depth": depth, "metrics": metrics, "records": records}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = RESULTS_DIR / f"e6_retrieval_{expected}"
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    lines = [
        f"# E6 retrieval baseline, index `{expected}`",
        "",
        f"Retriever `{cfg['embedding']['model']}`, "
        f"chunks {cfg['chunking']['tokens']}/{cfg['chunking']['overlap']}. "
        f"recall@{k}, MRR@{depth}. Unanswerable questions have no gold location and are excluded.",
        "",
        f"| split | n | recall@{k} | MRR@{depth} |",
        "|---|---:|---:|---:|",
    ]
    for split, m in metrics.items():
        lines.append(f"| {split} | {m['n']} | {m[f'recall_at_{k}']} | {m['mrr']} |")
    stem.with_suffix(".md").write_text("\n".join(lines) + "\n")

    with db.connect() as dbconn:
        db.migrate(dbconn)
        db.record_eval(dbconn, "e6_retrieval", expected, metrics)
    return metrics


def cmd_eval(cfg, args):
    from lacuna.deepread.evaluate import render_markdown, run, write_results
    from lacuna.deepread.store import connect

    if args.retrieval_only:
        with connect() as conn:
            metrics = retrieval_baseline(cfg, conn)
        print(json.dumps(metrics, indent=2))
        return
    questions = load_questions()
    if args.ids:
        questions = [q for q in questions if q["id"] in set(args.ids)]
    if args.limit:
        questions = questions[: args.limit]
    with connect() as conn:
        result = run(cfg, questions, conn, judge=not args.no_judge)
    print(render_markdown(result))
    if not (args.limit or args.ids or args.no_write):
        for path in write_results(result):
            print("wrote", path)


def cmd_spotcheck(cfg, args):
    from lacuna.deepread.evaluate import write_spotcheck

    chunks_by_id = {c["id"]: c for c in read_jsonl(CHUNKS_PATH)}
    print("wrote", write_spotcheck(Path(args.result), chunks_by_id, n=args.n))


def cmd_agreement(cfg, args):
    from lacuna.deepread.evaluate import spotcheck_agreement

    print(json.dumps(spotcheck_agreement(args.csv)))


def register(sub) -> None:
    p = sub.add_parser("deepread", help="full-text / table / figure QA over a pinned paper corpus")
    d = p.add_subparsers(dest="deepread_cmd", required=True)

    def add(name, help_, fn):
        sp = d.add_parser(name, help=help_)
        sp.set_defaults(fn=lambda args, fn=fn: fn(load_config(), args))
        return sp

    add("download", "download pinned PDFs from data/papers.csv", cmd_download)
    add("ingest", "extract per-page text", cmd_ingest)
    add("chunk", "chunk pages", cmd_chunk)
    add("index", "embed chunks and load them into Postgres", cmd_index)
    add("build", "ingest + chunk + index", cmd_build)
    sp = add("ask", "retrieve and answer one question (sanity check)", cmd_ask)
    sp.add_argument("question")
    add("check-questions", "validate questions.jsonl and the figure split", cmd_check_questions)
    sp = add("eval", "run the eval; full runs are written to evals/results/", cmd_eval)
    sp.add_argument("--limit", type=int)
    sp.add_argument("--ids", nargs="+")
    sp.add_argument("--no-judge", action="store_true", help="skip LLM judges")
    sp.add_argument("--no-write", action="store_true")
    sp.add_argument("--retrieval-only", action="store_true", help="retrieval metrics only; no API calls")
    sp = add("spotcheck", "sample faithfulness judgements for hand-labelling", cmd_spotcheck)
    sp.add_argument("result")
    sp.add_argument("-n", type=int, default=15)
    sp = add("agreement", "judge/human agreement on a filled spot-check CSV", cmd_agreement)
    sp.add_argument("csv")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="lacuna-deepread")
    sub = parser.add_subparsers(dest="cmd", required=True)
    register(sub)
    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
