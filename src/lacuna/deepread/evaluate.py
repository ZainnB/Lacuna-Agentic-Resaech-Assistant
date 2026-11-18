"""Phase 0 eval harness: retrieval (recall@k, MRR), answer accuracy, faithfulness.

Gold evidence is labelled by location (paper_id, page, evidence_ref, answer_span),
never by chunk ID, and mapped to chunks here at eval time.
"""

import csv
import json
import random
import re
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from lacuna.deepread import generate as gen
from lacuna.deepread.config import RESULTS_DIR, index_hash, stable_hash

SPLITS = ["text", "table", "figure", "unanswerable"]
ANSWER_TYPES = ["numeric", "short_span", "free_text", "unanswerable"]
REQUIRED_FIELDS = ["id", "question", "answer", "answer_type", "split"]
LOCATION_FIELDS = ["paper_id", "page", "evidence_ref", "answer_span"]
NOT_FOUND = "not found"

# ---------------------------------------------------------------- normalisation

NUMBER = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def norm_text(s: str) -> str:
    """For matching spans against PDF-extracted text."""
    s = unicodedata.normalize("NFKC", s).replace("−", "-")
    s = re.sub(r"-\n(?=[a-z])", "", s)  # re-join hyphenated line breaks
    return re.sub(r"\s+", " ", s).lower().strip()


def norm_answer(s: str) -> str:
    """SQuAD-style: lowercase, strip punctuation and articles."""
    s = norm_text(s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def contains_span(text: str, span: str) -> bool:
    """Boundary-aware containment, so "71.4" doesn't match "171.45" and
    "Table 3" doesn't match "Table 31"."""
    span = norm_text(span)
    if not span:
        return False
    pattern = re.escape(span)
    if span[0].isalnum():
        pattern = r"(?<![\w.])" + pattern
    if span[-1].isalnum():
        pattern += r"(?!\w|\.\d)"
    return re.search(pattern, norm_text(text)) is not None


def numbers(s: str) -> list[float]:
    s = unicodedata.normalize("NFKC", s).replace("−", "-")
    return [float(n.replace(",", "")) for n in NUMBER.findall(s)]


# ---------------------------------------------------------------- questions


def validate_questions(questions: list[dict]) -> list[str]:
    errors, seen = [], set()
    for q in questions:
        qid = q.get("id", "?")
        missing = [f for f in REQUIRED_FIELDS if f not in q]
        if missing:
            errors.append(f"{qid}: missing {missing}")
            continue
        if qid in seen:
            errors.append(f"{qid}: duplicate id")
        seen.add(qid)
        if q["split"] not in SPLITS:
            errors.append(f"{qid}: bad split {q['split']!r}")
        if q["answer_type"] not in ANSWER_TYPES:
            errors.append(f"{qid}: bad answer_type {q['answer_type']!r}")
        if (q["split"] == "unanswerable") != (q["answer_type"] == "unanswerable"):
            errors.append(f"{qid}: split and answer_type must both be 'unanswerable' or neither")
        if q["split"] == "unanswerable":
            if norm_answer(q["answer"]) != NOT_FOUND:
                errors.append(f"{qid}: unanswerable question must have answer 'not found'")
            continue
        missing = [f for f in LOCATION_FIELDS if not q.get(f)]
        if missing:
            errors.append(f"{qid}: missing location fields {missing}")
        if q["answer_type"] == "numeric" and len(numbers(str(q["answer"]))) != 1:
            errors.append(f"{qid}: numeric answer must contain exactly one number")
    return errors


def orphan_violations(questions: list[dict], pages: list[dict]) -> list[str]:
    """Questions whose gold location is no longer in the corpus, which happens
    when a paper is dropped from papers.csv. These would otherwise score as
    permanent retrieval misses rather than as a labelling error."""
    located = {(p["paper_id"], p["page"]) for p in pages}
    papers = {p["paper_id"] for p in pages}
    out = []
    for q in questions:
        if q["split"] == "unanswerable":
            continue
        if q["paper_id"] not in papers:
            out.append(f"{q['id']}: paper {q['paper_id']} is not in the corpus")
        elif (q["paper_id"], q["page"]) not in located:
            out.append(f"{q['id']}: {q['paper_id']} page {q['page']} was not extracted")
    return out


def figure_split_violations(questions: list[dict], pages: list[dict]) -> list[str]:
    """A figure question whose answer_span appears in the extracted text is not
    figure-only; move it to text/table. Set "span_check_override" with a reason
    when a hit is a coincidental match you have checked by hand."""
    paper_text = defaultdict(str)
    for p in pages:
        paper_text[p["paper_id"]] += "\n" + p["text"]
    out = []
    for q in questions:
        if q["split"] != "figure" or q.get("span_check_override"):
            continue
        if q["paper_id"] not in paper_text:
            out.append(f"{q['id']}: paper {q['paper_id']} not in extracted pages")
        elif contains_span(paper_text[q["paper_id"]], q["answer_span"]):
            out.append(f"{q['id']}: answer_span {q['answer_span']!r} appears in extracted text")
    return out


# ---------------------------------------------------------------- scoring


def is_hit(chunk: dict, q: dict) -> bool:
    if chunk["paper_id"] != q["paper_id"]:
        return False
    if contains_span(chunk["text"], q["answer_span"]):
        return True
    return chunk["page"] == q["page"] and contains_span(chunk["text"], q["evidence_ref"])


def retrieval_metrics(retrieved: list[dict], q: dict, k: int) -> dict:
    ranks = [i for i, c in enumerate(retrieved, start=1) if is_hit(c, q)]
    first = ranks[0] if ranks else None
    return {
        "hit_ranks": ranks,
        "recall_at_k": float(first is not None and first <= k),
        "rr": 1.0 / first if first else 0.0,
    }


def score_answer(pred: str | None, q: dict) -> tuple[bool | None, str]:
    """Returns (correct, method). correct is None when an LLM judge is needed."""
    if pred is None:
        return False, "no_answer"
    abstained = norm_answer(pred) == NOT_FOUND
    if q["answer_type"] == "unanswerable":
        return abstained, "abstention"
    if abstained:
        return False, "abstained"
    if q["answer_type"] == "numeric":
        got = numbers(pred)
        return bool(got) and abs(got[0] - numbers(str(q["answer"]))[0]) < 1e-9, "numeric_first"
    if q["answer_type"] == "short_span":
        gold, p = norm_answer(q["answer"]), norm_answer(pred)
        if gold == p:
            return True, "exact"
        return f" {gold} " in f" {p} ", "contains"
    return None, "llm_judge"


# ---------------------------------------------------------------- LLM judges

ANSWER_JUDGE_PROMPT = """You grade answers to questions about research papers against a reference answer.

The predicted answer is correct if it states the same key information as the reference. \
Extra detail is fine unless it contradicts the reference. Paraphrasing is fine. \
An answer that is vague, partial, or hedges between options is incorrect."""

FAITHFULNESS_PROMPT = """You check whether an answer is supported by the context excerpts it was generated from.

The answer is supported only if every factual claim in it, including every number, \
appears in the excerpts or follows directly from them. Judge support, not real-world truth: \
a correct answer that the excerpts do not contain is unsupported."""  # noqa: E501

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"}, "verdict": {"type": "boolean"}},
    "required": ["reasoning", "verdict"],
    "additionalProperties": False,
}


def judge_answer(q: dict, pred: str, cfg: dict) -> dict:
    user = f"Question: {q['question']}\nReference answer: {q['answer']}\nPredicted answer: {pred}"
    r = gen.call_json(ANSWER_JUDGE_PROMPT, user, VERDICT_SCHEMA, cfg, role="judge")
    parsed = r.pop("parsed") or {}
    return {"verdict": parsed.get("verdict"), "reasoning": parsed.get("reasoning"), **r}


def judge_faithfulness(pred: str, chunks: list[dict], cfg: dict) -> dict:
    user = f"<context>\n{gen.format_context(chunks)}\n</context>\n\nAnswer: {pred}"
    r = gen.call_json(FAITHFULNESS_PROMPT, user, VERDICT_SCHEMA, cfg, role="judge")
    parsed = r.pop("parsed") or {}
    return {"verdict": parsed.get("verdict"), "reasoning": parsed.get("reasoning"), **r}


# ---------------------------------------------------------------- run


def run_hash(cfg: dict) -> str:
    frozen = {k: v for k, v in cfg.items() if k != "db"}
    return stable_hash(
        frozen, gen.PROMPT_VERSION, gen.SYSTEM_PROMPT, ANSWER_JUDGE_PROMPT, FAITHFULNESS_PROMPT
    )


def evaluate_one(q: dict, retrieved: list[dict], cfg: dict, judge: bool) -> dict:
    k = cfg["retrieval"]["top_k"]
    context = retrieved[:k]
    out = gen.generate(q["question"], context, cfg["generator"])
    pred = out["answer"]
    correct, method = score_answer(pred, q)
    record = {
        "id": q["id"],
        "split": q["split"],
        "answer_type": q["answer_type"],
        "question": q["question"],
        "gold": q["answer"],
        "pred": pred,
        "citations": out["citations"],
        "stop_reason": out["stop_reason"],
        "model": out["model"],
        "correct": correct,
        "match": method,
        "retrieved": [{"id": c["id"], "score": round(float(c["score"]), 4)} for c in retrieved],
        "tokens": {"generator": [out["input_tokens"], out["output_tokens"]]},
        "faithful": None,
    }
    if q["split"] != "unanswerable":
        record.update(retrieval_metrics(retrieved, q, k))
    if not judge:
        return record
    if correct is None:
        v = judge_answer(q, pred, cfg["judge"])
        record["correct"], record["answer_judge"] = v["verdict"], v["reasoning"]
        record["tokens"]["answer_judge"] = [v["input_tokens"], v["output_tokens"]]
    if pred is not None and norm_answer(pred) != NOT_FOUND:
        v = judge_faithfulness(pred, context, cfg["judge"])
        record["faithful"], record["faithfulness_judge"] = v["verdict"], v["reasoning"]
        record["tokens"]["faithfulness_judge"] = [v["input_tokens"], v["output_tokens"]]
    return record


def mean(xs) -> float | None:
    xs = [float(x) for x in xs if x is not None]
    return round(sum(xs) / len(xs), 3) if xs else None


def summarise(records: list[dict]) -> dict:
    by_split = defaultdict(list)
    for r in records:
        by_split[r["split"]].append(r)
    summary = {}
    for split in SPLITS + ["all"]:
        rs = records if split == "all" else by_split.get(split, [])
        if not rs:
            continue
        answerable = [r for r in rs if r["split"] != "unanswerable"]
        summary[split] = {
            "n": len(rs),
            "recall_at_k": mean(r["recall_at_k"] for r in answerable),
            "mrr": mean(r["rr"] for r in answerable),
            "accuracy": mean(r["correct"] for r in rs),
            "faithfulness": mean(r["faithful"] for r in rs),
            "abstain_rate": mean(r["pred"] is not None and norm_answer(r["pred"]) == NOT_FOUND for r in rs),
        }
    return summary


def token_totals(records: list[dict]) -> dict:
    totals = defaultdict(lambda: [0, 0])
    for r in records:
        for role, (inp, out) in r["tokens"].items():
            totals[role][0] += inp
            totals[role][1] += out
    return {role: {"input": i, "output": o} for role, (i, o) in totals.items()}


def run(cfg: dict, questions: list[dict], conn, judge: bool = True, workers: int = 8) -> dict:
    from lacuna.deepread.embed import embed_queries
    from lacuna.deepread.retrieve import search
    from lacuna.deepread.store import stored_index_hash

    expected, actual = index_hash(cfg), stored_index_hash(conn)
    if actual != expected:
        raise SystemExit(
            f"index was built with config {actual}, current config is {expected}; run `lacuna deepread index`"
        )

    depth = max(cfg["retrieval"]["eval_depth"], cfg["retrieval"]["top_k"])
    vectors = embed_queries([q["question"] for q in questions], cfg["embedding"])
    retrieved = [search(conn, v, depth) for v in vectors]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(
            pool.map(lambda args: evaluate_one(*args, cfg, judge), zip(questions, retrieved, strict=True))
        )

    return {
        "date": date.today().isoformat(),
        "run_hash": run_hash(cfg),
        "index_hash": expected,
        "config": {k: v for k, v in cfg.items() if k != "db"},
        "prompt_version": gen.PROMPT_VERSION,
        "generator_models_served": sorted({r["model"] for r in records}),
        "judged": judge,
        "summary": summarise(records),
        "tokens": token_totals(records),
        "records": records,
    }


# ---------------------------------------------------------------- reporting


def fmt(x) -> str:
    return "-" if x is None else f"{x:.2f}"


def render_markdown(result: dict) -> str:
    cfg = result["config"]
    lines = [
        f"# Eval {result['date']} `{result['run_hash']}`",
        "",
        f"Retriever `{cfg['embedding']['model']}`, "
        f"chunks {cfg['chunking']['tokens']}/{cfg['chunking']['overlap']}, "
        f"top-{cfg['retrieval']['top_k']}. Generator `{cfg['generator']['model']}`, "
        f"judge `{cfg['judge']['model']}`, "
        f"prompt {result['prompt_version']}. Index `{result['index_hash']}`.",
        "",
        f"| split | n | recall@{cfg['retrieval']['top_k']} | MRR@{cfg['retrieval']['eval_depth']} "
        "| accuracy | faithfulness | abstain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, m in result["summary"].items():
        lines.append(
            f"| {split} | {m['n']} | {fmt(m['recall_at_k'])} | {fmt(m['mrr'])} | {fmt(m['accuracy'])} "
            f"| {fmt(m['faithfulness'])} | {fmt(m['abstain_rate'])} |"
        )
    lines += [
        "",
        "Tokens: "
        + ", ".join(f"{k} {v['input']:,} in / {v['output']:,} out" for k, v in result["tokens"].items()),
    ]
    wrong = [r for r in result["records"] if r["correct"] is False]
    if wrong:
        lines += [
            "",
            "## Wrong answers",
            "",
            "| id | split | question | gold | predicted |",
            "|---|---|---|---|---|",
        ]
        for r in sorted(wrong, key=lambda r: (SPLITS.index(r["split"]), r["id"])):
            q = r["question"].replace("|", "\\|")
            lines.append(f"| {r['id']} | {r['split']} | {q} | {r['gold']} | {r['pred']} |")
    return "\n".join(lines) + "\n"


def write_results(result: dict) -> tuple:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = RESULTS_DIR / f"{result['date']}_{result['run_hash']}"
    json_path, md_path = stem.with_suffix(".json"), stem.with_suffix(".md")
    json_path.write_text(json.dumps(result, indent=2) + "\n")
    md_path.write_text(render_markdown(result))
    return json_path, md_path


# ---------------------------------------------------------------- judge spot-check


def write_spotcheck(result_path, chunks_by_id: dict, n: int = 15, seed: int = 0):
    """Sample n faithfulness judgements into a CSV; fill in human_supported (yes/no) by hand."""
    result = json.loads(open(result_path).read())
    judged = [r for r in result["records"] if r["faithful"] is not None]
    sample = random.Random(seed).sample(judged, min(n, len(judged)))
    k = result["config"]["retrieval"]["top_k"]
    out = result_path.with_name(result_path.stem + "_spotcheck.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["id", "question", "answer", "judge_supported", "judge_reasoning", "human_supported", "context"]
        )
        for r in sample:
            context = gen.format_context([chunks_by_id[c["id"]] for c in r["retrieved"][:k]])
            w.writerow(
                [r["id"], r["question"], r["pred"], r["faithful"], r["faithfulness_judge"], "", context]
            )
    return out


def spotcheck_agreement(csv_path) -> dict:
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["human_supported"].strip()]
    agree = sum(
        (r["human_supported"].strip().lower() in {"yes", "y", "true", "1"})
        == (r["judge_supported"] == "True")
        for r in rows
    )
    return {"labelled": len(rows), "agree": agree, "agreement": round(agree / len(rows), 3) if rows else None}
