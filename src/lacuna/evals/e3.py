"""E3 — extraction quality.

Hand labels (set-level, per entity type) are written by the user; see evals/datasets/e3/README.md.
Predictions come from `lacuna.extract.llm.extract_paper`. Scoring is exact match after normalisation,
precision/recall/F1 per entity type, micro and macro averages over types. Labels are split into a
tuning half and a reporting half by `split_for`; only the reporting half is quoted as a result.
The per-provider schema-validity rate comes from `provider_usage` (1 - schema_failures / calls).
No labels, no scores: with no labels the harness reports label counts and validity only.
"""

from __future__ import annotations

import hashlib
import json
import re
import string
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from lacuna.settings import ROOT
from lacuna.util import stable_hash

DATASET_DIR = ROOT / "evals" / "datasets" / "e3"
RESULTS_DIR = ROOT / "evals" / "results"
ENTITY_TYPES = ("methods", "datasets", "tasks", "metrics")
SPLITS = ("tuning", "reporting")

Normalise = Callable[..., str]  # (name, entity_type=None) -> comparable key

# Variant suffixes: "ViT-B/16" -> "vit", "ResNet-50" -> "resnet", "ImageNet-1k" -> "imagenet".
_VARIANT = re.compile(r"(?:[\s_-]*(?:[a-z]/\d+|tiny|small|base|large|huge|xl|v?\d+(?:\.\d+)?[kmb]?))+$")
_PUNCT = str.maketrans("", "", string.punctuation + string.whitespace)


def default_normalise(name: str, entity_type: str | None = None) -> str:
    s = name.lower().strip()
    stripped = _VARIANT.sub("", s)
    s = stripped if stripped.strip(" -_") else s  # never reduce a name to nothing ("3D" stays)
    return s.translate(_PUNCT)


def get_normalise() -> Normalise:
    """Canonicalise like the extractor does, minus the database: methods lose their variant
    ("ViT-B/16" -> "ViT"), then the seed alias table maps synonyms to one name ("ILSVRC-2012" ->
    "ImageNet-1k"). Deterministic on purpose, so a score never depends on what `entities` holds.
    Falls back to the local default when lacuna.extract is unavailable."""
    try:
        from lacuna.extract.resolve import load_aliases, normalise, split_variant

        aliases = load_aliases()
    except Exception:  # noqa: BLE001 - not built, or config missing
        return default_normalise

    def canonical(name: str, entity_type: str | None = None) -> str:
        t = (entity_type or "").removesuffix("s")  # "methods" -> "method"
        base = split_variant(name)[0] if t == "method" else name
        for candidate in (base, name):
            canon = aliases.get((t, normalise(candidate)))
            if canon:
                return normalise(canon)
        return normalise(base)

    return canonical


# ------------------------------------------------------------------ scoring


def _prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
    }


def _norm_set(names, normalise: Normalise, entity_type: str | None = None) -> set[str]:
    return {n for n in (normalise(x, entity_type) for x in (names or ())) if n}


def _counts(gold: dict, pred: dict, normalise: Normalise, types=ENTITY_TYPES) -> dict[str, list[int]]:
    out = {}
    for t in types:
        g, p = _norm_set(gold.get(t), normalise, t), _norm_set(pred.get(t), normalise, t)
        out[t] = [len(g & p), len(p - g), len(g - p)]
    return out


def _summarise(counts: dict[str, list[int]]) -> dict:
    per_type = {t: _prf(*c) for t, c in counts.items()}
    micro = _prf(*(sum(c[i] for c in counts.values()) for i in range(3)))
    # Macro averages skip types that appear in neither gold nor prediction.
    live = [v for v in per_type.values() if v["tp"] + v["fp"] + v["fn"]]
    macro = {
        k: round(sum(v[k] for v in live) / len(live), 4) if live else 0.0
        for k in ("precision", "recall", "f1")
    }
    return {**per_type, "micro": micro, "macro": macro}


def score(
    gold: dict[str, set[str]],
    pred: dict[str, set[str]],
    normalise: Normalise = default_normalise,
) -> dict:
    """Per-type {tp, fp, fn, precision, recall, f1} plus "micro" and "macro" entries."""
    return _summarise(_counts(gold, pred, normalise))


def score_papers(pairs: list[tuple[dict, dict]], normalise: Normalise = default_normalise) -> dict:
    """Same shape as `score`, with counts summed over papers (so a paper's set is scored on its own)."""
    total = {t: [0, 0, 0] for t in ENTITY_TYPES}
    for gold, pred in pairs:
        for t, c in _counts(gold, pred, normalise).items():
            total[t] = [a + b for a, b in zip(total[t], c, strict=True)]
    return _summarise(total)


# ------------------------------------------------------------------ labels and split


def _hash_key(paper_id: str) -> str:
    return hashlib.sha256(paper_id.encode()).hexdigest()


def split_for(paper_id: str) -> str:
    """Stateless fallback: parity of the paper-id hash."""
    return SPLITS[int(_hash_key(paper_id), 16) % 2]


def assign_splits(paper_ids: list[str], existing: dict[str, str] | None = None) -> dict[str, str]:
    """Deterministic, balanced assignment. Existing assignments are never changed.

    New ids are taken in hash order and go to the smaller half (ties by hash parity), so any
    batch ends within one of half-and-half, and re-running gives the same answer.
    """
    out = dict(existing or {})
    counts = {s: sum(1 for v in out.values() if v == s) for s in SPLITS}
    for pid in sorted((p for p in paper_ids if p not in out), key=_hash_key):
        if counts["tuning"] == counts["reporting"]:
            s = split_for(pid)
        else:
            s = min(SPLITS, key=counts.get)
        out[pid] = s
        counts[s] += 1
    return out


def load_split(dataset_dir: Path = DATASET_DIR) -> dict[str, str]:
    path = dataset_dir / "split.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_labels(dataset_dir: Path = DATASET_DIR) -> list[dict]:
    """Completed labels: labels/<paper_id>.json with a labeller and at least one entity."""
    split = load_split(dataset_dir)
    out = []
    for path in sorted((dataset_dir / "labels").glob("*.json")):
        d = json.loads(path.read_text())
        if not d.get("labeller") or not any(d.get(t) for t in ENTITY_TYPES):
            continue  # an unfinished template copied into labels/
        d["split"] = split.get(d["paper_id"]) or d.get("split") or split_for(d["paper_id"])
        d.setdefault("scope", "abstract")
        out.append(d)
    return out


def write_template(
    conn, topic_id: str, n: int = 30, dataset_dir: Path = DATASET_DIR
) -> tuple[list[Path], dict[str, str]]:
    """todo/<paper_id>.json for the top-n topic papers that have an abstract. Entity lists stay empty."""
    rows = conn.execute(
        "SELECT p.paper_id, p.title, p.abstract FROM topic_papers tp JOIN papers p USING (paper_id) "
        "WHERE tp.topic_id = %s AND coalesce(btrim(p.abstract), '') <> '' "
        "ORDER BY tp.rank NULLS LAST, tp.score DESC NULLS LAST, p.paper_id LIMIT %s",
        (topic_id, n),
    ).fetchall()
    todo, labels = dataset_dir / "todo", dataset_dir / "labels"
    todo.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    split = assign_splits([r["paper_id"] for r in rows], load_split(dataset_dir))
    (dataset_dir / "split.json").write_text(json.dumps(dict(sorted(split.items())), indent=2) + "\n")
    written = []
    for r in rows:
        pid = r["paper_id"]
        if (todo / f"{pid}.json").exists() or (labels / f"{pid}.json").exists():
            continue  # never overwrite a labeller's work in progress
        doc = {
            "paper_id": pid,
            "title": r["title"],
            "labeller": "",
            "labelled_on": "",
            "scope": "abstract",
            "split": split[pid],
            "abstract": r["abstract"],
            "methods": [],
            "datasets": [],
            "tasks": [],
            "metrics": [],
            "notes": "",
        }
        (todo / f"{pid}.json").write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
        written.append(todo / f"{pid}.json")
    return written, split


# ------------------------------------------------------------------ schema validity


def schema_validity(conn, *, today_only: bool = False) -> list[dict]:
    """Per provider/model: calls, schema_failures and validity = 1 - schema_failures / calls."""
    from lacuna.db import quota_day

    where = "WHERE day >= %s " if today_only else ""
    rows = conn.execute(
        "SELECT provider, model, sum(calls)::int AS calls, sum(schema_failures)::int AS schema_failures "
        f"FROM provider_usage {where}GROUP BY provider, model HAVING sum(calls) > 0 ORDER BY provider, model",
        (quota_day("gemini"),) if today_only else (),  # the earliest current quota day
    ).fetchall()
    return [{**dict(r), "validity": round(1 - r["schema_failures"] / r["calls"], 4)} for r in rows]


def label_counts(labels: list[dict]) -> dict:
    return {s: sum(1 for x in labels if x["split"] == s) for s in SPLITS} | {"total": len(labels)}


# ------------------------------------------------------------------ run


def _predicted(extraction) -> dict[str, set[str]]:
    return {t: {e.name for e in getattr(extraction, t, [])} for t in ENTITY_TYPES}


def run_e3(
    conn,
    *,
    split: str = "reporting",
    chain=None,
    scope: str = "abstract",
    dataset_dir: Path = DATASET_DIR,
    extract=None,
    normalise: Normalise | None = None,
) -> dict:
    """Score extraction on the labels in `split` ("tuning", "reporting" or "all") and `scope`.

    A paper whose extraction raises is scored with empty predictions and listed in `failed`.
    `scope` selects which labels are used; what extract_paper reads is decided by the chunks
    available for the paper (abstract chunk only, or full text once approved).
    """
    labels = load_labels(dataset_dir)
    counts = label_counts(labels)
    selected = [x for x in labels if (split == "all" or x["split"] == split) and x["scope"] == scope]
    validity = schema_validity(conn, today_only=True)
    result = {
        "eval": "e3",
        "date": datetime.now().isoformat(timespec="seconds"),
        "split": split,
        "scope": scope,
        "label_counts": counts,
        "n_scored": len(selected),
        "schema_validity_today": validity,
    }
    config = {
        "split": split,
        "scope": scope,
        "chain": list(chain) if isinstance(chain, (list, tuple)) else (None if chain is None else str(chain)),
        "papers": sorted(x["paper_id"] for x in selected),
    }
    gold_fp = stable_hash(*([{t: sorted(x.get(t, [])) for t in ENTITY_TYPES} for x in selected] or [""]))
    result |= {"config": config, "config_hash": stable_hash(config, gold_fp)}
    if not selected:
        result["scores"] = None
        return result

    if extract is None:
        from lacuna.extract.llm import extract_paper as extract  # lazy: built by another package
    norm = normalise or get_normalise()
    pairs, per_paper, failed = [], [], []
    for lab in selected:
        pid = lab["paper_id"]
        gold = {t: set(lab.get(t, [])) for t in ENTITY_TYPES}
        try:
            pred = _predicted(extract(conn, pid, chain=chain).extraction)
        except Exception as e:  # noqa: BLE001 - any failure counts as an empty extraction
            pred = {t: set() for t in ENTITY_TYPES}
            failed.append({"paper_id": pid, "error": f"{type(e).__name__}: {e}"[:200]})
        pairs.append((gold, pred))
        per_paper.append({"paper_id": pid, "title": lab.get("title", ""), **score(gold, pred, norm)["micro"]})
    result |= {"scores": score_papers(pairs, norm), "failed": failed, "per_paper": per_paper}
    return result


def render_markdown(result: dict) -> str:
    c = result["label_counts"]
    lines = [
        f"# E3 extraction quality — {result['date']} `{result['config_hash']}`",
        "",
        f"Labels: {c['total']} ({c['tuning']} tuning, {c['reporting']} reporting). "
        f"Scored: {result['n_scored']} (split `{result['split']}`, scope `{result['scope']}`).",
        "",
    ]
    if result.get("scores"):
        lines += ["| entity type | tp | fp | fn | precision | recall | F1 |", "|---|---|---|---|---|---|---|"]
        for k, v in result["scores"].items():
            if k == "macro":
                lines.append(f"| macro | | | | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} |")
            else:
                lines.append(
                    f"| {k} | {v['tp']} | {v['fp']} | {v['fn']} | {v['precision']:.3f} | "
                    f"{v['recall']:.3f} | {v['f1']:.3f} |"
                )
        if result.get("failed"):
            lines += ["", f"{len(result['failed'])} paper(s) failed extraction and were scored as empty."]
    else:
        lines += ["No labels for this split and scope, so no scores. Nothing is estimated.", ""]
    lines += ["", "## Schema validity today", "", validity_table(result["schema_validity_today"])]
    return "\n".join(lines) + "\n"


def validity_table(rows: list[dict]) -> str:
    if not rows:
        return "No provider calls recorded."
    out = ["| provider | model | calls | schema failures | validity |", "|---|---|---|---|---|"]
    out += [
        f"| {r['provider']} | {r['model']} | {r['calls']} | {r['schema_failures']} | {r['validity']:.3f} |"
        for r in rows
    ]
    return "\n".join(out)


def write_results(result: dict, results_dir: Path = RESULTS_DIR) -> list[Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = results_dir / f"e3_{result['config_hash']}"
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2))
    stem.with_suffix(".md").write_text(render_markdown(result))
    return [stem.with_suffix(".json"), stem.with_suffix(".md")]
