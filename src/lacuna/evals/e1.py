"""E1 — survey recall.

Ground truth: the reference list of a recent CV survey. Query: a user-style question about the
survey's topic. A method is scored by the fraction of the survey's references it retrieves in
its top k (recall@50/100/200). Two $0 baselines in Phase 1: keyword search, and keyword search
plus depth-1 citation snowballing. Leakage rules: the survey itself is excluded, and nothing
published after the survey's date may be retrieved (`to_date` freeze).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from lacuna.scout.core import ScoutConfig, scout
from lacuna.settings import ROOT
from lacuna.util import stable_hash

DATASET_DIR = ROOT / "evals" / "datasets" / "e1"
RESULTS_DIR = ROOT / "evals" / "results"
KS = (50, 100, 200)


@dataclass
class Survey:
    openalex_id: str
    title: str
    query: str
    freeze: date
    gold: set[str]
    n_unreachable: int = 0  # references published after the freeze, dropped from gold


def after(ref: dict, freeze: date) -> bool:
    d = ref.get("publication_date")
    return bool(d) and date.fromisoformat(d) > freeze


def load_dataset(dataset_dir: Path = DATASET_DIR) -> list[Survey]:
    surveys = json.loads((dataset_dir / "surveys.json").read_text())
    freeze = json.loads((dataset_dir / "freeze.json").read_text())
    out = []
    for s in surveys:
        refs = json.loads((dataset_dir / "refs" / f"{s['openalex_id']}.json").read_text())
        frozen = date.fromisoformat(freeze[s["openalex_id"]])
        # The freeze is the earlier of the arXiv and journal dates, but OpenAlex stores the journal
        # version's reference list, which can cite work from after the freeze. No method may
        # retrieve those, so they are unreachable by construction and leave the gold set.
        reachable = {r["id"] for r in refs if not after(r, frozen)}
        out.append(
            Survey(s["openalex_id"], s["title"], s["query"], frozen, reachable, len(refs) - len(reachable))
        )
    return out


def recall_at(ranked_ids: list[str], gold: set[str], ks=KS) -> dict[str, float]:
    out = {}
    for k in ks:
        hits = len(set(ranked_ids[:k]) & gold)
        out[f"recall@{k}"] = round(hits / len(gold), 4) if gold else 0.0
    out["retrieved"] = len(ranked_ids)
    out["gold"] = len(gold)
    return out


def run_e1(openalex, *, surveys: list[Survey] | None = None, seeds: int = 20, ks=KS) -> dict:
    surveys = surveys or load_dataset()
    k_max = max(ks)
    per_survey = []
    for s in surveys:
        exclude = {s.openalex_id}
        keyword = [
            r.paper_id
            for r in openalex.search_works(s.query, to_date=s.freeze, max_results=k_max)
            if r.paper_id not in exclude
        ]
        snow = scout(
            s.query, openalex, ScoutConfig(max_papers=k_max, seeds=seeds, to_date=s.freeze), exclude=exclude
        )
        per_survey.append(
            {
                "openalex_id": s.openalex_id,
                "title": s.title,
                "query": s.query,
                "freeze": s.freeze.isoformat(),
                "n_unreachable": s.n_unreachable,
                "keyword": recall_at(keyword, s.gold, ks),
                "snowball": {
                    **recall_at(snow.ids(), s.gold, ks),
                    "status": snow.status,
                    "n_seeds": snow.n_seeds,
                },
            }
        )

    def macro(method: str) -> dict:
        return {
            f"recall@{k}": round(sum(p[method][f"recall@{k}"] for p in per_survey) / len(per_survey), 4)
            for k in ks
        }

    config = {"seeds": seeds, "ks": list(ks), "surveys": sorted(s.openalex_id for s in surveys)}
    gold_fp = stable_hash(*(sorted(s.gold) for s in surveys))
    return {
        "eval": "e1",
        "date": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "config_hash": stable_hash(config, gold_fp),
        "summary": {"keyword": macro("keyword"), "snowball": macro("snowball")},
        "per_survey": per_survey,
    }


def render_markdown(result: dict) -> str:
    ks = result["config"]["ks"]
    lines = [f"# E1 survey recall — {result['date']} `{result['config_hash']}`", ""]
    lines += [
        "Recall of each survey's reference list in the top k, with the survey excluded and nothing "
        "published after the survey's date. References published after that date are unreachable "
        "and are left out of the gold set (column `dropped`). Macro average over surveys.",
        "",
        "| method | " + " | ".join(f"recall@{k}" for k in ks) + " |",
        "|---|" + "---|" * len(ks),
    ]
    for m, row in result["summary"].items():
        lines.append(f"| {m} | " + " | ".join(f"{row[f'recall@{k}']:.3f}" for k in ks) + " |")
    lines += [
        "",
        "## Per survey",
        "",
        "| survey | freeze | gold | dropped | " + " | ".join(f"kw@{k} | sb@{k}" for k in ks) + " | status |",
        "|---|---|---|---|" + "---|---|" * len(ks) + "---|",
    ]
    for p in result["per_survey"]:
        cells = " | ".join(
            f"{p['keyword'][f'recall@{k}']:.3f} | {p['snowball'][f'recall@{k}']:.3f}" for k in ks
        )
        status = p["snowball"]["status"]
        gold, dropped = p["keyword"]["gold"], p.get("n_unreachable", 0)
        lines.append(f"| {p['title'][:60]} | {p['freeze']} | {gold} | {dropped} | {cells} | {status} |")
    return "\n".join(lines) + "\n"


def write_results(result: dict, results_dir: Path = RESULTS_DIR) -> list[Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    stem = results_dir / f"e1_{result['config_hash']}"
    stem.with_suffix(".json").write_text(json.dumps(result, indent=2))
    stem.with_suffix(".md").write_text(render_markdown(result))
    return [stem.with_suffix(".json"), stem.with_suffix(".md")]
