"""Per-provider extraction throughput and schema validity (E3's pre-label numbers).

Each spec runs as a single-element chain with every cache disabled, so the numbers are real calls
(this spends free-tier quota: run it deliberately). Nothing is written to the extraction cache or
the graph.
"""

from __future__ import annotations

import json
import statistics
import time
from datetime import date

from lacuna.extract.llm import NoChunksError, extract_paper
from lacuna.llm import LLMError, SpendStageError
from lacuna.settings import ROOT

RESULTS_DIR = ROOT / "evals" / "results"


LEDGER = ("calls", "errors", "rate_limited", "schema_failures")


def ledger_totals(conn, spec: str) -> dict:
    """All-time provider_usage totals for one spec; a before/after delta isolates one run."""
    provider, model = spec.split("/", 1)
    row = conn.execute(
        "SELECT "
        + ", ".join(f"coalesce(sum({c}), 0)::int AS {c}" for c in LEDGER)
        + " FROM provider_usage WHERE provider = %s AND model = %s",
        (provider, model),
    ).fetchone()
    return {c: int((row or {}).get(c) or 0) for c in LEDGER}


def failure_kind(msg: str) -> str:
    for needle, kind in (
        ("exceeds_tpm", "skipped: over TPM"),
        ("rate_limited", "rate limited"),
        ("quota", "quota"),
        ("max completion tokens", "output cap reached"),
        ("503", "provider 503"),
        ("ValidationError", "invalid JSON/schema"),
        ("no_key", "no key"),
    ):
        if needle in msg:
            return kind
    return msg[:60]


def measure(conn, paper_ids: list[str], specs: list[str], *, write: bool = True) -> dict:
    """Batch success is end to end (any failure counts). Schema validity is the ledger's rate over
    responses actually received, so skipped or errored calls never count against a model's JSON."""
    per_spec = {}
    for spec in specs:
        before = ledger_totals(conn, spec)
        row = {
            "spec": spec,
            "papers": 0,
            "calls": 0,
            "failed_calls": 0,
            "rejected_citations": 0,
            "items": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "errors": [],
            "failure_kinds": {},
        }
        latencies: list[float] = []
        t_start = time.monotonic()
        for pid in paper_ids:
            try:
                res = extract_paper(conn, pid, chain=[spec], use_cache=False, strict=False)
            except SpendStageError:
                raise
            except (LLMError, NoChunksError) as e:
                row["errors"].append({"paper_id": pid, "error": str(e)[:200]})
                continue
            row["papers"] += 1
            row["calls"] += res.calls
            row["failed_calls"] += res.failed_calls
            for f in res.failures:
                k = failure_kind(f)
                row["failure_kinds"][k] = row["failure_kinds"].get(k, 0) + 1
            row["rejected_citations"] += res.rejected_citations
            row["items"] += sum(res.extraction.counts().values())
            row["input_tokens"] += res.tokens["input"]
            row["output_tokens"] += res.tokens["output"]
            latencies += res.latencies_ms
        row["wall_s"] = round(time.monotonic() - t_start, 1)
        after = ledger_totals(conn, spec)
        delta = {c: after[c] - before[c] for c in LEDGER}
        responses = delta["calls"] - delta["errors"] - delta["rate_limited"]
        row["network_calls"] = delta["calls"]
        row["provider_errors"] = delta["errors"]
        row["rate_limited"] = delta["rate_limited"]
        row["schema_failures"] = delta["schema_failures"]
        row["batch_success_rate"] = (
            round((row["calls"] - row["failed_calls"]) / row["calls"], 3) if row["calls"] else None
        )
        row["schema_valid_rate"] = (
            round((responses - delta["schema_failures"]) / responses, 3) if responses > 0 else None
        )
        row["mean_latency_ms"] = round(statistics.fmean(latencies)) if latencies else None
        per_spec[spec] = row
    report = {"date": date.today().isoformat(), "papers": list(paper_ids), "specs": per_spec}
    if write:
        report["files"] = write_report(report)
    return report


def to_markdown(report: dict) -> str:
    lines = [
        f"# Extraction throughput, {report['date']}",
        "",
        f"{len(report['papers'])} papers, caches disabled, one provider per run.",
        "",
        "Batch success counts every failure, including calls skipped before the network. Schema validity is",
        "over responses actually received (ledger delta), so it measures the model's JSON, not the plumbing.",
        "",
        "| spec | papers ok | batches | batch success | network calls | schema-valid | items | mean ms "
        "| in tok | out tok | failures |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in report["specs"].values():
        lines.append(
            f"| {s['spec']} | {s['papers']} | {s['calls']} | {s['batch_success_rate']} | "
            f"{s['network_calls']} | {s['schema_valid_rate']} | {s['items']} | {s['mean_latency_ms']} | "
            f"{s['input_tokens']} | {s['output_tokens']} | "
            + (", ".join(f"{k} ×{n}" for k, n in s["failure_kinds"].items()) or "-")
            + " |"
        )
    errs = [(s["spec"], e) for s in report["specs"].values() for e in s["errors"]]
    if errs:
        lines += ["", "Errors:", *[f"- {spec}: {e['paper_id']}: {e['error']}" for spec, e in errs]]
    return "\n".join(lines) + "\n"


def write_report(report: dict) -> list[str]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base = RESULTS_DIR / f"extract_throughput_{report['date']}"
    base.with_suffix(".json").write_text(json.dumps(report, indent=2))
    base.with_suffix(".md").write_text(to_markdown(report))
    return [str(base.with_suffix(".json")), str(base.with_suffix(".md"))]
