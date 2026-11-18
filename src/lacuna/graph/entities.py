"""Graph writes and reads for extracted entities (blueprint 7.1).

Every statement is a parameterised constant; labels cannot be parameters, so there is one constant
per label. `MERGE`, never `CREATE`. Writes `MATCH` the `Paper` first: no Paper node, no write.

Edges from a paper (all carry chunk_id, confidence, extracted_by):
  PROPOSES      Paper -> Method     role "proposed"; `variant` on the edge
  USES_METHOD   Paper -> Method     role "used"/"compared" (not in the blueprint; see decisions.md)
  EXTENDS       Method -> Method    paper_id
  EVALUATES_ON  Paper -> Dataset    method ("" when not tied to one)
  ADDRESSES     Paper -> Task
  REPORTS       Paper -> Metric     method, variant, dataset, setting, value
  STATES_LIMITATION Paper -> Limitation   chunk_id

`replace=True` (default) first deletes this paper's previous extraction edges and Limitation nodes
in the same transaction, so the graph holds the latest extraction of a paper, never a mix.

Resolved input (built by lacuna.extract.pipeline.resolve_extraction):
  {"methods":[{entity_id, canonical_name, variant, role, extends_id, extends_name, chunk_id, confidence}],
   "datasets":[{entity_id, canonical_name, chunk_id, confidence}], "tasks":[...same...],
   "metrics":[{entity_id, canonical_name, higher_is_better, chunk_id, confidence}],
   "results":[{method_id, method, variant, dataset_id, dataset, metric_id, metric, value, setting,
               chunk_id, confidence}],
   "limitations":[{text, chunk_id}]}
"""

from __future__ import annotations

import hashlib

from neo4j import RoutingControl

ENSURE_ENTITY_CONSTRAINTS = tuple(
    f"CREATE CONSTRAINT {label.lower()}_entity_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.entity_id IS UNIQUE"
    for label in ("Method", "Dataset", "Task", "Metric")
) + ("CREATE CONSTRAINT limitation_id IF NOT EXISTS FOR (n:Limitation) REQUIRE n.limitation_id IS UNIQUE",)

PAPER_EXISTS = "MATCH (p:Paper {paper_id: $paper_id}) RETURN count(p) AS n"

# Remove the previous extraction of this paper. Entity nodes are shared and stay.
CLEAR_PAPER_EXTRACTION = """
MATCH (p:Paper {paper_id: $paper_id})
OPTIONAL MATCH (p)-[r:PROPOSES|USES_METHOD|EVALUATES_ON|ADDRESSES|REPORTS]->()
DELETE r
WITH DISTINCT p
OPTIONAL MATCH ()-[e:EXTENDS {paper_id: $paper_id}]->()
DELETE e
WITH DISTINCT p
OPTIONAL MATCH (p)-[:STATES_LIMITATION]->(l:Limitation)
DETACH DELETE l
"""

# rows: [{entity_id, canonical_name, variant, chunk_id, confidence}]
_METHOD_EDGE = """
MATCH (p:Paper {{paper_id: $paper_id}})
UNWIND $rows AS row
MERGE (n:Method {{entity_id: row.entity_id}})
SET n.canonical_name = row.canonical_name
SET n += $props
MERGE (p)-[r:{rel}]->(n)
SET r.variant = coalesce(row.variant, ''), r.role = row.role, r.chunk_id = row.chunk_id,
    r.confidence = row.confidence, r.extracted_by = $extracted_by
RETURN count(r) AS n
"""
UPSERT_PROPOSES = _METHOD_EDGE.format(rel="PROPOSES")
UPSERT_USES_METHOD = _METHOD_EDGE.format(rel="USES_METHOD")

# rows: [{src_id, src_name, dst_id, dst_name, chunk_id, confidence}]
UPSERT_EXTENDS = """
MATCH (p:Paper {paper_id: $paper_id})
UNWIND $rows AS row
MERGE (a:Method {entity_id: row.src_id})
SET a.canonical_name = row.src_name
SET a += $props
MERGE (b:Method {entity_id: row.dst_id})
SET b.canonical_name = row.dst_name
SET b += $props
MERGE (a)-[r:EXTENDS {paper_id: $paper_id}]->(b)
SET r.chunk_id = row.chunk_id, r.confidence = row.confidence, r.extracted_by = $extracted_by
RETURN count(r) AS n
"""

# rows: [{entity_id, canonical_name, method, chunk_id, confidence}]
UPSERT_EVALUATES_ON = """
MATCH (p:Paper {paper_id: $paper_id})
UNWIND $rows AS row
MERGE (n:Dataset {entity_id: row.entity_id})
SET n.canonical_name = row.canonical_name
SET n += $props
MERGE (p)-[r:EVALUATES_ON {method: row.method}]->(n)
SET r.chunk_id = row.chunk_id, r.confidence = row.confidence, r.extracted_by = $extracted_by
RETURN count(r) AS n
"""

UPSERT_ADDRESSES = """
MATCH (p:Paper {paper_id: $paper_id})
UNWIND $rows AS row
MERGE (n:Task {entity_id: row.entity_id})
SET n.canonical_name = row.canonical_name
SET n += $props
MERGE (p)-[r:ADDRESSES]->(n)
SET r.chunk_id = row.chunk_id, r.confidence = row.confidence, r.extracted_by = $extracted_by
RETURN count(r) AS n
"""

# rows: [{entity_id, canonical_name, higher_is_better, method, variant, dataset, setting, value,
#         chunk_id, confidence}]; method/variant/dataset/setting are "" when absent. value may be null.
UPSERT_REPORTS = """
MATCH (p:Paper {paper_id: $paper_id})
UNWIND $rows AS row
MERGE (n:Metric {entity_id: row.entity_id})
SET n.canonical_name = row.canonical_name
SET n.higher_is_better = coalesce(row.higher_is_better, n.higher_is_better)
SET n += $props
MERGE (p)-[r:REPORTS {method: row.method, variant: row.variant, dataset: row.dataset,
                      setting: row.setting}]->(n)
SET r.value = row.value, r.chunk_id = row.chunk_id, r.confidence = row.confidence,
    r.extracted_by = $extracted_by
RETURN count(r) AS n
"""

# rows: [{limitation_id, text, chunk_id}]
UPSERT_LIMITATIONS = """
MATCH (p:Paper {paper_id: $paper_id})
UNWIND $rows AS row
MERGE (l:Limitation {limitation_id: row.limitation_id})
SET l.text = row.text, l.paper_id = $paper_id, l.chunk_id = row.chunk_id
SET l += $props
MERGE (p)-[r:STATES_LIMITATION]->(l)
SET r.chunk_id = row.chunk_id, r.extracted_by = $extracted_by
RETURN count(r) AS n
"""

PAPER_METHODS = """
MATCH (p:Paper {paper_id: $paper_id})-[r:PROPOSES|USES_METHOD]->(m:Method)
RETURN m.entity_id AS entity_id, m.canonical_name AS name, r.variant AS variant, r.role AS role,
       r.chunk_id AS chunk_id, r.confidence AS confidence, r.extracted_by AS extracted_by
ORDER BY r.role, m.canonical_name
"""
PAPER_DATASETS = """
MATCH (p:Paper {paper_id: $paper_id})-[r:EVALUATES_ON]->(d:Dataset)
RETURN d.entity_id AS entity_id, d.canonical_name AS name, collect(DISTINCT r.method) AS methods,
       max(r.confidence) AS confidence
ORDER BY d.canonical_name
"""
PAPER_TASKS = """
MATCH (p:Paper {paper_id: $paper_id})-[r:ADDRESSES]->(t:Task)
RETURN t.entity_id AS entity_id, t.canonical_name AS name, r.confidence AS confidence
ORDER BY t.canonical_name
"""
PAPER_RESULTS = """
MATCH (p:Paper {paper_id: $paper_id})-[r:REPORTS]->(m:Metric)
RETURN m.entity_id AS entity_id, m.canonical_name AS metric, m.higher_is_better AS higher_is_better,
       r.method AS method, r.variant AS variant, r.dataset AS dataset, r.setting AS setting,
       r.value AS value, r.chunk_id AS chunk_id
ORDER BY m.canonical_name, r.dataset, r.method
"""
PAPER_LIMITATIONS = """
MATCH (p:Paper {paper_id: $paper_id})-[:STATES_LIMITATION]->(l:Limitation)
RETURN l.text AS text, l.chunk_id AS chunk_id
ORDER BY l.chunk_id, l.limitation_id
"""

# Most frequent entities among the papers a topic INCLUDES, by distinct paper count.
TOPIC_METHODS = """
MATCH (:Topic {topic_id: $topic_id})-[:INCLUDES]->(p:Paper)-[:PROPOSES|USES_METHOD]->(n:Method)
RETURN n.entity_id AS entity_id, n.canonical_name AS name, count(DISTINCT p) AS papers
ORDER BY papers DESC, name LIMIT $top
"""
TOPIC_DATASETS = """
MATCH (:Topic {topic_id: $topic_id})-[:INCLUDES]->(p:Paper)-[:EVALUATES_ON]->(n:Dataset)
RETURN n.entity_id AS entity_id, n.canonical_name AS name, count(DISTINCT p) AS papers
ORDER BY papers DESC, name LIMIT $top
"""
TOPIC_TASKS = """
MATCH (:Topic {topic_id: $topic_id})-[:INCLUDES]->(p:Paper)-[:ADDRESSES]->(n:Task)
RETURN n.entity_id AS entity_id, n.canonical_name AS name, count(DISTINCT p) AS papers
ORDER BY papers DESC, name LIMIT $top
"""
TOPIC_METRICS = """
MATCH (:Topic {topic_id: $topic_id})-[:INCLUDES]->(p:Paper)-[:REPORTS]->(n:Metric)
RETURN n.entity_id AS entity_id, n.canonical_name AS name, count(DISTINCT p) AS papers
ORDER BY papers DESC, name LIMIT $top
"""

WRITE_STATEMENTS = (
    UPSERT_PROPOSES,
    UPSERT_USES_METHOD,
    UPSERT_EXTENDS,
    UPSERT_EVALUATES_ON,
    UPSERT_ADDRESSES,
    UPSERT_REPORTS,
    UPSERT_LIMITATIONS,
)


# ---------------------------------------------------------------- row shaping


def limitation_id(paper_id: str, chunk_id: str, text: str) -> str:
    return hashlib.sha256(f"{paper_id}\x00{chunk_id}\x00{text}".encode()).hexdigest()[:24]


def build_rows(resolved: dict, paper_id: str = "") -> dict[str, list[dict]]:
    """Shape a resolved extraction into the row lists each statement takes. Pure; no I/O."""
    methods = resolved.get("methods", [])
    proposes = [m for m in methods if m["role"] == "proposed"]
    uses = [m for m in methods if m["role"] != "proposed"]
    extends = [
        {
            "src_id": m["entity_id"],
            "src_name": m["canonical_name"],
            "dst_id": m["extends_id"],
            "dst_name": m["extends_name"],
            "chunk_id": m["chunk_id"],
            "confidence": m["confidence"],
        }
        for m in methods
        if m.get("extends_id") and m["extends_id"] != m["entity_id"]
    ]
    # EVALUATES_ON: each listed dataset (method ""), plus every (method, dataset) a result ties together.
    ev: dict[tuple[str, str], dict] = {}
    for d in resolved.get("datasets", []):
        ev.setdefault(
            (d["entity_id"], ""),
            {**{k: d[k] for k in ("entity_id", "canonical_name", "chunk_id", "confidence")}, "method": ""},
        )
    for r in resolved.get("results", []):
        ev.setdefault(
            (r["dataset_id"], r["method"]),
            {
                "entity_id": r["dataset_id"],
                "canonical_name": r["dataset"],
                "method": r["method"],
                "chunk_id": r["chunk_id"],
                "confidence": r["confidence"],
            },
        )
    reported: set[str] = set()
    reports = []
    metric_meta = {m["entity_id"]: m for m in resolved.get("metrics", [])}
    for r in resolved.get("results", []):
        reported.add(r["metric_id"])
        reports.append(
            {
                "entity_id": r["metric_id"],
                "canonical_name": r["metric"],
                "higher_is_better": metric_meta.get(r["metric_id"], {}).get("higher_is_better"),
                "method": r["method"],
                "variant": r.get("variant") or "",
                "dataset": r["dataset"],
                "setting": r.get("setting") or "",
                "value": r["value"],
                "chunk_id": r["chunk_id"],
                "confidence": r["confidence"],
            }
        )
    for m in resolved.get("metrics", []):  # a metric named without any reported number
        if m["entity_id"] not in reported:
            reports.append(
                {
                    **{k: m[k] for k in ("entity_id", "canonical_name", "higher_is_better")},
                    "method": "",
                    "variant": "",
                    "dataset": "",
                    "setting": "",
                    "value": None,
                    "chunk_id": m["chunk_id"],
                    "confidence": m["confidence"],
                }
            )
    return {
        "proposes": proposes,
        "uses": uses,
        "extends": extends,
        "evaluates_on": list(ev.values()),
        "addresses": list(resolved.get("tasks", [])),
        "reports": reports,
        "limitations": [
            {
                "limitation_id": limitation_id(paper_id, lim["chunk_id"], lim["text"]),
                "text": lim["text"],
                "chunk_id": lim["chunk_id"],
            }
            for lim in resolved.get("limitations", [])
        ],
    }


# ---------------------------------------------------------------- writes


def ensure_entity_constraints(repo) -> None:
    for statement in ENSURE_ENTITY_CONSTRAINTS:
        repo.driver.execute_query(statement, routing_=RoutingControl.WRITE)


def write_extraction(
    repo,
    paper_id: str,
    resolved: dict,
    *,
    extracted_by: str,
    extra_props: dict | None = None,
    replace: bool = True,
) -> dict:
    """Write one resolved extraction in a single transaction. Returns counts; `written` is False
    (and nothing is touched) when the Paper node does not exist. `extra_props` lands on every
    entity/Limitation node touched (live tests tag `test_run` this way)."""
    n = repo.driver.execute_query(
        PAPER_EXISTS, parameters_={"paper_id": paper_id}, routing_=RoutingControl.READ
    ).records[0]["n"]
    if not n:
        return {"written": False, "reason": f"no Paper node {paper_id}"}
    rows = build_rows(resolved, paper_id)
    plan = [
        (UPSERT_PROPOSES, rows["proposes"], "proposes"),
        (UPSERT_USES_METHOD, rows["uses"], "uses_method"),
        (UPSERT_EXTENDS, rows["extends"], "extends"),
        (UPSERT_EVALUATES_ON, rows["evaluates_on"], "evaluates_on"),
        (UPSERT_ADDRESSES, rows["addresses"], "addresses"),
        (UPSERT_REPORTS, rows["reports"], "reports"),
        (UPSERT_LIMITATIONS, rows["limitations"], "states_limitation"),
    ]
    props = dict(extra_props or {})

    def work(tx):
        if replace:
            tx.run(CLEAR_PAPER_EXTRACTION, paper_id=paper_id).consume()
        counts = {}
        for query, batch, name in plan:
            if not batch:
                counts[name] = 0
                continue
            rec = tx.run(
                query, paper_id=paper_id, rows=batch, extracted_by=extracted_by, props=props
            ).single()
            counts[name] = rec["n"] if rec else 0
        return counts

    with repo.driver.session() as session:
        counts = session.execute_write(work)
    return {"written": True, "edges": counts}


# ---------------------------------------------------------------- reads


def _read(repo, query: str, **params) -> list[dict]:
    return [
        r.data()
        for r in repo.driver.execute_query(query, parameters_=params, routing_=RoutingControl.READ).records
    ]


def paper_entities(repo, paper_id: str) -> dict:
    """Everything extracted for one paper, ready to show or cite."""
    return {
        "paper_id": paper_id,
        "methods": _read(repo, PAPER_METHODS, paper_id=paper_id),
        "datasets": _read(repo, PAPER_DATASETS, paper_id=paper_id),
        "tasks": _read(repo, PAPER_TASKS, paper_id=paper_id),
        "results": _read(repo, PAPER_RESULTS, paper_id=paper_id),
        "limitations": _read(repo, PAPER_LIMITATIONS, paper_id=paper_id),
    }


def topic_entities(repo, topic_id: str, top: int = 20) -> dict:
    """Most frequent methods, datasets, tasks and metrics across a topic's papers (paper counts)."""
    top = int(top)
    return {
        "topic_id": topic_id,
        "methods": _read(repo, TOPIC_METHODS, topic_id=topic_id, top=top),
        "datasets": _read(repo, TOPIC_DATASETS, topic_id=topic_id, top=top),
        "tasks": _read(repo, TOPIC_TASKS, topic_id=topic_id, top=top),
        "metrics": _read(repo, TOPIC_METRICS, topic_id=topic_id, top=top),
    }
