"""Postgres connection, idempotent schema and the usage ledgers.

The deepread module owns `chunks` and `index_meta` (sql/init.sql). Everything else is here.
`migrate()` is safe to run on every start.
"""

from __future__ import annotations

import json
import sys
from datetime import date

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from lacuna.settings import get_settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id         text PRIMARY KEY,
    doi              text,
    arxiv_id         text,
    s2_id            text,
    title            text NOT NULL,
    year             int,
    publication_date date,
    venue            text,
    citation_count   int,
    abstract         text,
    source           text NOT NULL,
    ingest_status    text NOT NULL DEFAULT 'metadata',   -- metadata | pdf | chunked | extracted
    record           jsonb NOT NULL,                      -- the full PaperRecord
    updated_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS papers_doi_idx   ON papers (doi);
CREATE INDEX IF NOT EXISTS papers_arxiv_idx ON papers (arxiv_id);

-- References as IDs, not graph stubs (AuraDB Free design rule, blueprint 6).
CREATE TABLE IF NOT EXISTS paper_references (
    paper_id text NOT NULL,
    ref_id   text NOT NULL,
    PRIMARY KEY (paper_id, ref_id)
);
CREATE INDEX IF NOT EXISTS paper_references_ref_idx ON paper_references (ref_id);

CREATE TABLE IF NOT EXISTS topics (
    topic_id   text PRIMARY KEY,
    question   text NOT NULL,
    config     jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS topic_papers (
    topic_id text NOT NULL REFERENCES topics (topic_id) ON DELETE CASCADE,
    paper_id text NOT NULL,
    role     text NOT NULL,          -- seed | snowball_ref | snowball_cite
    rank     int,
    score    real,
    reason   text,
    PRIMARY KEY (topic_id, paper_id)
);

-- Key/value caches. One row per request; `value` is the parsed response.
CREATE TABLE IF NOT EXISTS api_cache (
    key        text PRIMARY KEY,
    namespace  text NOT NULL,
    value      jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS llm_cache (
    key        text PRIMARY KEY,
    namespace  text NOT NULL,
    value      jsonb NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- Ledgers. One row per day per provider/source; counters only go up.
CREATE TABLE IF NOT EXISTS provider_usage (
    day           date NOT NULL,
    provider      text NOT NULL,
    model         text NOT NULL,
    calls         int  NOT NULL DEFAULT 0,
    input_tokens  bigint NOT NULL DEFAULT 0,
    output_tokens bigint NOT NULL DEFAULT 0,
    errors        int  NOT NULL DEFAULT 0,
    rate_limited  int  NOT NULL DEFAULT 0,
    fallbacks     int  NOT NULL DEFAULT 0,
    schema_failures int NOT NULL DEFAULT 0,   -- invalid JSON or schema violations (E3 validity rate)
    PRIMARY KEY (day, provider, model)
);
CREATE TABLE IF NOT EXISTS api_usage (
    day          date NOT NULL,
    source       text NOT NULL,
    calls        int  NOT NULL DEFAULT 0,
    cost_usd     numeric(12, 6) NOT NULL DEFAULT 0,
    rate_limited int  NOT NULL DEFAULT 0,
    PRIMARY KEY (day, source)
);

-- Real-use events behind the Stage 2 triggers (blueprint 2).
CREATE TABLE IF NOT EXISTS usage_log (
    id       bigserial PRIMARY KEY,
    at       timestamptz NOT NULL DEFAULT now(),
    event    text NOT NULL,
    topic_id text,
    mode     text,            -- host | server
    user_id  text,
    outcome  text,
    detail   jsonb
);

CREATE TABLE IF NOT EXISTS eval_results (
    id          bigserial PRIMARY KEY,
    eval        text NOT NULL,
    config_hash text NOT NULL,
    created_at  timestamptz NOT NULL DEFAULT now(),
    metrics     jsonb NOT NULL,
    UNIQUE (eval, config_hash)
);
"""

LEDGER_COLUMNS = {
    "provider_usage": {
        "calls",
        "input_tokens",
        "output_tokens",
        "errors",
        "rate_limited",
        "fallbacks",
        "schema_failures",
    },
    "api_usage": {"calls", "cost_usd", "rate_limited"},
}
LEDGER_KEYS = {"provider_usage": ("day", "provider", "model"), "api_usage": ("day", "source")}


def connect(dsn: str | None = None) -> psycopg.Connection:
    return psycopg.connect(dsn or get_settings().postgres_dsn, autocommit=True, row_factory=dict_row)


def migrate(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)


def bump_usage(conn: psycopg.Connection, table: str, keys: dict, **increments) -> None:
    """Add `increments` to one ledger row, creating it if needed.

    `keys` must supply every key column of `table`, e.g.
    bump_usage(conn, "api_usage", {"day": date.today(), "source": "openalex"}, calls=1, cost_usd=0.001)
    """
    cols = LEDGER_COLUMNS[table]
    bad = set(increments) - cols
    if bad or tuple(keys) != LEDGER_KEYS[table]:
        raise ValueError(f"bad ledger write to {table}: keys={tuple(keys)} increments={sorted(increments)}")
    key_cols = ", ".join(keys)
    inc_cols = ", ".join(increments)
    placeholders = ", ".join(["%s"] * (len(keys) + len(increments)))
    updates = ", ".join(f"{c} = {table}.{c} + EXCLUDED.{c}" for c in increments)
    conn.execute(
        f"INSERT INTO {table} ({key_cols}, {inc_cols}) VALUES ({placeholders}) "
        f"ON CONFLICT ({key_cols}) DO UPDATE SET {updates}",
        (*keys.values(), *increments.values()),
    )


def usage_today(conn: psycopg.Connection, table: str, **keys) -> dict:
    """The ledger row for today matching `keys` (e.g. source="openalex"), or zeros."""
    where = " AND ".join(f"{k} = %s" for k in keys)
    row = conn.execute(
        f"SELECT * FROM {table} WHERE day = %s AND {where}", (date.today(), *keys.values())
    ).fetchone()
    return dict(row) if row else {c: 0 for c in LEDGER_COLUMNS[table]}


def log_usage(conn: psycopg.Connection, event: str, **fields) -> None:
    detail = fields.pop("detail", None)
    cols = ["event", *fields]
    vals = [event, *fields.values()]
    if detail is not None:
        cols.append("detail")
        vals.append(Jsonb(detail))
    conn.execute(f"INSERT INTO usage_log ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(vals))})", vals)


def record_eval(conn: psycopg.Connection, eval_name: str, config_hash: str, metrics: dict) -> None:
    conn.execute(
        "INSERT INTO eval_results (eval, config_hash, metrics) VALUES (%s, %s, %s) "
        "ON CONFLICT (eval, config_hash) DO UPDATE SET metrics = EXCLUDED.metrics, created_at = now()",
        (eval_name, config_hash, Jsonb(metrics)),
    )


# ------------------------------------------------------------------ CLI: `lacuna db ...`


def cmd_migrate(args) -> None:
    with connect() as conn:
        migrate(conn)
    print("schema up to date")


def cmd_status(args) -> None:
    with connect() as conn:
        migrate(conn)
        counts = {
            t: conn.execute(f"SELECT count(*) AS n FROM {t}").fetchone()["n"]
            for t in ("papers", "paper_references", "topics", "api_cache", "llm_cache", "eval_results")
        }
        usage = conn.execute(
            "SELECT * FROM api_usage WHERE day = %s UNION ALL "
            "SELECT day, provider || '/' || model, calls, NULL, rate_limited "
            "FROM provider_usage WHERE day = %s",
            (date.today(), date.today()),
        ).fetchall()
    print(json.dumps({"rows": counts, "usage_today": [dict(u) for u in usage]}, indent=2, default=str))


def register(sub) -> None:
    p = sub.add_parser("db", help="Postgres schema and ledgers")
    s = p.add_subparsers(dest="db_cmd", required=True)
    s.add_parser("migrate", help="create or update the schema (idempotent)").set_defaults(fn=cmd_migrate)
    s.add_parser("status", help="row counts and today's API/LLM usage").set_defaults(fn=cmd_status)


if __name__ == "__main__":
    sys.exit("run via `lacuna db ...`")
