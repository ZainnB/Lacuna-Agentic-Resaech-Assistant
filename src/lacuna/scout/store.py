"""Persist a scout result: papers and references in Postgres, Paper/Author/CITES/Topic in Neo4j.

The topic id is a hash of the question and the config, so re-running the same scout overwrites
its own rows instead of accumulating duplicates.
"""

from __future__ import annotations

import psycopg
from psycopg.types.json import Jsonb

from lacuna.models import PaperRecord
from lacuna.scout.core import ScoutConfig, ScoutResult
from lacuna.util import stable_hash


def topic_id_for(question: str, cfg: ScoutConfig) -> str:
    return "t_" + stable_hash(question.strip().lower(), cfg.as_dict(), n=10)


def upsert_papers(conn: psycopg.Connection, records: list[PaperRecord]) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO papers (paper_id, doi, arxiv_id, s2_id, title, year, publication_date, venue,
                                citation_count, abstract, source, record)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (paper_id) DO UPDATE SET
                doi = COALESCE(EXCLUDED.doi, papers.doi),
                arxiv_id = COALESCE(EXCLUDED.arxiv_id, papers.arxiv_id),
                s2_id = COALESCE(EXCLUDED.s2_id, papers.s2_id),
                title = EXCLUDED.title, year = EXCLUDED.year,
                publication_date = EXCLUDED.publication_date, venue = EXCLUDED.venue,
                citation_count = EXCLUDED.citation_count,
                abstract = COALESCE(EXCLUDED.abstract, papers.abstract),
                record = EXCLUDED.record, updated_at = now()
            """,
            [
                (
                    r.paper_id,
                    r.doi,
                    r.arxiv_id,
                    r.s2_id,
                    r.title,
                    r.year,
                    r.publication_date,
                    r.venue,
                    r.citation_count,
                    r.abstract,
                    r.source,
                    Jsonb(r.model_dump(mode="json")),
                )
                for r in records
            ],
        )
        cur.executemany(
            "INSERT INTO paper_references (paper_id, ref_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            [(r.paper_id, ref) for r in records for ref in r.referenced_ids],
        )


def save_topic(conn: psycopg.Connection, topic_id: str, result: ScoutResult, cfg: ScoutConfig) -> None:
    with conn.transaction():
        conn.execute(
            "INSERT INTO topics (topic_id, question, config) VALUES (%s, %s, %s) "
            "ON CONFLICT (topic_id) DO UPDATE SET question = EXCLUDED.question, config = EXCLUDED.config",
            (topic_id, result.question, Jsonb({**cfg.as_dict(), "status": result.status})),
        )
        conn.execute("DELETE FROM topic_papers WHERE topic_id = %s", (topic_id,))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO topic_papers (topic_id, paper_id, role, rank, score, reason) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                [(topic_id, r.record.paper_id, r.role, r.rank, r.score, r.reason) for r in result.papers],
            )


def persist(conn: psycopg.Connection, result: ScoutResult, cfg: ScoutConfig, graph=None) -> str:
    """Write everything; returns the topic id. `graph` is a GraphRepo or None to skip Neo4j."""
    topic_id = topic_id_for(result.question, cfg)
    records = [r.record for r in result.papers]
    upsert_papers(conn, records)
    save_topic(conn, topic_id, result, cfg)
    if graph is not None:
        graph.ensure_constraints()
        graph.upsert_papers(records)
        graph.upsert_topic(
            topic_id,
            result.question,
            [
                {"paper_id": r.record.paper_id, "role": r.role, "rank": r.rank, "score": r.score}
                for r in result.papers
            ],
        )
        graph.link_citations(records)
    return topic_id
