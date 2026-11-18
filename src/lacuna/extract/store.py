"""Postgres access for extraction: chunks, the extraction cache, and the content hash."""

from __future__ import annotations

import hashlib

from psycopg.types.json import Jsonb

from lacuna.extract.prompts import PROMPT_VERSION
from lacuna.extract.schemas import Extraction


def load_chunks(conn, paper_id: str) -> list[dict]:
    """All chunks of a paper, abstract first (ord 0), then reading order."""
    return list(
        conn.execute(
            "/* chunks_for_paper */ SELECT chunk_id, kind, ord, section, text, content_hash "
            "FROM paper_chunks WHERE paper_id = %s ORDER BY ord, chunk_id",
            (paper_id,),
        ).fetchall()
    )


def content_hash(chunks: list[dict]) -> str:
    """Hash of the concatenated chunk hashes: changes iff any chunk's text or the chunk set changes."""
    return hashlib.sha256("".join(c["content_hash"] for c in chunks).encode()).hexdigest()


def cache_get(conn, paper_id: str, chash: str, extracted_by: list[str] | None = None) -> tuple | None:
    """A cached (Extraction, extracted_by) for this paper+content+prompt, optionally limited to the
    given extractors (tried in order); None when absent."""
    rows = conn.execute(
        "/* extraction_cache_get */ SELECT extracted_by, value FROM extraction_cache "
        "WHERE paper_id = %s AND content_hash = %s AND prompt_version = %s",
        (paper_id, chash, PROMPT_VERSION),
    ).fetchall()
    by = {r["extracted_by"]: r["value"] for r in rows}
    order = extracted_by if extracted_by is not None else sorted(by)
    for name in order:
        if name in by:
            return Extraction.model_validate(by[name]), name
    return None


def cache_put(conn, paper_id: str, chash: str, extracted_by: str, extraction: Extraction) -> None:
    conn.execute(
        "/* extraction_cache_put */ INSERT INTO extraction_cache "
        "(paper_id, content_hash, prompt_version, extracted_by, value) VALUES (%s, %s, %s, %s, %s) "
        "ON CONFLICT (paper_id, content_hash, prompt_version, extracted_by) "
        "DO UPDATE SET value = EXCLUDED.value, created_at = now()",
        (paper_id, chash, PROMPT_VERSION, extracted_by, Jsonb(extraction.model_dump(mode="json"))),
    )
