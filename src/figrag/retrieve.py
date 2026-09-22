import numpy as np
import psycopg


def search(conn: psycopg.Connection, query_vec: np.ndarray, k: int) -> list[dict]:
    """Exact top-k by cosine similarity (embeddings are normalized)."""
    rows = conn.execute(
        "SELECT id, paper_id, page, section, text, 1 - (embedding <=> %s) AS score "
        "FROM chunks ORDER BY embedding <=> %s LIMIT %s",
        (query_vec, query_vec, k),
    ).fetchall()
    cols = ["id", "paper_id", "page", "section", "text", "score"]
    return [dict(zip(cols, row)) for row in rows]
