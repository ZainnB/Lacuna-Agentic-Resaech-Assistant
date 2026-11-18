"""Chunk store tests. They need Postgres (skipped when absent) but no network and no real model."""

import hashlib

import numpy as np
import pymupdf
import pytest

from lacuna.papers import chunks as pc

pytestmark = pytest.mark.live

TOPIC, OTHER = "TEST_topic", "TEST_other"
P1, P2, P3 = "TEST_W1", "TEST_W2", "TEST_W3"


def fake_embed(texts, *, query=False):
    out = []
    for t in texts:
        seed = int(hashlib.sha256(t.encode()).hexdigest()[:8], 16)
        v = np.random.default_rng(seed).normal(size=768)
        out.append(v / np.linalg.norm(v))
    return np.array(out, dtype=np.float32)


class FakeArxiv:
    def __init__(self, pages=3):
        self.pages, self.calls = pages, []

    def download_pdf(self, arxiv_id, dest_dir):
        self.calls.append(arxiv_id)
        doc = pymupdf.open()
        for i in range(self.pages):
            page = doc.new_page()
            page.insert_text((72, 72), f"1 Introduction\nPage {i + 1} body " + "word " * 600)
        path = dest_dir / f"{arxiv_id}.pdf"
        doc.save(path)
        return path


@pytest.fixture
def db(pg_conn, monkeypatch):
    monkeypatch.setattr(pc, "embed_texts", fake_embed)
    c = pg_conn
    _clean(c)
    c.execute("INSERT INTO topics (topic_id, question) VALUES (%s, 'q'), (%s, 'q')", (TOPIC, OTHER))
    rows = [(P1, "Alpha", "about vision", "2101.00001"), (P2, "Beta", None, None), (P3, "Gamma", "x", None)]
    for pid, title, abstract, aid in rows:
        c.execute(
            "INSERT INTO papers (paper_id, title, abstract, arxiv_id, source, record) "
            "VALUES (%s, %s, %s, %s, 'test', '{}')",
            (pid, title, abstract, aid),
        )
    for t, pid, rank in [(TOPIC, P1, 1), (TOPIC, P2, 2), (OTHER, P3, 1)]:
        c.execute(
            "INSERT INTO topic_papers (topic_id, paper_id, role, rank) VALUES (%s, %s, 'seed', %s)",
            (t, pid, rank),
        )
    yield c
    _clean(c)


def _clean(c):
    c.execute("DELETE FROM served_chunks WHERE chunk_id LIKE 'TEST_%%' OR session LIKE 'TEST_%%'")
    c.execute("DELETE FROM paper_chunks WHERE paper_id LIKE 'TEST_%%'")
    c.execute("DELETE FROM topic_papers WHERE topic_id LIKE 'TEST_%%'")
    c.execute("DELETE FROM topics WHERE topic_id LIKE 'TEST_%%'")
    c.execute("DELETE FROM papers WHERE paper_id LIKE 'TEST_%%'")


def test_abstract_chunks_and_idempotence(db):
    assert pc.ensure_abstract_chunks(db, [P1, P2]) == 2
    assert pc.ensure_abstract_chunks(db, [P1, P2]) == 0
    texts = {c["chunk_id"]: c["text"] for p in (P1, P2) for c in pc.get_chunks(db, p, record_served=False)}
    assert texts[f"{P1}:abs"] == "Alpha\n\nabout vision"
    assert texts[f"{P2}:abs"] == "Beta"  # title only
    db.execute("UPDATE papers SET abstract = 'changed' WHERE paper_id = %s", (P1,))
    assert pc.ensure_abstract_chunks(db, [P1, P2]) == 1


def test_approval_gate_and_top(db, tmp_path):
    arxiv = FakeArxiv()
    assert pc.fetch_fulltext(db, P1, arxiv=arxiv, pdf_dir=tmp_path) == "not_approved"
    assert arxiv.calls == []
    assert pc.approve(db, TOPIC, top=1) == [P1]
    assert pc.approved_papers(db, TOPIC) == [P1]
    assert pc.approve(db, TOPIC, [P2]) == [P2]
    assert pc.fetch_fulltext(db, P2, arxiv=arxiv, pdf_dir=tmp_path) == "no_arxiv"


def test_fulltext_ids_order_and_stale_deletion(db, tmp_path):
    pc.ensure_abstract_chunks(db, [P1])
    pc.approve(db, TOPIC, [P1])
    arxiv = FakeArxiv()
    assert pc.fetch_fulltext(db, P1, arxiv=arxiv, pdf_dir=tmp_path) == "chunked"
    assert pc.fetch_fulltext(db, P1, arxiv=arxiv, pdf_dir=tmp_path) == "already"
    got = pc.get_chunks(db, P1, record_served=False)
    ft = [c for c in got if c["kind"] == "fulltext"]
    assert got[0]["chunk_id"] == f"{P1}:abs"
    assert [c["chunk_id"] for c in ft] == [f"{P1}:{n}" for n in range(1, len(ft) + 1)]
    assert len(ft) > 2 and all(c["page"] for c in ft)
    assert [c["ord"] for c in got] == sorted(c["ord"] for c in got)
    # a stale chunk beyond the new count must be deleted on re-ingest
    db.execute(
        "INSERT INTO paper_chunks (chunk_id, paper_id, kind, ord, text, content_hash) "
        "VALUES (%s, %s, 'fulltext', 999, 'stale', 'x')",
        (f"{P1}:999", P1),
    )
    db.execute("UPDATE papers SET ingest_status = 'metadata' WHERE paper_id = %s", (P1,))
    assert pc.fetch_fulltext(db, P1, arxiv=arxiv, pdf_dir=tmp_path) == "chunked"
    ids = {c["chunk_id"] for c in pc.get_chunks(db, P1, record_served=False)}
    assert f"{P1}:999" not in ids
    status = db.execute("SELECT ingest_status FROM papers WHERE paper_id = %s", (P1,)).fetchone()
    assert status["ingest_status"] == "chunked"


def test_get_chunks_records_served(db):
    pc.ensure_abstract_chunks(db, [P1])
    pc.get_chunks(db, P1, session="TEST_s1")
    rows = db.execute("SELECT chunk_id FROM served_chunks WHERE session = 'TEST_s1'").fetchall()
    assert [r["chunk_id"] for r in rows] == [f"{P1}:abs"]
    pc.get_chunks(db, P1, session="TEST_s2", record_served=False)
    assert not db.execute("SELECT 1 FROM served_chunks WHERE session = 'TEST_s2'").fetchall()


def test_search_restricted_to_topic(db):
    pc.ensure_abstract_chunks(db, [P1, P2, P3])
    res = pc.search_chunks(db, "anything", topic_id=TOPIC, k=10, session="TEST_s3")
    assert {r["paper_id"] for r in res} == {P1, P2}
    assert res[0]["score"] >= res[-1]["score"] and "title" in res[0]
    assert {r["paper_id"] for r in pc.search_chunks(db, "x", paper_ids=[P3])} == {P3}
    served = db.execute("SELECT count(*) AS n FROM served_chunks WHERE session = 'TEST_s3'").fetchone()
    assert served["n"] == 2


def test_index_topic_abstracts_only(db):
    out = pc.index_topic(db, TOPIC)
    assert out == {"papers": 2, "abstract_chunks_written": 2}
