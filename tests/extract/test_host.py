from lacuna.extract.host import save_host_extraction

GOOD = {
    "methods": [], "tasks": [], "metrics": [], "results": [], "limitations": [],
    "datasets": [{"name": "ImageNet", "chunk_id": "W1:abs", "confidence": 0.9}],
}  # fmt: skip


def setup(conn, served=True):
    conn.add_chunk("W1", "W1:abs", "abstract text")
    conn.add_chunk("W1", "W1:1", "body text", ord_=1)
    conn.add_chunk("W2", "W2:abs", "another paper")
    if served:
        conn.served |= {("W1:abs", "s1"), ("W2:abs", "s1")}


def save(conn, payload, **kw):
    kw.setdefault("extracted_by", "host:claude-opus")
    kw.setdefault("session", "s1")
    kw.setdefault("write_graph", False)
    return save_host_extraction(conn, "W1", payload, **kw)


def test_accepts_served_chunks_and_caches(conn):
    setup(conn)
    out = save(conn, GOOD)
    assert out["ok"] and out["errors"] == [] and out["counts"]["datasets"] == 1
    assert [k[3] for k in conn.cache] == ["host:claude-opus"]


def test_rejects_unserved_chunk(conn):
    setup(conn)
    out = save(conn, {**GOOD, "tasks": [{"name": "t", "chunk_id": "W1:1", "confidence": 0.5}]})
    assert not out["ok"] and "never served" in out["errors"][0] and "W1:1" in out["errors"][0]
    assert not conn.cache  # whole submission refused, including the valid dataset


def test_rejects_served_chunk_of_another_paper(conn):
    setup(conn)
    out = save(conn, {**GOOD, "tasks": [{"name": "t", "chunk_id": "W2:abs", "confidence": 0.5}]})
    assert not out["ok"] and "not belonging to paper W1" in out["errors"][0]
    assert not conn.cache


def test_rejects_chunk_served_to_a_different_session(conn):
    setup(conn)
    out = save(conn, GOOD, session="other")
    assert not out["ok"] and "never served" in out["errors"][0]


def test_rejects_nonexistent_chunk(conn):
    setup(conn)
    assert not save(conn, {**GOOD, "datasets": [{"name": "X", "chunk_id": "nope", "confidence": 1}]})["ok"]


def test_extracted_by_must_be_host(conn):
    setup(conn)
    for bad in ("api:gemini/x", "claude", "host:", ""):
        out = save(conn, GOOD, extracted_by=bad)
        assert not out["ok"] and "host:<model>" in out["errors"][0]
    assert not conn.cache


def test_schema_failure_returns_errors_not_exception(conn):
    setup(conn)
    out = save(conn, {"methods": [{"name": "A", "role": "wizard", "chunk_id": "W1:abs", "confidence": 1}]})
    assert not out["ok"] and out["errors"] and "role" in out["errors"][0]
    out = save(conn, {"results": "nope"})
    assert not out["ok"]


def test_unknown_paper(conn):
    out = save_host_extraction(conn, "WX", GOOD, extracted_by="host:m", write_graph=False)
    assert not out["ok"] and "no chunks" in out["errors"][0]
