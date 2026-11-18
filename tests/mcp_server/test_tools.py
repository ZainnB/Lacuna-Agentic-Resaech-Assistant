"""In-process tool tests: underlying packages are faked, no network and no services."""

import asyncio
import json

import pytest

from lacuna import db, papers
from lacuna.mcp_server import tools as tools_mod
from lacuna.mcp_server.server import build_server
from lacuna.mcp_server.state import State
from lacuna.models import PaperRecord


class FakeResult:
    def __init__(self, rows=None):
        self.rows = rows or []

    def fetchall(self):
        return self.rows

    def fetchone(self):
        return self.rows[0] if self.rows else None


class FakeConn:
    closed = False

    def __init__(self):
        self.sql: list[tuple] = []
        self.responders: list = []

    def execute(self, sql, params=None):
        self.sql.append((sql, params))
        for match, rows in self.responders:
            if match in sql:
                return FakeResult(rows(params) if callable(rows) else rows)
        return FakeResult()

    def rollback(self):
        pass


class FakeRepo:
    def __init__(self):
        self.driver = self

    def verify_connectivity(self):
        pass


@pytest.fixture
def env(monkeypatch):
    conn = FakeConn()
    state = State(session="test-session")
    state._conn = conn
    state._repo = FakeRepo()
    usage = []
    monkeypatch.setattr(db, "log_usage", lambda c, event, **f: usage.append((event, f)))
    server = build_server(state)
    return server, server.lacuna_tools, conn, usage


def rec(i, arxiv=True):
    return PaperRecord(
        paper_id=f"W{i}", title="T" * 300, year=2020, citation_count=i, abstract="a" * 2000,
        arxiv_id="1234.5" if arxiv else None,
    )  # fmt: skip


def test_tool_and_prompt_lists():
    server = build_server(State())
    names = {t.name: t for t in asyncio.run(server.list_tools())}
    assert set(names) == {
        "search_papers", "scout_topic", "approve_papers", "get_chunks", "save_extraction",
        "extract_free_tier", "ask", "related", "topic_overview", "validate_citations",
    }  # fmt: skip
    assert all(len(t.description) > 60 for t in names.values())
    assert "ASK THE USER" in names["approve_papers"].description
    assert (
        "chunk_id" in names["save_extraction"].description
        and "methods" in names["save_extraction"].description
    )
    prompts = {p.name for p in asyncio.run(server.list_prompts())}
    assert prompts == {"scout_topic", "draft_review", "deep_read"}
    text = asyncio.run(server.get_prompt("draft_review", {"topic_id": "t_x"}))
    assert "validate_citations" in json.dumps(text.model_dump(), default=str)


def test_call_through_server_returns_dict(env):
    server, tools, conn, usage = env
    conn.responders.append(("FROM paper_chunks", []))
    res = asyncio.run(server.call_tool("get_chunks", {"paper_id": "W1"}))
    payload = res.structured_content or json.loads(res.content[0].text)
    assert payload["chunks"] == []
    assert usage[0][0] == "mcp:get_chunks"


def test_search_papers_caps_and_truncates(env, monkeypatch):
    _, tools, _, usage = env
    seen = {}

    class OA:
        def search_works(self, q, *, from_date, to_date, max_results):
            seen.update(max_results=max_results, from_date=from_date)
            return [rec(i) for i in range(max_results)]

    class Clients:
        openalex = OA()

        def close(self):
            pass

    monkeypatch.setattr("lacuna.ingest.make_clients", lambda conn=None: Clients())
    out = tools.search_papers("x", limit=5000, from_year=2020)
    assert seen["max_results"] == tools_mod.MAX_SEARCH == len(out["papers"])
    assert len(out["papers"][0]["abstract"]) <= 600 and len(out["papers"][0]["title"]) <= 140
    assert str(seen["from_date"]) == "2020-01-01"
    assert usage[0][0] == "mcp:search_papers" and usage[0][1]["mode"] == "host"
    assert usage[0][1]["outcome"] == "ok"


def test_errors_are_returned_not_raised(env, monkeypatch):
    _, tools, _, usage = env
    monkeypatch.setattr(papers, "search_chunks", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out = tools.ask("t1", "q")
    assert "RuntimeError: boom" in out["error"]
    assert usage[-1][1]["outcome"] == "error" and usage[-1][1]["topic_id"] == "t1"


def test_ask_caps_k_and_returns_evidence(env, monkeypatch):
    _, tools, _, usage = env
    got = {}

    def fake_search(conn, q, *, topic_id, k, session, **kw):
        got.update(k=k, session=session, topic_id=topic_id)
        return [
            {"chunk_id": f"W{i}:abs", "paper_id": f"W{i}", "title": "t", "score": 0.5, "text": "x"}
            for i in range(k)
        ]

    monkeypatch.setattr(papers, "search_chunks", fake_search)
    monkeypatch.setattr("lacuna.graph.entities.topic_entities", lambda repo, tid, top: {"methods": []})
    out = tools.ask("t1", "q", k=999)
    assert got == {"k": 20, "session": "test-session", "topic_id": "t1"}
    assert len(out["evidence"]) == 20 and out["top_entities"] == {"methods": []}


def test_get_chunks_cap_marks_only_returned_as_served(env, monkeypatch):
    _, tools, conn, _ = env
    rows = [
        {"chunk_id": f"W1:{i}", "kind": "fulltext", "page": 1, "section": None, "text": "t"}
        for i in range(100)
    ]
    monkeypatch.setattr(papers, "get_chunks", lambda *a, **k: rows)
    out = tools.get_chunks("W1")
    assert len(out["chunks"]) == tools_mod.MAX_CHUNKS and out["truncated"] and out["total"] == 100
    served = [p for s, p in conn.sql if "INSERT INTO served_chunks" in s]
    assert len(served) == 60 and served[0] == ("W1:0", "test-session")
    out2 = tools.get_chunks("W1", offset=60)
    assert len(out2["chunks"]) == 40 and not out2["truncated"]
    assert "error" in tools.get_chunks("W1", kinds=["bogus"])


def test_scout_topic_caps_and_persists(env, monkeypatch):
    _, tools, _, usage = env
    import lacuna.scout as scout_pkg
    from lacuna.scout import store

    seen = {}

    def fake_scout(question, openalex, cfg):
        seen["cfg"] = cfg
        return scout_pkg.ScoutResult(
            question=question, status="ok",
            papers=[scout_pkg.Ranked(rec(i), "seed", 1.0, "r", rank=i + 1) for i in range(3)],
        )  # fmt: skip

    class Clients:
        openalex = object()

        def close(self):
            pass

    monkeypatch.setattr("lacuna.ingest.make_clients", lambda conn=None: Clients())
    monkeypatch.setattr(scout_pkg, "scout", fake_scout)
    monkeypatch.setattr(store, "persist", lambda conn, result, cfg, graph: "t_abc")
    monkeypatch.setattr(papers, "ensure_abstract_chunks", lambda conn, ids: len(ids))
    out = tools.scout_topic("q", max_papers=9999, seeds=9999)
    assert (seen["cfg"].max_papers, seen["cfg"].seeds) == (100, 30)
    assert out["topic_id"] == "t_abc" and out["papers"][0]["rank"] == 1 and out["papers"][0]["has_arxiv"]
    assert usage[-1][1]["topic_id"] == "t_abc"


def test_approve_requires_selection_and_caps_fulltext(env, monkeypatch):
    _, tools, _, _ = env
    assert "error" in tools.approve_papers("t1")
    ids = [f"W{i}" for i in range(30)]
    monkeypatch.setattr(papers, "approve", lambda conn, tid, pids, top=None: ids)
    fetched = []
    monkeypatch.setattr(papers, "fetch_fulltext", lambda conn, pid, arxiv=None: fetched.append(pid) or "ok")

    class Clients:
        arxiv = object()

        def close(self):
            pass

    monkeypatch.setattr("lacuna.ingest.make_clients", lambda conn=None: Clients())
    out = tools.approve_papers("t1", top=30)
    assert len(fetched) == 20 and out["fulltext"]["W29"] == "skipped_cap"
    fetched.clear()
    tools.approve_papers("t1", top=3, fetch_fulltext=False)
    assert not fetched


def test_save_extraction_passes_host_model_and_session(env, monkeypatch):
    _, tools, _, usage = env
    got = {}

    def fake_save(conn, pid, payload, *, extracted_by, session, repo):
        got.update(pid=pid, by=extracted_by, session=session, repo=repo)
        return {"ok": True, "errors": [], "counts": {"methods": 1}}

    monkeypatch.setattr("lacuna.extract.host.save_host_extraction", fake_save)
    out = tools.save_extraction("W1", {"methods": []}, "claude-sonnet-5")
    assert out["ok"] and got["by"] == "host:claude-sonnet-5" and got["session"] == "test-session"
    assert got["repo"] is not None
    assert "error" in tools.save_extraction("W1", {}, " ")
    monkeypatch.setattr(
        "lacuna.extract.host.save_host_extraction",
        lambda *a, **k: {"ok": False, "errors": ["bad"], "counts": {}},
    )
    rejected = tools.save_extraction("W1", {}, "m")
    assert rejected["error"] and rejected["errors"] == ["bad"] and usage[-1][1]["outcome"] == "error"


def test_validate_citations(env):
    _, tools, conn, _ = env
    conn.responders += [
        (
            "FROM paper_chunks",
            [{"chunk_id": "W1:abs", "paper_id": "W1"}, {"chunk_id": "W2:abs", "paper_id": "W2"}],
        ),
        ("FROM served_chunks", [{"chunk_id": "W1:abs"}]),
        ("FROM topic_papers", [{"paper_id": "W1"}]),
    ]
    out = tools.validate_citations(["W1:abs", "W2:abs", "nope"], topic_id="t1")
    by = {r["chunk_id"]: r for r in out["results"]}
    assert by["W1:abs"] == {
        "chunk_id": "W1:abs", "exists": True, "paper_id": "W1", "in_topic": True, "served_this_session": True,
    }  # fmt: skip
    assert by["W2:abs"]["in_topic"] is False and not by["W2:abs"]["served_this_session"]
    assert not by["nope"]["exists"]
    assert out["ok"] is False and out["n_invalid"] == 2
    assert "error" in tools.validate_citations([])


def test_topic_overview_and_related(env, monkeypatch):
    _, tools, conn, _ = env
    conn.responders.append(("FROM topic_papers", [{"papers": 5, "approved": 2}]))

    class Repo(FakeRepo):
        def topic_summary(self, tid, top_n=10):
            if tid == "bad":
                return None
            return {"question": "q", "n_papers": 5, "n_cites_within": 3, "top_cited": [{"title": "x" * 500}]}

        def neighbourhood(self, pid):
            return None

    tools.state._repo = Repo()
    ent = {"methods": [1], "datasets": [], "tasks": [], "metrics": []}
    monkeypatch.setattr("lacuna.graph.entities.topic_entities", lambda repo, tid, top: ent)
    out = tools.topic_overview("t1")
    assert (
        out["n_papers"] == 5 and out["n_approved_fulltext"] == 2 and len(out["most_cited"][0]["title"]) <= 140
    )
    assert "error" in tools.topic_overview("bad")
    assert "error" in tools.related("W9")


def test_usage_log_failure_does_not_break_tool(env, monkeypatch):
    _, tools, _, _ = env

    def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(db, "log_usage", boom)
    monkeypatch.setattr(papers, "get_chunks", lambda *a, **k: [])
    assert tools.get_chunks("W1")["chunks"] == []
