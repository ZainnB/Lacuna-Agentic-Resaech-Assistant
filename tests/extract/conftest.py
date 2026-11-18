"""Fakes for extract unit tests: an in-memory stand-in for the Postgres tables the package touches,
keyed by the `/* tag */` comment each query carries, and a deterministic embedder."""

import hashlib
import json

import numpy as np
import pytest

from lacuna.extract.prompts import PROMPT_VERSION


class Cursor:
    def __init__(self, rows):
        self.rows = rows

    def fetchall(self):
        return list(self.rows)

    def fetchone(self):
        return self.rows[0] if self.rows else None


def _parse_vec(s: str) -> np.ndarray:
    return np.array(json.loads(s), dtype=float)


class FakeConn:
    def __init__(self):
        self.chunks: list[dict] = []  # {paper_id, chunk_id, kind, ord, section, text, content_hash}
        self.served: set[tuple[str, str]] = set()  # (chunk_id, session)
        self.cache: dict[tuple, dict] = {}
        self.entities: dict[str, dict] = {}
        self.queries: list[str] = []

    def add_chunk(self, paper_id, chunk_id, text, ord_=0):
        h = hashlib.sha256(text.encode()).hexdigest()
        self.chunks.append(
            dict(paper_id=paper_id, chunk_id=chunk_id, kind="abstract" if ord_ == 0 else "fulltext",
                 ord=ord_, section=None, text=text, content_hash=h)
        )  # fmt: skip

    def execute(self, sql, params=()):
        self.queries.append(sql)
        tag = sql.split("*/")[0].replace("/*", "").strip() if sql.startswith("/*") else ""
        return Cursor(getattr(self, f"_q_{tag}", lambda p: [])(params))

    def _q_chunks_for_paper(self, p):
        rows = [c for c in self.chunks if c["paper_id"] == p[0]]
        return sorted(rows, key=lambda c: (c["ord"], c["chunk_id"]))

    def _q_served_for_session(self, p):
        session, ids = p
        return [{"chunk_id": c} for c in ids if (c, session) in self.served]

    def _q_extraction_cache_get(self, p):
        return [
            {"extracted_by": k[3], "value": v} for k, v in self.cache.items() if k[:3] == (p[0], p[1], p[2])
        ]

    def _q_extraction_cache_put(self, p):
        paper_id, chash, version, by, jsonb = p
        self.cache[(paper_id, chash, version, by)] = jsonb.obj
        return []

    def _q_entity_exact(self, p):
        type_, eid, norm = p
        return [
            {"entity_id": e["entity_id"], "canonical_name": e["canonical_name"]}
            for e in self.entities.values()
            if e["type"] == type_ and (e["entity_id"] == eid or norm in e["aliases"])
        ][:1]

    def _q_entity_get(self, p):
        return [{"entity_id": p[0]}] if p[0] in self.entities else []

    def _q_entity_insert(self, p):
        eid, type_, canon, aliases, vec = p
        self.entities.setdefault(
            eid,
            {"entity_id": eid, "type": type_, "canonical_name": canon, "aliases": list(aliases),
             "embedding": _parse_vec(vec)},
        )  # fmt: skip
        return []

    def _q_entity_add_alias(self, p):
        alias, eid = p[0], p[1]
        if alias not in self.entities[eid]["aliases"]:
            self.entities[eid]["aliases"].append(alias)
        return []

    def _q_entity_similar(self, p):
        q, type_ = _parse_vec(p[0]), p[1]
        best = None
        for e in self.entities.values():
            if e["type"] != type_:
                continue
            sim = float(q @ e["embedding"] / (np.linalg.norm(q) * np.linalg.norm(e["embedding"])))
            if best is None or sim > best["sim"]:
                best = dict(entity_id=e["entity_id"], canonical_name=e["canonical_name"], sim=sim)
        return [best] if best else []


class FakeEmbedder:
    """Hash-seeded unit vectors (near-orthogonal in 32-d); `near` maps a text to another text whose
    vector it should be a tiny perturbation of."""

    def __init__(self, near=None):
        self.near = near or {}
        self.calls = []

    def _vec(self, text):
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        v = np.random.default_rng(seed).normal(size=32)
        return v / np.linalg.norm(v)

    def __call__(self, texts):
        out = []
        for t in texts:
            self.calls.append(t)
            if t in self.near:
                v = self._vec(self.near[t]) + 0.02 * self._vec(t)
                v = v / np.linalg.norm(v)
            else:
                v = self._vec(t)
            out.append(v)
        return np.array(out)


@pytest.fixture
def conn():
    return FakeConn()


@pytest.fixture
def make_embedder():
    return FakeEmbedder


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def version():
    return PROMPT_VERSION


class FakeTx:
    def __init__(self):
        self.runs = []

    def run(self, query, **params):
        self.runs.append((query, params))

        class R:
            def consume(self_):
                return None

            def single(self_):
                return {"n": len(params.get("rows", []))}

        return R()


class _FakeRepo:
    def __init__(self, exists=1):
        self.tx = FakeTx()
        self.exists = exists
        outer = self

        class Session:
            def __enter__(s):
                return s

            def __exit__(s, *a):
                return None

            def execute_write(s, work):
                return work(outer.tx)

        class Result:
            records = [{"n": exists}]

        class Driver:
            def execute_query(d, q, parameters_=None, routing_=None):
                return Result

            def session(d):
                return Session()

        self.driver = Driver()


@pytest.fixture
def FakeRepo():
    return _FakeRepo
