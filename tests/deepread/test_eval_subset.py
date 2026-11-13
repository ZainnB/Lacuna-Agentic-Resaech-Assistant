"""Eval on a fixed 10-question subset: the seed of the Phase 2 CI gate."""
import os
import random

import pytest

from lacuna.deepread.config import QUESTIONS_PATH, index_hash, load_config, read_jsonl

pytestmark = pytest.mark.live
SUBSET_SIZE = 10


@pytest.fixture(scope="module")
def conn():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    cfg = load_config()
    try:
        from lacuna.deepread.store import connect, stored_index_hash

        c = connect(cfg["db"]["dsn"])
    except Exception as e:
        pytest.skip(f"Postgres unavailable: {e}")
    if stored_index_hash(c) != index_hash(cfg):
        pytest.skip("index missing or stale; run `figrag build`")
    yield c
    c.close()


def test_eval_subset(conn):
    questions = read_jsonl(QUESTIONS_PATH) if QUESTIONS_PATH.exists() else []
    if len(questions) < SUBSET_SIZE:
        pytest.skip(f"need at least {SUBSET_SIZE} questions")
    from lacuna.deepread.evaluate import run

    subset = random.Random(0).sample(sorted(questions, key=lambda q: q["id"]), SUBSET_SIZE)
    result = run(load_config(), subset, conn, judge=True)

    assert len(result["records"]) == SUBSET_SIZE
    assert all(r["pred"] is not None for r in result["records"]), "generator returned no answer (refusal or truncation)"
    for metrics in result["summary"].values():
        for key in ("recall_at_k", "mrr", "accuracy", "faithfulness"):
            assert metrics[key] is None or 0.0 <= metrics[key] <= 1.0
    # Phase 2: assert result["summary"]["all"]["accuracy"] >= a committed baseline floor.
