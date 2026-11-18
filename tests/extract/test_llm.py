import pytest

from lacuna.extract import llm
from lacuna.extract.llm import NoChunksError, extract_paper, make_batches
from lacuna.extract.schemas import Extraction
from lacuna.llm import ChainExhaustedError, LLMResult

SPEC = "gemini/gemini-3.5-flash-lite"


def ds(name, chunk_id, conf=0.8):
    return {"name": name, "chunk_id": chunk_id, "confidence": conf}


def payload(**kw):
    base = {k: [] for k in Extraction.LISTS}
    return base | kw


class FakeLLM:
    """Stands in for lacuna.llm.complete_json; `answer(user, schema)` returns the parsed dict."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def __call__(
        self, role, system, user, schema, *, chain=None, max_tokens=1024, temperature=0.0, use_cache=True
    ):
        self.calls.append(
            {"role": role, "user": user, "chain": chain, "use_cache": use_cache, "schema": schema}
        )
        out = self.answer(user)
        if isinstance(out, Exception):
            raise out
        provider, model = (chain or [SPEC])[0].split("/", 1)
        return LLMResult(parsed=out, text=None, provider=provider, model=model, input_tokens=100,
                         output_tokens=20, latency_ms=50)  # fmt: skip


@pytest.fixture
def fake(monkeypatch):
    def install(answer):
        f = FakeLLM(answer)
        monkeypatch.setattr(llm, "complete_json", f)
        return f

    return install


def seed(conn, paper="W1", n=1):
    for i in range(n):
        conn.add_chunk(paper, f"{paper}:abs" if i == 0 else f"{paper}:{i}", f"chunk text {i} " * 5, ord_=i)


def test_make_batches_by_count_and_tokens():
    chunks = [{"chunk_id": str(i), "text": "x" * 400} for i in range(14)]
    assert [len(b) for b in make_batches(chunks, max_chunks=6)] == [6, 6, 2]
    big = [{"chunk_id": str(i), "text": "x" * 8000} for i in range(5)]  # ~2000 tokens each
    assert [len(b) for b in make_batches(big, max_chunks=6, token_budget=6000)] == [2, 2, 1]
    assert make_batches([], max_chunks=6) == []


def test_abstract_first_and_prompt_carries_chunk_ids(conn, fake):
    seed(conn, n=3)
    f = fake(lambda u: payload())
    extract_paper(conn, "W1", chain=[SPEC])
    user = f.calls[0]["user"]
    assert user.index("[chunk_id: W1:abs]") < user.index("[chunk_id: W1:1]") < user.index("[chunk_id: W1:2]")
    assert f.calls[0]["role"] == "extractor" and f.calls[0]["chain"] == [SPEC]


def test_foreign_citations_dropped_and_counted(conn, fake):
    seed(conn)
    fake(lambda u: payload(datasets=[ds("ImageNet", "W1:abs"), ds("COCO", "W9:abs"), ds("VOC", "made-up")]))
    res = extract_paper(conn, "W1", chain=[SPEC])
    assert [d.name for d in res.extraction.datasets] == ["ImageNet"]
    assert res.rejected_citations == 2
    assert res.extracted_by == f"api:{SPEC}"


def test_citation_must_be_in_its_own_batch(conn, fake):
    seed(conn, n=3)

    def answer(user):  # cites chunk 2 regardless of which batch it was shown
        return payload(datasets=[ds("A", "W1:abs"), ds("B", "W1:2")])

    f = fake(answer)
    res = extract_paper(conn, "W1", chain=[SPEC], max_chunks_per_call=2)
    assert res.calls == 2 and len(f.calls) == 2
    # batch 1 = {abs, 1}: B rejected; batch 2 = {2}: A rejected. Each survives once.
    assert res.rejected_citations == 2
    assert sorted(d.name for d in res.extraction.datasets) == ["A", "B"]
    assert res.tokens == {"input": 200, "output": 40}


def test_batches_merge_and_dedupe(conn, fake):
    seed(conn, n=4)
    fake(lambda u: payload(tasks=[{"name": "t", "chunk_id": "W1:abs", "confidence": 0.5}]))
    res = extract_paper(conn, "W1", chain=[SPEC], max_chunks_per_call=2)
    assert len(res.extraction.tasks) == 1  # only batch 1 may cite abs; batch 2's copy is rejected
    assert res.rejected_citations == 1


def test_cache_hit_makes_no_call(conn, fake):
    seed(conn)
    f = fake(lambda u: payload(datasets=[ds("ImageNet", "W1:abs")]))
    first = extract_paper(conn, "W1", chain=[SPEC])
    assert not first.cached and len(conn.cache) == 1
    second = extract_paper(conn, "W1", chain=[SPEC])
    assert second.cached and second.calls == 0 and len(f.calls) == 1
    assert second.extraction == first.extraction and second.extracted_by == first.extracted_by
    # no explicit chain: any api: extraction is a hit
    assert extract_paper(conn, "W1").cached


def test_cache_misses_on_new_chunk_text_or_other_extractor(conn, fake):
    seed(conn)
    f = fake(lambda u: payload())
    extract_paper(conn, "W1", chain=[SPEC])
    extract_paper(conn, "W1", chain=["groq/openai/gpt-oss-120b"])
    assert len(f.calls) == 2
    conn.add_chunk("W1", "W1:1", "a new chunk", ord_=1)
    extract_paper(conn, "W1", chain=[SPEC])
    assert len(f.calls) == 3 and len(conn.cache) == 3


def test_use_cache_false_bypasses_and_writes_nothing(conn, fake):
    seed(conn)
    f = fake(lambda u: payload())
    extract_paper(conn, "W1", chain=[SPEC], use_cache=False)
    extract_paper(conn, "W1", chain=[SPEC], use_cache=False)
    assert len(f.calls) == 2 and not conn.cache and f.calls[0]["use_cache"] is False


def test_strict_false_counts_failures_and_skips_cache(conn, fake):
    seed(conn, n=3)
    n = {"i": 0}

    def answer(user):
        n["i"] += 1
        return ChainExhaustedError([{"outcome": "invalid"}]) if n["i"] == 1 else payload()

    fake(answer)
    res = extract_paper(conn, "W1", chain=[SPEC], max_chunks_per_call=2, strict=False)
    assert res.calls == 2 and res.failed_calls == 1 and not conn.cache
    with pytest.raises(ChainExhaustedError):
        n["i"] = 0
        extract_paper(conn, "W1", chain=[SPEC], max_chunks_per_call=2)


def test_invalid_item_is_dropped_and_counted_not_silently_kept(conn, fake):
    seed(conn)
    fake(
        lambda u: payload(
            methods=[
                {"name": "A", "role": "bogus", "chunk_id": "W1:abs", "confidence": 1},
                {
                    "name": "B",
                    "variant": None,
                    "role": "used",
                    "extends": None,
                    "chunk_id": "W1:abs",
                    "confidence": 1,
                },
            ]
        )
    )
    res = extract_paper(conn, "W1", chain=[SPEC], strict=True)  # strict: an item drop is not a failure
    assert res.failed_calls == 0 and res.invalid_items == 1
    assert [m.name for m in res.extraction.methods] == ["B"]


def test_no_chunks(conn, fake):
    fake(lambda u: payload())
    with pytest.raises(NoChunksError):
        extract_paper(conn, "WNONE")
