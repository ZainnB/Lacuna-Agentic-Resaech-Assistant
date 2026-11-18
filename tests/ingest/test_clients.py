import json
from datetime import date
from pathlib import Path

import httpx
import pytest

from lacuna.cache import MemoryCache
from lacuna.ingest import make_clients, normalise_arxiv_id, resolve_hits
from lacuna.ingest.arxiv import ArxivClient, parse_atom
from lacuna.ingest.http import BudgetExceeded, CachedClient
from lacuna.ingest.openalex import PRICES, OpenAlexClient, work_to_record
from lacuna.ingest.semantic_scholar import S2Client
from lacuna.models import SearchHit
from lacuna.ratelimit import RateLimiter

FIX = Path(__file__).parents[1] / "fixtures" / "ingest"
OA = json.loads((FIX / "openalex_search.json").read_text())


class Fake:
    """httpx transport recording requests and replaying queued responses (or a handler)."""

    def __init__(self, handler):
        self.requests: list[httpx.Request] = []
        self.handler = handler

    def __call__(self, request):
        self.requests.append(request)
        return self.handler(request, len(self.requests))


def mk(handler, **kw):
    f = Fake(handler)
    return f, httpx.MockTransport(f)


def limiter():
    return RateLimiter(1000, burst=1000, sleep=lambda s: None)


def oa_client(settings, handler, **kw):
    f, t = mk(handler)
    return f, OpenAlexClient(
        settings, MemoryCache(), transport=t, limiter=limiter(), sleep=lambda s: None, **kw
    )


def page(n, start=0):
    return {
        "results": [{"id": f"https://openalex.org/W{start + i}", "title": f"t{start + i}"} for i in range(n)]
    }


def test_work_mapping(settings):
    r = work_to_record(OA["results"][1])
    assert r.paper_id == "W4385187240"
    assert r.doi == "10.1109/iscas46773.2023.10181988"
    assert r.arxiv_id == "2302.09108"
    assert r.abstract and " " in r.abstract and not r.abstract.startswith(" ")
    assert r.year == 2023 and r.publication_date == date.fromisoformat(OA["results"][1]["publication_date"])
    assert r.authors[0].author_id.startswith("A") and r.authors[0].position == 0
    assert r.authors[1].position == 1
    assert len(r.referenced_ids) == 5 and all(x.startswith("W") for x in r.referenced_ids)
    assert r.venue


def test_abstract_rebuild_and_arxiv_from_doi():
    r = work_to_record(
        {
            "id": "https://openalex.org/W1",
            "doi": "https://doi.org/10.48550/arXiv.1706.03762v5",
            "abstract_inverted_index": {"world": [1], "Hello": [0], "again": [2, 3]},
            "authorships": [{"raw_author_name": "Jane  Doe", "author": {}}],
        }
    )
    assert r.abstract == "Hello world again again"
    assert r.arxiv_id == "1706.03762" and r.doi == "10.48550/arxiv.1706.03762v5"
    assert r.authors[0].author_id == "name:jane doe"


def test_search_pagination_and_params(settings):
    def h(req, n):
        p = req.url.params
        assert p["search"] == "vit" and "select" in p and "api_key" not in p
        assert p["filter"] == "from_publication_date:2020-01-01,to_publication_date:2021-12-31"
        return httpx.Response(200, json=page(int(p["per_page"]) if n < 2 else 50, 0 if n < 2 else 200))

    f, c = oa_client(settings, h)
    out = c.search_works("vit", from_date=date(2020, 1, 1), to_date=date(2021, 12, 31), max_results=250)
    assert len(out) == 250
    assert [r.url.params["per_page"] for r in f.requests] == ["200", "50"]
    assert [r.url.params["page"] for r in f.requests] == ["1", "2"]
    assert c.http.usage_today()["cost_usd"] == pytest.approx(2 * PRICES["search"])


def test_works_by_ids_batches_of_100(settings):
    def h(req, n):
        ids = req.url.params["filter"].removeprefix("openalex:").split("|")
        return httpx.Response(
            200, json={"results": [{"id": f"https://openalex.org/{i}", "title": i} for i in ids]}
        )

    f, c = oa_client(settings, h)
    ids = [f"W{i}" for i in range(1, 251)] + ["https://openalex.org/W1"]
    out = c.works_by_ids(ids)
    assert [len(r.url.params["filter"].split("|")) for r in f.requests] == [100, 100, 50]
    assert [r.paper_id for r in out] == [f"W{i}" for i in range(1, 251)]


def test_cited_by_filter(settings):
    f, c = oa_client(settings, lambda req, n: httpx.Response(200, json=page(3)))
    assert len(c.cited_by("https://openalex.org/W9", max_results=50)) == 3
    assert f.requests[0].url.params["filter"] == "cites:W9"


def test_api_key_and_mailto_only_when_set(settings):
    f, c = oa_client(settings, lambda req, n: httpx.Response(200, json=page(1)))
    c.cited_by("W1")
    assert "api_key" not in f.requests[0].url.params and "mailto" not in f.requests[0].url.params
    assert f.requests[0].headers["user-agent"] == "lacuna/0.1"
    s = settings.model_copy(update={"openalex_api_key": "K", "contact_email": "x@example.org"})
    f, c = oa_client(s, lambda req, n: httpx.Response(200, json=page(1)))
    c.cited_by("W1")
    assert (
        f.requests[0].url.params["api_key"] == "K" and f.requests[0].url.params["mailto"] == "x@example.org"
    )
    assert c.budget_usd == 0.90


def test_unauthenticated_budget_is_capped_at_shared_allowance(settings):
    _, c = oa_client(settings, lambda req, n: httpx.Response(200, json=page(0)))
    assert c.budget_usd < 0.10


def test_budget_exceeded_before_request(settings):
    s = settings.model_copy(update={"openalex_daily_budget_usd": 0.0015, "openalex_api_key": "K"})
    f, c = oa_client(s, lambda req, n: httpx.Response(200, json=page(0) | OA["results"][0]))
    c.search_works("a")  # 0.001 spent
    with pytest.raises(BudgetExceeded):
        c.search_works("b")  # would be 0.002
    assert len(f.requests) == 1
    c.work("W1")  # free lookups still pass
    assert len(f.requests) == 2


def test_cache_hit_no_http_no_ledger(settings):
    f, c = oa_client(settings, lambda req, n: httpx.Response(200, json=page(2)))
    c.search_works("a", max_results=2)
    before = c.http.usage_today()
    c.search_works("a", max_results=2)
    assert len(f.requests) == 1 and c.http.usage_today() == before


def test_cache_hit_does_not_check_budget(settings):
    s = settings.model_copy(update={"openalex_daily_budget_usd": 0.001, "openalex_api_key": "K"})
    f, c = oa_client(s, lambda req, n: httpx.Response(200, json=page(1)))
    c.search_works("a", max_results=1)
    assert len(c.search_works("a", max_results=1)) == 1  # served from cache at the cap


def test_429_retry_with_retry_after_and_penalise():
    sleeps = []
    lim = limiter()
    penalised = []
    lim.penalise = penalised.append

    def h(req, n):
        if n < 3:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200, json={"ok": 1})

    f, t = mk(h)
    c = CachedClient("x", MemoryCache(), lim, transport=t, sleep=sleeps.append)
    assert c.get("https://x.test/a", cost=0.5) == {"ok": 1}
    assert sleeps == [7.0, 7.0] and penalised == [7.0, 7.0]
    u = c.usage_today()
    assert u["rate_limited"] == 2 and u["calls"] == 3 and u["cost_usd"] == 0.5


def test_429_surviving_retries_raises():
    sleeps = []
    f, t = mk(lambda req, n: httpx.Response(429))
    c = CachedClient("x", MemoryCache(), limiter(), transport=t, sleep=sleeps.append)
    with pytest.raises(httpx.HTTPStatusError) as e:
        c.get("https://x.test/a")
    assert e.value.response.status_code == 429
    assert len(f.requests) == 5 and sleeps == [1.0, 2.0, 4.0, 8.0]
    assert c.usage_today()["cost_usd"] == 0 and c.cache.data == {}


def test_5xx_retried_then_ok():
    f, t = mk(lambda req, n: httpx.Response(503) if n == 1 else httpx.Response(200, json=[1, 2]))
    c = CachedClient("x", MemoryCache(), limiter(), transport=t, sleep=lambda s: None)
    assert c.get("https://x.test/a") == [1, 2]
    assert c.get("https://x.test/a") == [1, 2]  # list payloads survive the cache
    assert len(f.requests) == 2


def test_work_404_is_none(settings):
    f, c = oa_client(settings, lambda req, n: httpx.Response(404, json={}))
    assert c.work("doi:10.1/nope") is None
    assert f.requests[0].url.path == "/works/doi:10.1/nope"


def test_work_arxiv_path(settings):
    f, c = oa_client(settings, lambda req, n: httpx.Response(200, json=OA["results"][1]))
    c.work("arXiv:2302.09108v2")
    assert f.requests[0].url.path == "/works/doi:10.48550/arxiv.2302.09108"


def test_s2_header_only_with_key(settings):
    f, t = mk(lambda req, n: httpx.Response(200, json=json.loads((FIX / "s2_search.json").read_text())))
    c = S2Client(settings, MemoryCache(), transport=t, limiter=limiter())
    hits = c.search("vit", 2)
    assert "x-api-key" not in f.requests[0].headers
    assert S2Client(settings, MemoryCache()).http.limiter.rate == pytest.approx(1 / 3)
    assert hits[0].doi == "10.48550/arxiv.2010.11929" and hits[0].arxiv_id == "2010.11929"
    assert hits[0].source == "s2" and hits[1].doi is None
    f, t = mk(lambda req, n: httpx.Response(200, json={"data": []}))
    c = S2Client(
        settings.model_copy(update={"s2_api_key": "S"}), MemoryCache(), transport=t, limiter=limiter()
    )
    c.search("vit", 2)
    assert f.requests[0].headers["x-api-key"] == "S"
    assert S2Client(settings.model_copy(update={"s2_api_key": "S"}), MemoryCache()).http.limiter.rate == 1.0


def test_s2_batch_and_edges(settings):
    def h(req, n):
        if req.method == "POST":
            assert json.loads(req.content) == {"ids": ["a", "b"]}
            return httpx.Response(200, json=[{"paperId": "a", "title": "A"}, None])
        return httpx.Response(200, json={"data": [{"citingPaper": {"paperId": "z", "title": "Z"}}]})

    f, t = mk(h)
    c = S2Client(settings, MemoryCache(), transport=t, limiter=limiter())
    assert [x.source_id for x in c.batch(["a", "b"])] == ["a"]
    assert [x.source_id for x in c.citations("a")] == ["z"]


def test_arxiv_parse_and_normalise(settings):
    hits = parse_atom((FIX / "arxiv_search.xml").read_text())
    assert [h.arxiv_id for h in hits] == ["2206.10552", "2108.01390"]
    assert all(h.source == "arxiv" and h.title and "\n" not in h.title and h.year for h in hits)
    for raw, want in {
        "arXiv:2010.11929v2": "2010.11929",
        "https://arxiv.org/pdf/2010.11929v3.pdf": "2010.11929",
        "http://arxiv.org/abs/2010.11929": "2010.11929",
        "hep-th/9901001v1": "hep-th/9901001",
        "2010.11929": "2010.11929",
        "nonsense": None,
    }.items():
        assert normalise_arxiv_id(raw) == want
    f, t = mk(lambda req, n: httpx.Response(200, text=(FIX / "arxiv_search.xml").read_text()))
    c = ArxivClient(settings, MemoryCache(), transport=t, limiter=limiter())
    assert len(c.search("vision transformer", 2)) == 2
    assert f.requests[0].url.params["search_query"] == "all:vision transformer"
    assert ArxivClient(settings, MemoryCache()).http.limiter.rate == pytest.approx(1 / 3)
    c.search("vision transformer", 2)
    assert len(f.requests) == 1


def test_arxiv_download(settings, tmp_path):
    f, t = mk(lambda req, n: httpx.Response(200, content=b"%PDF-1"))
    c = ArxivClient(settings, MemoryCache(), transport=t, limiter=limiter())
    p = c.download_pdf("arXiv:2010.11929v2", tmp_path)
    assert p.read_bytes() == b"%PDF-1" and p.name == "2010.11929.pdf"
    c.download_pdf("2010.11929", tmp_path)
    assert len(f.requests) == 1


def test_resolve_hits_order_dedupe_drop(settings):
    recs = {
        "/works/doi:10.1/a": {"id": "https://openalex.org/W1", "title": "A", "doi": "https://doi.org/10.1/a"},
        "/works/doi:10.48550/arxiv.2001.00002": {"id": "https://openalex.org/W2", "title": "B"},
    }

    def h(req, n):
        if req.url.path in recs:
            return httpx.Response(200, json=recs[req.url.path])
        if req.url.path == "/works":
            return httpx.Response(200, json={"results": [{"id": "https://openalex.org/W9", "title": "N"}]})
        return httpx.Response(404, json={})

    f, c = oa_client(settings, h)
    hits = [
        SearchHit(source="s2", source_id="s1", title="A", doi="10.1/A"),
        SearchHit(source="arxiv", source_id="2001.00002", title="B", arxiv_id="2001.00002"),
        SearchHit(source="s2", source_id="s3", title="A again", doi="10.1/a", arxiv_id="2001.00002"),
        SearchHit(source="s2", source_id="s4", title="Unknown", doi="10.9/zzz"),
        SearchHit(source="openalex", source_id="W9", title="N"),
    ]
    out = resolve_hits(hits, c)
    # hit order is kept (W1 from s1, W2 from the arXiv hit, s3 dedupes, s4 dropped, then the OpenAlex hit)
    assert [r.paper_id for r in out] == ["W1", "W2", "W9"]
    assert next(r for r in out if r.paper_id == "W1").s2_id == "s1"


def test_make_clients_memory(settings):
    c = make_clients(settings=settings)
    assert c.openalex and c.s2 and c.arxiv


@pytest.mark.live
def test_live_openalex_lookup(settings):
    c = make_clients(settings=settings)
    r = c.openalex.work("W2626778328")
    assert r and r.title.startswith("Attention") and c.openalex.http.spent_today() == 0


@pytest.mark.live
def test_live_arxiv_query(settings):
    hits = make_clients(settings=settings).arxiv.search("vision transformer", 2)
    assert hits and hits[0].arxiv_id
