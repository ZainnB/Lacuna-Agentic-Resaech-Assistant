"""Scout ranking and leakage rules against a fake OpenAlex client. No network."""

from datetime import date

from lacuna.models import PaperRecord
from lacuna.scout.core import ScoutConfig, candidate_score, in_window, scout


def rec(pid, year=2023, cites=10, refs=(), pub=None):
    return PaperRecord(
        paper_id=pid,
        title=pid,
        year=year,
        citation_count=cites,
        referenced_ids=list(refs),
        publication_date=pub or date(year, 6, 1),
    )


class FakeOpenAlex:
    def __init__(self, search, by_id, cited_by):
        self.search_results, self.by_id, self.cited = search, by_id, cited_by
        self.calls = []

    def search_works(self, query, *, from_date=None, to_date=None, max_results=50):
        self.calls.append(("search", query, to_date, max_results))
        return self.search_results[:max_results]

    def works_by_ids(self, ids):
        self.calls.append(("by_ids", tuple(ids)))
        return [self.by_id[i] for i in ids if i in self.by_id]

    def cited_by(self, work_id, max_results=50):
        self.calls.append(("cited_by", work_id))
        return self.cited.get(work_id, [])


def make_client():
    seeds = [rec("S1", refs=["R1", "R2"]), rec("S2", refs=["R1", "R3"]), rec("S3", refs=["R1"])]
    by_id = {"R1": rec("R1", 2019, 500), "R2": rec("R2", 2020, 50), "R3": rec("R3", 2021, 5)}
    cited = {"S1": [rec("C1", 2024, 3), rec("S2")], "S2": [rec("C1", 2024, 3)], "S3": [rec("LATE", 2026, 0)]}
    return FakeOpenAlex(seeds, by_id, cited)


def test_seeds_first_then_most_linked_candidates():
    result = scout("q", make_client(), ScoutConfig(max_papers=10, seeds=3), today=date(2025, 1, 1))
    assert result.status == "ok"
    ids = result.ids()
    assert ids[:3] == ["S1", "S2", "S3"]
    assert ids[3] == "R1"  # cited by all three seeds
    roles = {r.record.paper_id: r.role for r in result.papers}
    assert roles["R1"] == "snowball_ref" and roles["C1"] == "snowball_cite"
    assert [r.rank for r in result.papers] == list(range(1, len(ids) + 1))


def test_seen_papers_are_not_recounted_as_candidates():
    result = scout("q", make_client(), ScoutConfig(max_papers=10, seeds=3), today=date(2025, 1, 1))
    assert result.ids().count("S2") == 1


def test_freeze_date_drops_later_papers_and_exclude_drops_ids():
    cfg = ScoutConfig(max_papers=10, seeds=3, to_date=date(2024, 12, 31))
    result = scout("q", make_client(), cfg, exclude={"R2"}, today=date(2025, 1, 1))
    assert "LATE" not in result.ids() and "R2" not in result.ids()


def test_depth_zero_is_keyword_only_and_caps_apply():
    client = make_client()
    result = scout("q", client, ScoutConfig(max_papers=2, seeds=3, depth=0))
    assert result.ids() == ["S1", "S2"]
    assert [c[0] for c in client.calls] == ["search"]


def test_no_results_status():
    client = FakeOpenAlex([], {}, {})
    assert scout("q", client, ScoutConfig()).status == "no_results"


def test_in_window_requires_a_date_when_frozen():
    undated = PaperRecord(paper_id="U", title="u")
    assert not in_window(undated, ScoutConfig(to_date=date(2024, 1, 1)))
    assert in_window(rec("A", 2020), ScoutConfig(to_date=date(2024, 1, 1)))


def test_more_seed_links_beats_citations():
    today = date(2025, 1, 1)
    assert candidate_score(rec("a", 2020, 5), 3, today) > candidate_score(rec("b", 2020, 50000), 1, today)
