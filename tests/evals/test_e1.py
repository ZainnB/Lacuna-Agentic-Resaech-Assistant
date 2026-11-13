import json
from datetime import date

from lacuna.evals.e1 import Survey, load_dataset, recall_at, render_markdown, run_e1
from lacuna.models import PaperRecord


def test_recall_at_counts_hits_within_k():
    gold = {"a", "b", "c", "d"}
    m = recall_at(["a", "x", "b", "y", "c"], gold, ks=(2, 5))
    assert m == {"recall@2": 0.25, "recall@5": 0.75, "retrieved": 5, "gold": 4}


class KeywordOnly:
    """Search returns gold items; the snowball finds nothing extra."""

    def search_works(self, query, *, from_date=None, to_date=None, max_results=50):
        self.to_date = to_date
        return [
            PaperRecord(paper_id=p, title=p, year=2020, publication_date=date(2020, 1, 1))
            for p in ("SURVEY", "a", "b")
        ]

    def works_by_ids(self, ids):
        return []

    def cited_by(self, work_id, max_results=50):
        return []


def test_run_e1_excludes_survey_and_applies_freeze():
    client = KeywordOnly()
    s = Survey("SURVEY", "t", "q", date(2021, 1, 1), {"a", "b", "c", "d"})
    result = run_e1(client, surveys=[s], ks=(50,))
    assert client.to_date == date(2021, 1, 1)
    row = result["per_survey"][0]
    assert row["keyword"]["recall@50"] == 0.5 and row["snowball"]["recall@50"] == 0.5
    assert result["summary"]["keyword"]["recall@50"] == 0.5
    assert "SURVEY" not in json.dumps(row["keyword"])
    assert "| keyword |" in render_markdown(result)


def test_load_dataset_shape(tmp_path):
    (tmp_path / "refs").mkdir()
    (tmp_path / "surveys.json").write_text(json.dumps([{"openalex_id": "W1", "title": "T", "query": "Q"}]))
    (tmp_path / "freeze.json").write_text(json.dumps({"W1": "2025-03-01"}))
    (tmp_path / "refs" / "W1.json").write_text(json.dumps([{"id": "W2"}, {"id": "W3"}]))
    [s] = load_dataset(tmp_path)
    assert s.freeze == date(2025, 3, 1) and s.gold == {"W2", "W3"}
