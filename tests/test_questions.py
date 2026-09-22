import pytest

from figrag.config import PAGES_PATH, QUESTIONS_PATH, read_jsonl
from figrag.evaluate import figure_split_violations, validate_questions

questions = read_jsonl(QUESTIONS_PATH) if QUESTIONS_PATH.exists() else []
pytestmark = pytest.mark.skipif(not questions, reason="eval/questions.jsonl is empty")


def test_questions_are_valid():
    assert validate_questions(questions) == []


@pytest.mark.skipif(not PAGES_PATH.exists(), reason="run `figrag ingest` first")
def test_figure_split_is_figure_only():
    assert figure_split_violations(questions, read_jsonl(PAGES_PATH)) == []
