from figrag.chunk import heading_offsets, windows
from figrag.evaluate import contains_span, is_hit, retrieval_metrics, score_answer, validate_questions
from figrag.ingest import strip_references


def q(**kw):
    base = {"id": "q1", "question": "?", "answer": "71.4", "answer_type": "numeric", "split": "table",
            "paper_id": "p", "page": 7, "evidence_ref": "Table 3", "answer_span": "71.4"}
    return {**base, **kw}


def test_contains_span_respects_number_boundaries():
    assert contains_span("top-1 of 71.4 on K400", "71.4")
    assert not contains_span("top-1 of 171.45", "71.4")
    assert not contains_span("see Table 31", "Table 3")
    assert contains_span("Table 3: ablation", "Table 3")
    assert contains_span("recog-\nnition rate", "recognition")


def test_numeric_scoring_uses_first_number():
    assert score_answer("71.4%", q()) == (True, "numeric_first")
    assert score_answer("71.40", q())[0]
    assert not score_answer("69.2", q())[0]
    assert score_answer("not found", q()) == (False, "abstained")
    assert score_answer(None, q()) == (False, "no_answer")


def test_span_and_unanswerable_scoring():
    span = q(answer_type="short_span", answer="Layer normalization")
    assert score_answer("layer normalization.", span) == (True, "exact")
    assert score_answer("They use layer normalization", span) == (True, "contains")
    unans = q(answer_type="unanswerable", split="unanswerable", answer="not found")
    assert score_answer("Not found.", unans)[0]
    assert not score_answer("42", unans)[0]
    assert score_answer("a paragraph", q(answer_type="free_text")) == (None, "llm_judge")


def test_hit_by_span_or_page_and_ref():
    question = q(answer_span="85.3")
    assert is_hit({"paper_id": "p", "page": 2, "text": "we get 85.3 top-1"}, question)
    assert is_hit({"paper_id": "p", "page": 7, "text": "Table 3: garbled cells"}, question)
    assert not is_hit({"paper_id": "p", "page": 8, "text": "Table 3: garbled cells"}, question)
    assert not is_hit({"paper_id": "other", "page": 7, "text": "85.3 Table 3"}, question)


def test_retrieval_metrics():
    question = q(answer_span="85.3")
    chunks = [{"paper_id": "x", "page": 1, "text": ""}] * 5 + [{"paper_id": "p", "page": 1, "text": "85.3"}]
    m = retrieval_metrics(chunks, question, k=5)
    assert m["recall_at_k"] == 0.0 and m["rr"] == 1 / 6


def test_validate_questions():
    assert validate_questions([q()]) == []
    errs = validate_questions([q(), q(split="figure", answer_type="unanswerable")])
    assert any("duplicate" in e for e in errs)
    assert any("unanswerable" in e for e in errs)


def test_windows_overlap():
    offsets = [(i, i + 1) for i in range(1000)]
    spans = list(windows(offsets, 400, 80))
    assert spans[0] == (0, 400) and spans[1] == (320, 720) and spans[-1][1] == 1000


def test_heading_detection_skips_table_rows():
    text = "3 Method\nsome text\n12 ViT-L/16 85.3\n4.2 Ablation Studies\n"
    assert [t for _, t in heading_offsets(text)] == ["3 Method", "4.2 Ablation Studies"]


def test_strip_references_keeps_appendix():
    lines = [(1, "Intro"), (8, "References"), (8, "[1] A. Dosovitskiy, L. Beyer"), (9, "A Additional Results"), (9, "x")]
    assert [l for _, l in strip_references(lines)] == ["Intro", "A Additional Results", "x"]
