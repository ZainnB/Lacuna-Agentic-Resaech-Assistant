import json
from types import SimpleNamespace as NS

from lacuna.evals import e3


class FakeConn:
    def __init__(self, rows=None, usage=None):
        self.rows, self.usage = rows or [], usage or []

    def execute(self, sql, params=()):
        data = self.usage if "provider_usage" in sql else self.rows
        return NS(fetchall=lambda: data)


def ent(**kw):
    return NS(**{t: [NS(name=n) for n in kw.get(t, [])] for t in e3.ENTITY_TYPES})


def label(dataset, pid, **kw):
    doc = {"paper_id": pid, "title": pid, "labeller": "me", "labelled_on": "2026-11-20", "scope": "abstract"}
    doc |= {t: kw.get(t, []) for t in e3.ENTITY_TYPES}
    (dataset / "labels").mkdir(parents=True, exist_ok=True)
    (dataset / "labels" / f"{pid}.json").write_text(json.dumps(doc))


def test_score_math():
    s = e3.score({"methods": {"A", "B", "C"}}, {"methods": {"a", "b", "x"}})
    m = s["methods"]
    assert (m["tp"], m["fp"], m["fn"]) == (2, 1, 1)
    assert m["precision"] == m["recall"] == m["f1"] == round(2 / 3, 4)
    assert s["tasks"]["f1"] == 0.0
    assert s["micro"]["tp"] == 2
    assert s["macro"]["f1"] == m["f1"]  # empty types are skipped


def test_score_empty_edges():
    s = e3.score({}, {})
    assert s["micro"]["f1"] == 0.0 and s["macro"]["f1"] == 0.0
    s = e3.score({"methods": {"A"}}, {})
    assert s["methods"]["recall"] == 0.0 and s["methods"]["precision"] == 0.0 and s["methods"]["fn"] == 1
    s = e3.score({}, {"methods": {"A"}})
    assert s["methods"]["fp"] == 1 and s["methods"]["f1"] == 0.0


def test_default_normalise():
    n = e3.default_normalise
    assert n("ViT-B/16") == n("ViT") == "vit"
    assert n("ResNet-50") == "resnet"
    assert n("ImageNet-1k") == "imagenet"
    assert n(" Top-1  Accuracy ") == "top1accuracy"
    assert n("3D") == "3d"


def test_split_deterministic_and_balanced():
    ids = [f"W{i}" for i in range(30)]
    a = e3.assign_splits(ids)
    assert a == e3.assign_splits(list(reversed(ids)))
    counts = [list(a.values()).count(s) for s in e3.SPLITS]
    assert counts == [15, 15]
    more = e3.assign_splits(ids + ["W99"], a)
    assert all(more[i] == a[i] for i in ids)
    assert e3.split_for("W1") == e3.split_for("W1")


def test_template_has_no_entities(tmp_path):
    rows = [{"paper_id": f"W{i}", "title": f"T{i}", "abstract": "We use ViT on ImageNet."} for i in range(6)]
    written, split = e3.write_template(FakeConn(rows), "t_x", 6, tmp_path)
    assert len(written) == 6 and json.loads((tmp_path / "split.json").read_text()) == split
    for p in written:
        d = json.loads(p.read_text())
        assert all(d[t] == [] for t in e3.ENTITY_TYPES)
        assert d["split"] == split[d["paper_id"]] and d["labeller"] == ""
    # Existing work is not overwritten, and unfinished templates are not loaded as labels.
    assert e3.write_template(FakeConn(rows), "t_x", 6, tmp_path)[0] == []
    assert e3.load_labels(tmp_path) == []


def test_run_e3_with_fake_extractor(tmp_path):
    ids = [f"W{i}" for i in range(4)]
    split = e3.assign_splits(ids)
    (tmp_path / "split.json").write_text(json.dumps(split))
    for pid in ids:
        label(tmp_path, pid, methods=["ViT"], datasets=["ImageNet"])
    seen = []

    def fake(conn, pid, *, chain=None):
        seen.append((pid, chain))
        if pid == "W0":
            raise RuntimeError("boom")
        return NS(extraction=ent(methods=["ViT-B/16"], datasets=["CIFAR"]))

    usage = [{"provider": "groq", "model": "m", "calls": 10, "schema_failures": 1}]
    r = e3.run_e3(FakeConn(usage=usage), dataset_dir=tmp_path, extract=fake, split="all", chain=["x/y"])
    assert r["n_scored"] == 4 and sorted(p for p, _ in seen) == ids and seen[0][1] == ["x/y"]
    assert r["scores"]["methods"]["tp"] == 3 and r["scores"]["methods"]["fn"] == 1
    assert r["scores"]["datasets"]["fp"] == 3 and len(r["failed"]) == 1
    assert r["schema_validity_today"][0]["validity"] == 0.9
    rep = e3.run_e3(FakeConn(), dataset_dir=tmp_path, extract=fake)
    assert rep["n_scored"] == sum(1 for v in split.values() if v == "reporting")
    paths = e3.write_results(r, tmp_path / "results")
    assert all(p.exists() for p in paths) and "methods" in paths[1].read_text()


def test_run_e3_no_labels(tmp_path):
    def boom(*a, **k):
        raise AssertionError("must not extract")

    r = e3.run_e3(FakeConn(), dataset_dir=tmp_path, extract=boom)
    assert r["scores"] is None and r["label_counts"]["total"] == 0
    assert "no scores" in e3.render_markdown(r).lower()


def test_canonical_normalise_matches_variants_and_aliases_by_type():
    n = e3.get_normalise()
    assert n("ViT-B/16", "methods") == n("Vision Transformer", "methods") == n("ViT", "methods")
    assert n("ILSVRC-2012", "datasets") == n("ImageNet", "datasets")
    # Datasets are never variant-split: CIFAR-100 and CIFAR-10 stay distinct.
    assert n("CIFAR-100", "datasets") != n("CIFAR-10", "datasets")
