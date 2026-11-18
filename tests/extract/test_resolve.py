import pytest

from lacuna.extract.resolve import Resolver, entity_id, load_aliases, normalise, split_variant


@pytest.mark.parametrize(
    ("raw", "norm"),
    [
        ("ILSVRC-2012", "ilsvrc 2012"),
        ("  ImageNet-1k ", "imagenet 1k"),
        ("Top-1  Acc.", "top 1 acc"),
        ("mAP@0.5", "map 0.5"),
        ("Fréchet", "fréchet"),
    ],
)
def test_normalise(raw, norm):
    assert normalise(raw) == norm


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ViT-B/16", ("ViT", "B/16")),
        ("ViT-L/14@336", ("ViT", "L/14@336")),
        ("ViT-Base/16", ("ViT", "Base/16")),
        ("Swin-T", ("Swin", "T")),
        ("BERT-Large", ("BERT", "Large")),
        ("ResNet-50", ("ResNet", "50")),
        ("ResNet101", ("ResNet", "101")),
        ("VGG-16", ("VGG", "16")),
        ("ViT", ("ViT", None)),
        ("GPT-3", ("GPT-3", None)),
        ("Faster R-CNN", ("Faster R-CNN", None)),
        ("Vision Transformer", ("Vision Transformer", None)),
    ],
)
def test_split_variant(raw, expected):
    assert split_variant(raw) == expected


def test_alias_file_loads_and_has_required_merges():
    a = load_aliases()
    assert len(a) > 100
    assert a[("dataset", normalise("ILSVRC 2012"))] == "ImageNet-1k"
    assert a[("dataset", normalise("ILSVRC-2012"))] == "ImageNet-1k"
    assert a[("dataset", "imagenet")] == "ImageNet-1k"
    assert a[("dataset", normalise("Microsoft COCO"))] == "COCO"
    assert a[("metric", normalise("Top-1 Acc"))] == "top-1 accuracy"
    assert a[("metric", normalise("mean average precision"))] == "mAP"
    assert a[("method", normalise("Residual Network"))] == "ResNet"


@pytest.fixture
def resolver(conn, embedder):
    return Resolver(conn, embedder=embedder)


def test_variant_is_a_property_and_family_merges(resolver):
    a = resolver.resolve("method", "ViT-B/16")
    b = resolver.resolve("method", "Vision Transformer")
    c = resolver.resolve("method", "ViT-L/16")
    assert a == ("method:vit", "ViT", "B/16")
    assert b == ("method:vit", "ViT", None)
    assert c[0] == "method:vit" and c[2] == "L/16"


def test_explicit_variant_wins_and_is_kept(resolver):
    assert resolver.resolve("method", "ViT", "H/14") == ("method:vit", "ViT", "H/14")
    assert resolver.resolve("method", "ViT-B/16", "B/32")[2] == "B/32"


def test_imagenet_aliases_merge(resolver):
    ids = {
        resolver.resolve("dataset", n)[0] for n in ["ILSVRC-2012", "ImageNet", "ImageNet-1k", "ILSVRC 2012"]
    }
    assert ids == {"dataset:imagenet-1k"}
    assert resolver.resolve("dataset", "ILSVRC-2012")[1] == "ImageNet-1k"


def test_dataset_names_are_not_variant_split(resolver):
    assert resolver.resolve("dataset", "CIFAR-10")[0] == "dataset:cifar-10"


def test_type_scoping(resolver):
    assert resolver.resolve("task", "image classification")[0] == "task:image-classification"
    assert resolver.resolve("metric", "accuracy")[0] == "metric:accuracy"


def test_exact_match_on_stored_entity_and_alias(conn, resolver):
    first = resolver.resolve("method", "Foo Net")
    assert first[0] == "method:foo-net" and resolver.stats["created"] == 1
    fresh = Resolver(conn, embedder=resolver._embedder)  # no memo
    assert fresh.resolve("method", "foo-net")[0] == "method:foo-net"
    assert fresh.stats["exact"] == 1 and fresh.stats["created"] == 0


def test_similarity_merge_above_threshold(conn, make_embedder):
    emb = make_embedder(near={"flashattn": "flash attention"})
    r = Resolver(conn, threshold=0.9, embedder=emb)
    base = r.resolve("method", "Flash Attention")
    near = r.resolve("method", "FlashAttn")
    assert near[0] == base[0] == "method:flash-attention"
    assert r.stats["similar"] == 1 and r.stats["created"] == 1
    assert "flashattn" in conn.entities["method:flash-attention"]["aliases"]


def test_dissimilar_name_creates_new_entity(conn, make_embedder):
    r = Resolver(conn, threshold=0.9, embedder=make_embedder())
    a = r.resolve("method", "Flash Attention")
    b = r.resolve("method", "Mamba Blocks")
    assert a[0] != b[0] and r.stats["created"] == 2 and r.stats["similar"] == 0


def test_similarity_only_within_type(conn, make_embedder):
    emb = make_embedder(near={"flashattn": "flash attention"})
    r = Resolver(conn, embedder=emb)
    r.resolve("dataset", "Flash Attention")
    assert r.resolve("method", "FlashAttn")[0] == "method:flashattn"


def test_threshold_is_configurable(conn, make_embedder):
    emb = make_embedder(near={"flashattn": "flash attention"})
    r = Resolver(conn, threshold=0.9999, embedder=emb)
    r.resolve("method", "Flash Attention")
    assert r.resolve("method", "FlashAttn")[0] == "method:flashattn"


def test_empty_and_unknown_type(resolver):
    with pytest.raises(ValueError):
        resolver.resolve("method", "  --  ")
    with pytest.raises(ValueError):
        resolver.resolve("gadget", "x")


def test_entity_id_format():
    assert entity_id("dataset", "ImageNet-1k") == "dataset:imagenet-1k"
