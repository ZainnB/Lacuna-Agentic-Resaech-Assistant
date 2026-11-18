"""Entity resolution: normalise -> variant split -> alias table -> exact match -> bge similarity -> create.

Entity ids are "<type>:<normalised canonical name, spaces as hyphens>". The similarity threshold is
a constructor argument and is UNTUNED until E3 gold labels exist (see docs/decisions.md).
"""

from __future__ import annotations

import re
import tomllib
import unicodedata
from collections.abc import Callable

from lacuna.settings import ROOT

ALIASES_PATH = ROOT / "config" / "aliases.toml"
ENTITY_TYPES = ("method", "dataset", "task", "metric")
DEFAULT_THRESHOLD = 0.90

_SIZE = r"(?:Ti|T|S|B|L|H|g|G|Tiny|Small|Base|Large|Huge)"
_PATCH = re.compile(rf"^(?P<base>.+?)[-\s](?P<var>{_SIZE}/\d+(?:@\d+)?)$")
_SIZE_ONLY = re.compile(rf"^(?P<base>[A-Za-z][A-Za-z0-9]*)[-\s](?P<var>{_SIZE}|XL)$")
_DEPTH = re.compile(r"^(?P<base>[A-Za-z][A-Za-z]+?)[-\s]?(?P<var>\d{2,3})$")


def normalise(name: str) -> str:
    """Lowercase, NFKC, every run of punctuation/space becomes one space. 'ILSVRC-2012' -> 'ilsvrc 2012'."""
    s = unicodedata.normalize("NFKC", name).casefold()
    s = re.sub(r"[^\w+.]+|_", " ", s)
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def slug(norm: str) -> str:
    return norm.replace(" ", "-")


def entity_id(type_: str, canonical: str) -> str:
    return f"{type_}:{slug(normalise(canonical))}"


def split_variant(name: str) -> tuple[str, str | None]:
    """Split a model name into (family, variant): 'ViT-B/16' -> ('ViT', 'B/16').

    Applies to methods only; the caller decides. Names without a recognised size, patch or depth
    suffix come back unchanged with variant None.
    """
    n = name.strip()
    for pat in (_PATCH, _SIZE_ONLY, _DEPTH):
        m = pat.match(n)
        if m:
            return m.group("base").strip(), m.group("var")
    return n, None


def load_aliases(path=ALIASES_PATH) -> tuple[dict, dict]:
    """(alias_map {(type, normalised alias): canonical}, canonical_names {(type, norm): canonical})."""
    with open(path, "rb") as f:
        entries = tomllib.load(f).get("entity", [])
    amap: dict[tuple[str, str], str] = {}
    for e in entries:
        canon = e["canonical"]
        for a in [canon, *e.get("aliases", [])]:
            amap[(e["type"], normalise(a))] = canon
    return amap


def _vec(v) -> str:
    return "[" + ",".join(f"{float(x):.6f}" for x in v) + "]"


def default_embedder() -> Callable[[list[str]], list]:
    """bge-base via deepread's cached model loader (device and model from config/deepread.toml)."""
    from lacuna.deepread.config import load_config
    from lacuna.deepread.embed import get_model

    cfg = load_config()["embedding"]

    def embed(texts: list[str]):
        model = get_model(cfg["model"], cfg["device"])
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    return embed


class Resolver:
    """Resolves surface names to canonical entities, persisting new ones to `entities`."""

    def __init__(
        self,
        conn,
        threshold: float = DEFAULT_THRESHOLD,
        *,
        embedder: Callable[[list[str]], list] | None = None,
        aliases: dict | None = None,
    ) -> None:
        self.conn = conn
        self.threshold = threshold
        self._embedder = embedder
        self.aliases = aliases if aliases is not None else load_aliases()
        self._memo: dict[tuple[str, str], tuple[str, str]] = {}
        self.stats = {"alias": 0, "exact": 0, "similar": 0, "created": 0}

    def embed(self, text: str):
        if self._embedder is None:
            self._embedder = default_embedder()
        return self._embedder([text])[0]

    def resolve(self, type_: str, name: str, variant: str | None = None) -> tuple[str, str, str | None]:
        """-> (entity_id, canonical_name, variant). An explicit `variant` beats one split from the name."""
        if type_ not in ENTITY_TYPES:
            raise ValueError(f"unknown entity type {type_!r}")
        base, var = split_variant(name) if type_ == "method" else (name.strip(), None)
        variant = (variant.strip() if variant else None) or var
        norm = normalise(base)
        if not norm:
            raise ValueError(f"empty entity name {name!r}")
        key = (type_, norm)
        if key not in self._memo:
            self._memo[key] = self._resolve_base(type_, base, norm)
        eid, canon = self._memo[key]
        return eid, canon, variant

    # ------------------------------------------------------------ steps

    def _resolve_base(self, type_: str, surface: str, norm: str) -> tuple[str, str]:
        canon = self.aliases.get((type_, norm))
        if canon is not None:
            self.stats["alias"] += 1
            return self._ensure(type_, canon, norm if normalise(canon) != norm else None)
        eid = f"{type_}:{slug(norm)}"
        row = self.conn.execute(
            "/* entity_exact */ SELECT entity_id, canonical_name FROM entities "
            "WHERE type = %s AND (entity_id = %s OR %s = ANY(aliases))",
            (type_, eid, norm),
        ).fetchone()
        if row:
            self.stats["exact"] += 1
            return row["entity_id"], row["canonical_name"]
        vec = self.embed(norm)
        near = self.conn.execute(
            "/* entity_similar */ SELECT entity_id, canonical_name, 1 - (embedding <=> %s::vector) AS sim "
            "FROM entities WHERE type = %s AND embedding IS NOT NULL "
            "ORDER BY embedding <=> %s::vector LIMIT 1",
            (_vec(vec), type_, _vec(vec)),
        ).fetchone()
        if near and near["sim"] >= self.threshold:
            self.stats["similar"] += 1
            self.conn.execute(
                "/* entity_add_alias */ UPDATE entities SET aliases = array_append(aliases, %s) "
                "WHERE entity_id = %s AND NOT (%s = ANY(aliases))",
                (norm, near["entity_id"], norm),
            )
            return near["entity_id"], near["canonical_name"]
        self.stats["created"] += 1
        return self._ensure(type_, surface, None, vec=vec)

    def _ensure(self, type_: str, canonical: str, alias_norm: str | None, *, vec=None) -> tuple[str, str]:
        """Upsert the canonical entity row (embedding computed on first insert); add an alias if given."""
        eid = entity_id(type_, canonical)
        row = self.conn.execute(
            "/* entity_get */ SELECT entity_id FROM entities WHERE entity_id = %s", (eid,)
        ).fetchone()
        if row is None:
            if vec is None:
                vec = self.embed(normalise(canonical))
            self.conn.execute(
                "/* entity_insert */ INSERT INTO entities "
                "(entity_id, type, canonical_name, aliases, embedding) "
                "VALUES (%s, %s, %s, %s, %s::vector) ON CONFLICT (entity_id) DO NOTHING",
                (eid, type_, canonical, [alias_norm] if alias_norm else [], _vec(vec)),
            )
        elif alias_norm:
            self.conn.execute(
                "/* entity_add_alias */ UPDATE entities SET aliases = array_append(aliases, %s) "
                "WHERE entity_id = %s AND NOT (%s = ANY(aliases))",
                (alias_norm, eid, alias_norm),
            )
        return eid, canonical
