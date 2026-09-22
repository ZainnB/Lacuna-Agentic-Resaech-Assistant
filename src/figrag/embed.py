"""bge-base embeddings. The query instruction prefix goes on queries only."""
from functools import lru_cache

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from figrag.config import CACHE_DIR, stable_hash


def resolve_device(device: str) -> str:
    if device == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device


@lru_cache(maxsize=1)
def get_model(name: str, device: str) -> SentenceTransformer:
    return SentenceTransformer(name, device=resolve_device(device))


def embed_chunks(chunks: list[dict], cfg: dict) -> np.ndarray:
    """Embed chunk texts, cached on disk per (embedding config, chunk contents)."""
    key = stable_hash(cfg, [(c["id"], c["text"]) for c in chunks], n=16)
    path = CACHE_DIR / f"emb_{key}.npy"
    if path.exists():
        return np.load(path)
    model = get_model(cfg["model"], cfg["device"])
    vectors = model.encode(
        [c["text"] for c in chunks],
        batch_size=cfg["batch_size"],
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, vectors)
    return vectors


def embed_queries(queries: list[str], cfg: dict) -> np.ndarray:
    model = get_model(cfg["model"], cfg["device"])
    return model.encode(
        [cfg["query_prefix"] + q for q in queries],
        batch_size=cfg["batch_size"],
        normalize_embeddings=True,
    ).astype(np.float32)
