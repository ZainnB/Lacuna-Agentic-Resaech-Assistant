"""Paths and the frozen config for the deep-read module.

Data files stay where FAMM-RAG kept them (data/); the config lives in config/deepread.toml, the
eval questions in evals/datasets/e6/ and results in evals/results/.
"""

import csv
import tomllib
from pathlib import Path

from lacuna.settings import ROOT
from lacuna.util import read_jsonl, stable_hash, write_jsonl

CONFIG_PATH = ROOT / "config" / "deepread.toml"

DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
CACHE_DIR = DATA_DIR / "cache"
PAPERS_CSV = DATA_DIR / "papers.csv"
PAGES_PATH = DATA_DIR / "pages.jsonl"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
QUESTIONS_PATH = ROOT / "evals" / "datasets" / "e6" / "questions.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"

__all__ = [
    "CACHE_DIR",
    "CHUNKS_PATH",
    "CONFIG_PATH",
    "DATA_DIR",
    "PAGES_PATH",
    "PAPERS_CSV",
    "PDF_DIR",
    "QUESTIONS_PATH",
    "RESULTS_DIR",
    "ROOT",
    "index_hash",
    "load_config",
    "read_jsonl",
    "stable_hash",
    "write_jsonl",
]


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def index_hash(cfg: dict) -> str:
    """Identifies the chunk table contents: pinned corpus + chunking + embedding config."""
    with open(PAPERS_CSV, newline="") as f:
        corpus = sorted(row["arxiv_id"] for row in csv.DictReader(f))
    return stable_hash(corpus, cfg["chunking"], cfg["embedding"])
