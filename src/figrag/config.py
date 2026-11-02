import csv
import hashlib
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.toml"

DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
CACHE_DIR = DATA_DIR / "cache"
PAPERS_CSV = DATA_DIR / "papers.csv"
PAGES_PATH = DATA_DIR / "pages.jsonl"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
QUESTIONS_PATH = ROOT / "eval" / "questions.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def stable_hash(*parts, n: int = 8) -> str:
    h = hashlib.sha256()
    for part in parts:
        if not isinstance(part, str):
            part = json.dumps(part, sort_keys=True)
        h.update(part.encode())
        h.update(b"\x00")
    return h.hexdigest()[:n]


def index_hash(cfg: dict) -> str:
    """Identifies the chunk table contents: pinned corpus + chunking + embedding config."""
    with open(PAPERS_CSV, newline="") as f:
        corpus = sorted(row["arxiv_id"] for row in csv.DictReader(f))
    return stable_hash(corpus, cfg["chunking"], cfg["embedding"])


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
