import hashlib
import json
from pathlib import Path


def stable_hash(*parts, n: int = 8) -> str:
    """Short, order-sensitive hash of strings and JSON-serialisable values."""
    h = hashlib.sha256()
    for part in parts:
        if not isinstance(part, str):
            part = json.dumps(part, sort_keys=True, default=str)
        h.update(part.encode())
        h.update(b"\x00")
    return h.hexdigest()[:n]


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")
