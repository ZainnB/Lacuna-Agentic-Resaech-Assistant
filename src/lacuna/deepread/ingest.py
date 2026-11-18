"""Download the pinned arXiv PDFs and extract text per page with PyMuPDF.

Tables are left exactly as PyMuPDF extracts them: the messy text is the
realistic baseline.
"""

import csv
import re
import time

import httpx
import pymupdf

from lacuna.deepread.config import PAGES_PATH, PAPERS_CSV, PDF_DIR, write_jsonl

ABS_VERSION = re.compile(r"\[v(\d+)\]")
USER_AGENT = "lacuna-deepread/0.1"
RETRY_STATUS = {406, 429, 500, 502, 503}
VERSIONED_ID = re.compile(r"^\d{4}\.\d{4,5}v\d+$")
# Math fonts map bracket glyphs to control codes; Postgres rejects the NULs.
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
REF_HEADING = re.compile(r"^\s*(references|bibliography)\s*$", re.I)
# Appendix headings that commonly follow the references. "A Title" with no
# period after the letter, so author initials ("A. Dosovitskiy") don't match.
APPENDIX_HEADING = re.compile(
    r"^\s*((?i:appendix|appendices|supplementary material)\b.{0,60}"
    r"|[A-H]\s+[A-Z][A-Za-z\-]+(?:\s+[A-Za-z\-]+){0,6})\s*$"
)


def read_papers() -> list[dict]:
    with open(PAPERS_CSV, newline="") as f:
        return list(csv.DictReader(f))


def write_papers(rows: list[dict]) -> None:
    with open(PAPERS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["arxiv_id", "title", "why_chosen"])
        writer.writeheader()
        writer.writerows(rows)


def fetch(url: str, tries: int = 6) -> httpx.Response:
    """GET with exponential backoff; arXiv answers bursts with 406/429/503.

    The body is streamed and read in full, with a 60s timeout, so a stalled PDF download fails
    instead of hanging.
    """
    timeout = httpx.Timeout(60.0)
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=timeout, follow_redirects=True) as client:
        for attempt in range(tries):
            with client.stream("GET", url) as resp:
                if resp.status_code == 200:
                    resp.read()
                    return resp
                if resp.status_code not in RETRY_STATUS or attempt == tries - 1:
                    resp.read()
                    resp.raise_for_status()
            wait = 5 * 2**attempt
            print(f"arXiv returned {resp.status_code}; retrying in {wait}s")
            time.sleep(wait)
    raise AssertionError("unreachable")


def latest_version(bare_id: str) -> str:
    """Resolve a bare arXiv ID to its current version, e.g. 2010.11929 -> 2010.11929v2.

    Reads the abs page's submission history rather than the export API, which
    intermittently answers 406.
    """
    html = fetch(f"https://arxiv.org/abs/{bare_id}").text
    versions = [int(v) for v in ABS_VERSION.findall(html)]
    if not versions:
        raise SystemExit(f"{bare_id}: no versions found on the arXiv abs page")
    return f"{bare_id}v{max(versions)}"


def download() -> None:
    """Download every paper in papers.csv. Bare IDs are resolved to their
    current version and written back to the CSV, so the corpus stays pinned."""
    papers = read_papers()
    for paper in papers:
        if not VERSIONED_ID.match(paper["arxiv_id"]):
            pinned = latest_version(paper["arxiv_id"])
            print(f"pinning {paper['arxiv_id']} -> {pinned}")
            paper["arxiv_id"] = pinned
            write_papers(papers)
            time.sleep(3)  # arXiv asks for a few seconds between requests
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    for paper in papers:
        target = PDF_DIR / f"{paper['arxiv_id']}.pdf"
        if target.exists():
            continue
        print(f"downloading {paper['arxiv_id']}  {paper['title']}")
        pdf = fetch(f"https://arxiv.org/pdf/{paper['arxiv_id']}").content
        if not pdf.startswith(b"%PDF"):
            raise SystemExit(f"{paper['arxiv_id']}: response is not a PDF")
        target.write_bytes(pdf)
        time.sleep(3)  # arXiv asks for a few seconds between requests


def strip_references(lines: list[tuple[int, str]], headings=frozenset()) -> list[tuple[int, str]]:
    """Drop the references block; keep any appendix that follows it.

    The block ends at the first line matching APPENDIX_HEADING or listed in
    `headings` as (page, stripped line)."""
    ref_idx = next(
        (i for i, (page, line) in enumerate(lines) if page > 1 and REF_HEADING.match(line)),
        None,
    )
    if ref_idx is None:
        return lines
    end_idx = next(
        (
            i
            for i in range(ref_idx + 1, len(lines))
            if APPENDIX_HEADING.match(lines[i][1]) or (lines[i][0], lines[i][1].strip()) in headings
        ),
        len(lines),
    )
    return lines[:ref_idx] + lines[end_idx:]


def headings_like_references(doc) -> set[tuple[int, str]]:
    """(page, line) of every line set in the same font and size as the References
    heading. These are section headings, e.g. "A. Linear-probe evaluation" in
    CVPR/ICML style, which APPENDIX_HEADING can't tell apart from author initials."""
    styled = []
    for page_no, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = [s for s in line["spans"] if s["text"].strip()]
                if spans:
                    text = "".join(s["text"] for s in line["spans"]).strip()
                    styled.append((page_no, text, spans[0]["font"], round(spans[0]["size"], 1)))
    ref = next(((f, s) for p, t, f, s in styled if p > 1 and REF_HEADING.match(t)), None)
    return {(p, t) for p, t, f, s in styled if (f, s) == ref}


def extract_pages(pdf_path) -> list[dict]:
    paper_id = pdf_path.stem
    lines = []
    with pymupdf.open(pdf_path) as doc:
        for page_no, page in enumerate(doc, start=1):
            lines.extend(
                (page_no, CONTROL_CHARS.sub("", line)) for line in page.get_text("text").splitlines()
            )
        headings = headings_like_references(doc)
    pages: dict[int, list[str]] = {}
    for page_no, line in strip_references(lines, headings):
        pages.setdefault(page_no, []).append(line)
    return [
        {"paper_id": paper_id, "page": page_no, "text": "\n".join(page_lines)}
        for page_no, page_lines in sorted(pages.items())
        if "".join(page_lines).strip()
    ]


def ingest() -> list[dict]:
    pinned = [p["arxiv_id"] for p in read_papers()]
    missing = [pid for pid in pinned if not (PDF_DIR / f"{pid}.pdf").exists()]
    if missing:
        raise SystemExit(f"missing PDFs (run `lacuna deepread download`): {missing}")
    pages = [page for pid in pinned for page in extract_pages(PDF_DIR / f"{pid}.pdf")]
    write_jsonl(PAGES_PATH, pages)
    print(f"{len(pages)} pages from {len(pinned)} papers -> {PAGES_PATH}")
    return pages
