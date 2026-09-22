"""Download the pinned arXiv PDFs and extract text per page with PyMuPDF.

Tables are left exactly as PyMuPDF extracts them: the messy text is the
realistic baseline.
"""
import csv
import re

import arxiv
import pymupdf

from figrag.config import PAGES_PATH, PAPERS_CSV, PDF_DIR, write_jsonl

VERSIONED_ID = re.compile(r"^\d{4}\.\d{4,5}v\d+$")
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


def download() -> None:
    """Download every paper in papers.csv. Bare IDs are resolved to their
    current version and written back to the CSV, so the corpus stays pinned."""
    papers = read_papers()
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    client = arxiv.Client()
    search = arxiv.Search(id_list=[p["arxiv_id"] for p in papers])
    by_base_id = {r.get_short_id().split("v")[0]: r for r in client.results(search)}
    for paper in papers:
        result = by_base_id[paper["arxiv_id"].split("v")[0]]
        pinned = result.get_short_id()
        if not VERSIONED_ID.match(paper["arxiv_id"]):
            print(f"pinning {paper['arxiv_id']} -> {pinned}")
            paper["arxiv_id"] = pinned
        target = PDF_DIR / f"{pinned}.pdf"
        if target.exists():
            continue
        print(f"downloading {pinned}  {result.title}")
        result.download_pdf(dirpath=str(PDF_DIR), filename=target.name)
    write_papers(papers)


def strip_references(lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Drop the references block; keep any appendix that follows it."""
    ref_idx = next(
        (i for i, (page, line) in enumerate(lines) if page > 1 and REF_HEADING.match(line)),
        None,
    )
    if ref_idx is None:
        return lines
    end_idx = next(
        (i for i in range(ref_idx + 1, len(lines)) if APPENDIX_HEADING.match(lines[i][1])),
        len(lines),
    )
    return lines[:ref_idx] + lines[end_idx:]


def extract_pages(pdf_path) -> list[dict]:
    paper_id = pdf_path.stem
    lines = []
    with pymupdf.open(pdf_path) as doc:
        for page_no, page in enumerate(doc, start=1):
            lines.extend((page_no, line) for line in page.get_text("text").splitlines())
    pages: dict[int, list[str]] = {}
    for page_no, line in strip_references(lines):
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
        raise SystemExit(f"missing PDFs (run `figrag download`): {missing}")
    pages = [page for pid in pinned for page in extract_pages(PDF_DIR / f"{pid}.pdf")]
    write_jsonl(PAGES_PATH, pages)
    print(f"{len(pages)} pages from {len(pinned)} papers -> {PAGES_PATH}")
    return pages
