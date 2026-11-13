"""Token-window chunking within each page, so every chunk has exactly one page."""
import re

from transformers import AutoTokenizer

from lacuna.deepread.config import CHUNKS_PATH, PAGES_PATH, read_jsonl, write_jsonl

# "3 Method", "4.2 Ablation Studies", "5. Conclusion". Letters-only titles keep
# table rows ("12 ViT-L/16 85.3") from being mistaken for headings.
SECTION_HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2})*)\.?\s+([A-Z][A-Za-z\-:,&/ ]{2,60})$", re.M)


def heading_offsets(text: str) -> list[tuple[int, str]]:
    return [(m.start(), f"{m.group(1)} {m.group(2).strip()}") for m in SECTION_HEADING.finditer(text)]


def windows(offsets: list[tuple[int, int]], size: int, overlap: int):
    """Yield (start_char, end_char) for token windows of `size` with `overlap`."""
    step = size - overlap
    for start in range(0, len(offsets), step):
        window = offsets[start : start + size]
        yield window[0][0], window[-1][1]
        if start + size >= len(offsets):
            break


def chunk_pages(pages: list[dict], cfg: dict) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    tokenizer.model_max_length = 10**9  # we window ourselves; silence the >512 warning
    chunks = []
    section = {}  # paper_id -> last heading seen, carried across pages
    for page in pages:
        text, paper_id = page["text"], page["paper_id"]
        headings = heading_offsets(text)
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        for i, (start, end) in enumerate(windows(enc["offset_mapping"], cfg["tokens"], cfg["overlap"])):
            current = section.get(paper_id)
            for offset, title in headings:
                if offset <= start:
                    current = title
            chunks.append({
                "id": f"{paper_id}:p{page['page']}:c{i}",
                "paper_id": paper_id,
                "page": page["page"],
                "section": current,
                "text": text[start:end],
            })
        if headings:
            section[paper_id] = headings[-1][1]
    return chunks


def chunk(cfg: dict) -> list[dict]:
    chunks = chunk_pages(read_jsonl(PAGES_PATH), cfg["chunking"])
    write_jsonl(CHUNKS_PATH, chunks)
    print(f"{len(chunks)} chunks -> {CHUNKS_PATH}")
    return chunks
