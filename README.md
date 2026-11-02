# FAMM-RAG

RAG over computer-vision papers usually retrieves text only. Many answers in these papers are in tables and figures, so text-only retrieval can't reach them. This repo measures that gap and closes it one phase at a time:

| Phase | Change | Status |
|---|---|---|
| 0 | Text-only baseline and a hand-labelled eval set | in progress |
| 1 | Figure extraction, captioning and multimodal retrieval | - |
| 2 | Eval as a CI gate, and cost per query | - |
| 3 | LoRA-tuned retriever | - |

Only retrieval changes between phases. The generator model, its prompt and the chunking config are frozen (`config.toml`, `src/figrag/generate.py`), and every result file is named by a hash of them.

## Corpus

Seven arXiv papers, pinned by version in `data/papers.csv`: ViT, CLIP, SigLIP, YOLOv3, AdderNet, Mip-NeRF and FlashAttention (133 pages after the references are stripped; appendices are kept). The pinned IDs feed the index hash, so changing the corpus makes the stored index stale until it is rebuilt.

## Eval set

`eval/questions.jsonl` has 41 hand-written questions in four splits:

| split | n | answer lives in |
|---|---:|---|
| text | 14 | prose |
| table | 15 | a number in a table (PyMuPDF extracts table cells as messy text) |
| figure | 5 | only a plot or diagram; checked mechanically, so the answer span must not appear in the extracted text |
| unanswerable | 7 | nowhere; the correct answer is "not found", which measures hallucination |

Gold evidence is recorded by location (`paper_id`, `page`, `evidence_ref`, `answer_span`), not by chunk ID, so the labels still hold when chunking changes.

Most plots in these papers are vector graphics, and their tick labels and legends come out of PyMuPDF as text. A figure question therefore needs an answer that exists only as pixels. The current figure questions use the raster plots in AdderNet (Figure 3) and FlashAttention (Figures 5, 6 and 8).

## Setup (macOS, Apple Silicon)

Needs [uv](https://docs.astral.sh/uv/) and Docker (OrbStack or Docker Desktop). Put `ANTHROPIC_API_KEY` in your environment.

```bash
uv sync --extra dev
docker compose up -d --wait        # postgres + pgvector on localhost:5433
uv run figrag download             # pins bare arXiv IDs in data/papers.csv and fetches the PDFs
uv run figrag build                # extract pages -> chunk -> embed (bge-base on MPS) -> load into Postgres
uv run figrag ask "What patch size does ViT-B/16 use?"   # quick sanity check
```

## Reproduce the results

```bash
make reproduce                     # download + build + eval -> eval/results/<date>_<hash>.{json,md}
uv run figrag check-questions      # schema check and figure-split check
uv run pytest                      # unit tests, plus the 10-question eval subset when the DB and API key are available
```

Faithfulness is scored by an LLM judge. To measure how far the judge can be trusted, sample 15 of its judgements (`uv run figrag spotcheck eval/results/<run>.json`), fill in the `human_supported` column by hand, then run `uv run figrag agreement <csv>`.

## Baseline results

_Pending the first full eval run._
