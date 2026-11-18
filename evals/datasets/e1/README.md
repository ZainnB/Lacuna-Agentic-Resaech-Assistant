# E1 survey recall dataset

Ground truth: the resolved reference lists (OpenAlex `referenced_works`) of 5 recent CV/ML surveys.
Downloaded 2026-11-14, unauthenticated, no key and no email sent. Format is defined by
`src/lacuna/evals/e1.py::load_dataset`.

## Files

- `surveys.json`: per survey `openalex_id, arxiv_id, doi, title, publication_date, query, sub_area, n_refs,
  n_refs_resolved`, plus `arxiv_first_date` (extra field, ignored by the loader).
- `refs/<openalex_id>.json`: `id, doi, arxiv_id, title, publication_year, publication_date, cited_by_count, type`
  for every reference OpenAlex returned. `arxiv_id` is parsed only from `10.48550/arxiv.*` DOIs, so it is null
  for most references.
- `freeze.json`: retrieval cutoff per survey = **earlier of** the OpenAlex record date and the arXiv v1
  date. OpenAlex often holds the later journal version (e.g. W4410431342 is dated 2025-05-16, arXiv v1
  2025-03-04), so the arXiv date is the stricter leakage guard.

## Selection criteria

arXiv-hosted (arXiv id verified through the arXiv API), first posted 2025-01-01 or later, OpenAlex record
with at least 80 resolved references, five different sub-areas, question written as a user would ask it
(not the title).

| survey | arXiv | sub-area | refs / resolved |
|---|---|---|---|
| Vision transformers on the edge: model compression and acceleration | 2503.02891 | efficient ViTs / edge | 164 / 159 |
| A Survey on Efficient Vision-Language Models | 2504.09724 | vision-language models | 198 / 195 |
| A Survey on 3D Gaussian Splatting Applications: Segmentation, Editing, and Generation | 2508.09977 | 3D Gaussian splatting | 254 / 254 |
| Looking Beyond the Obvious: Abstract Concept Recognition for Video Understanding | 2508.20765 | video understanding | 281 / 279 |
| Instruction-based image editing: data, models, evaluation, applications | 2607.25642 | diffusion image editing | 110 / 106 |

## Candidates rejected

- OpenAlex record has 0 or <80 references: Image Editing with Diffusion Models: A Survey (2504.13226, 0),
  Training-free Open-Vocabulary Semantic Segmentation survey (2505.22209, 0), SSL for Image Segmentation
  (2505.13584, 15), Token Compression Meets Compact ViTs (2507.09702, 25), Video Scene Parsing (2506.13552, 19).
- First posted before 2025: Diffusion Model-Based Image Editing: A Survey (arXiv 2402.17525), Video
  Understanding With LLMs: A Survey (arXiv 2312.17432), A Survey on 3D Gaussian Splatting (arXiv 2401.03890).
- Same sub-area as a chosen survey: Vision-Language Models for Edge Networks (VLM), 3DGS Compression
  (2502.19457), SUCCESS-GS (2512.07197), VLMs without labels (2508.05547).
- Open-vocabulary detection and self-supervised learning: no arXiv survey met the 80-reference bar in the
  OpenAlex lists searched, so they are not covered. Sub-area coverage is therefore ViT, VLM, 3DGS, video,
  image editing.

## Requests

- `GET /works/W…?select=id,doi,title,publication_date,ids,referenced_works` (free)
- `GET /works?filter=openalex_id:W1|…|W100&per_page=100&select=id,doi,title,publication_year,publication_date,cited_by_count,ids,type`
- Candidate search: `GET /works?search=<topic>&filter=from_publication_date:2025-01-01,type:article|preprint&per_page=15&select=…`
  and `GET /works/doi:10.48550/arxiv.<id>` lookups.
- 0.15 s sleep between calls.

## Calls and cost

37 OpenAlex calls: 7 keyword searches ($0.001 each), 12 filter batch calls ($0.0001 each), 18 single-entity
lookups (free). Estimated total about $0.008, well under the $0.10 keyless daily budget. About 6 arXiv API
calls (3.5 s apart).

## Caveats

- `n_refs_resolved < n_refs`: some referenced IDs were not returned by the filter (merged or deleted works);
  they are missing from gold.
- Gold is OpenAlex's parse of each reference list, not the paper's bibliography; it can miss or mis-match items,
  and the OpenAlex record of two surveys is the journal version, whose reference list may differ from the arXiv one.
- Some references postdate the OpenAlex or arXiv date only if OpenAlex metadata is wrong; not checked per item.
- Queries and sub-area labels are author judgement.
