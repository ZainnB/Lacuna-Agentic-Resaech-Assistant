# Plan

Working plan for FAMM-RAG. Written for whoever is working on the repo, including future me.

## What this project is

A **benchmark for figure-aware retrieval over scientific PDFs**, with a reference pipeline attached — not a RAG demo.

The pipeline (`ingest → chunk → embed → pgvector → generate`) is commodity and is not the contribution. The contribution is the measurement apparatus:

- Gold evidence labelled by **location** (`paper_id`, `page`, `evidence_ref`, `answer_span`), never by chunk ID, so labels survive re-chunking.
- A **mechanical figure-split check** that rejects a figure question if its answer span appears anywhere in the extracted text. The hard split is honest by construction, not by assertion.
- A **frozen generator** (model, prompt, settings) plus config hashing, so retrieval is the only variable between phases.
- A **judge-calibration path** (`spotcheck` → `agreement`) that treats the LLM judge as something to validate rather than trust.

The reusable question for anyone else is: *is my retrieval actually reaching my tables and figures, or only my prose?*

## Current state (2026-11-04)

| | |
|---|---|
| Pipeline | Works end to end through retrieval. 7 papers → 246 pages → 773 chunks indexed. |
| Eval set | 41 questions, `figrag check-questions` passing. |
| Tests | 14 pass, 1 skips without an API key. |
| **Results** | **None. The eval has never run, and the generator has to move off the paid Anthropic API first.** |
| **Visibility** | **None. The GitHub remote returns 404.** |
| CI | None. |

Retrieval-only recall@5 on the 26 questions that survived the corpus rebuild: text 0.50, table 0.82, figure 0.80. Not yet re-measured on the full set.

Everything downstream is blocked on one run of the eval.

## Corpus

### Selection rule

A paper earns its place by supplying **figure** questions, because that split is the bottleneck. Text and table questions are oversupplied by any paper.

For a figure question to be valid it needs a **raster data plot** — a bar chart, line plot or histogram stored as a bitmap. Two failure modes disqualify most papers:

1. **Vector plots.** If the authors embedded the plot as vector graphics, PyMuPDF extracts every tick label and legend entry as text, so the answer leaks and the span check correctly rejects the question. This is the common case in ML papers.
2. **Qualitative imagery.** Photos, renderings and attention maps are raster but have no single verifiable answer. "Which looks sharper" is not gradeable.

**Fast manual test when vetting a candidate PDF:** open it in a viewer and try to select the text of a plot's axis label or legend entry.
- Highlights → vector → **useless** for the figure split.
- Does not highlight → raster → **candidate**, if the plot is a data chart rather than a photo.

### Keep

| Paper | Role | Evidence |
|---|---|---|
| **AdderNet** `1912.13200v6` | **Figure-capable** | Fig 3 (learning curves) and Fig 4 (histograms) are raster. Confirmed: `counts`, `6000`, `accuracy(%)` appear nowhere in the extracted text. |
| **FlashAttention** `2205.14135v2` | **Figure-capable** | Figs 5, 6, 8 (speedup bar charts, pp. 28–30) are raster. Confirmed: `Dropout + Masking`, `No Masking, No Dropout`, `Masking Only` appear nowhere in the text. Also 21 tables. |
| **ViT** `2010.11929v2` | Tables + distractor | 8 tables, dense ablations. |
| **CLIP** `2103.00020v1` | Tables + distractor | 20 tables, 48 pages of distractor mass. |
| **SigLIP** `2303.15343v4` | Tables + distractor | 9 tables. |

ViT, CLIP and SigLIP form one lineage — all image-text pretraining, all reporting ImageNet zero-shot — so they are mutually confusable. That makes retrieval genuinely hard, which is a feature of a retrieval benchmark, not a redundancy.

### Cut

| Paper | Why |
|---|---|
| **YOLOv3** `1804.02767v1` | Zero figure capability: the raster content on p. 6 is photographs, and Figure 5's own labels (`Person: 99%`, `Horse: 52%`, `Detector #1`) extract as vector text, so they leak. Only 6 pages and 3 tables — the smallest contribution in the corpus. Prose is largely jokes. |
| **Mip-NeRF** `2103.13415v3` | Zero figure capability: all raster content (pp. 2, 7, 8, 17, 18) is qualitative scene renderings, and panel labels (`Ground-Truth`, `NeRF + Area, Center, Misc`) leak. Off-theme relative to the other four, so it is trivially separable and adds little retrieval difficulty. Its tables are good, but tables are already oversupplied. |

Cutting both costs 10 of the 41 questions (4 + 1 unanswerable each), leaving 31.

### Add (verified 2026-11-02)

Five arXiv candidates were downloaded and measured. Raster coverage was computed per page, then candidate answer spans were run through the same boundary-aware check the eval uses.

| Candidate | Raster pages (>5% coverage) | Verdict |
|---|---|---|
| **Llama 2** `2307.09288` | **15** — Fig 12 grouped win/tie/loss bars confirmed | **Add** |
| **GPT-3** `2005.14165` | **26** — Fig 1.3 accuracy-vs-parameters lines confirmed | **Add** |
| Whisper `2212.04356` | **0** | Reject — all plots vector |
| Scaling Laws `2001.08361` | **0** | Reject — all plots vector |
| Chinchilla `2203.15556` | **0** | Reject — all plots vector |

Confirmed clean (pixels only, usable as answer spans): `Llama-2-34b-chat vs. Falcon-40b-instruct`, `Win Rate %`, `Aggregate Performance Across Benchmarks`, `Parameters in LM (Billions)`, `0.4B`.

Confirmed leaking, do not use: `Single Turn`, `Multi-Turn`, `Few Shot`, `One Shot`, `6.7B`, `13B`.

**Labelling rule for long papers.** In a 75-page paper most short answers appear somewhere in prose, so an answer span must be a *distinctive* string or a value read off the chart: panel headers, chart titles, axis captions, or a bar height that appears nowhere in the text. Short generic labels will leak and be rejected.

### Not pursued

Non-arXiv journal papers (PLOS ONE, Scientific Reports) were considered and declined on two grounds. They are off-theme, so they are trivially separable from the rest of the corpus and add no retrieval difficulty — the same reason Mip-NeRF was cut. And they are not arXiv-hosted, so they would need a new fetch path and ID scheme: `src/figrag/ingest.py` hardcodes `arxiv.org/pdf/{id}` and `VERSIONED_ID` expects `^\d{4}\.\d{4,5}v\d+$`.

They are unnecessary in any case. Estimated figure-question capacity after the additions: AdderNet ~4, FlashAttention ~8–10, Llama 2 ~8–12, GPT-3 ~8–12 — comfortably past the target of 20.

### Resulting corpus (7 papers) — rebuilt 2026-11-04

AdderNet, FlashAttention, ViT, CLIP, SigLIP, Llama 2 `2307.09288v2`, GPT-3 `2005.14165v4`.

The through-line is **transformers and scaling, across vision and language**. The README currently says "computer-vision papers" and must be reworded. Note the size imbalance: Llama 2 (77 pp), GPT-3 (75 pp) and CLIP (48 pp) are about 70% of the corpus by page count.

Build: **246 pages → 773 chunks**, index `ce40123b`. Reference stripping verified by hand on both new papers — GPT-3's references run to the end of the document (pp. 68–75, correctly dropped), Llama 2's sit mid-document (pp. 37–45) with the appendix after them correctly kept.

### Relabelled 2026-11-04

Dropping YOLOv3 and Mip-NeRF broke 10 of the 41 questions. All 10 are replaced and verified on the page they cite: q007, q008, q022, q023, q035 against Llama 2; q011, q012, q026, q027, q039 against GPT-3.

q035 and q039 are near misses. Llama 2's 31.5 tie rate exists only for 70B, and `Llama-2-13b-chat vs. ChatGPT` is absent. GPT-3 reports no carbon figure, while Llama 2 reports 291.42 tCO2eq, so q039 also tests confusion across papers.

### New guard

`orphan_violations()` in `src/figrag/evaluate.py` catches questions whose paper or page has left the corpus. Previously `validate_questions` passed them and they scored as permanent retrieval misses — a labelling error disguised as a retrieval result. Wired into `figrag check-questions` and covered by `tests/test_questions.py::test_no_orphaned_questions`.

### Note on unanswerable questions

A good unanswerable question is a **near miss**: the paper is in the corpus, the topic is plausible, but the specific number is absent. That is what makes it a hallucination probe. If a paper is removed from the corpus, its unanswerable question degrades into a trivial out-of-corpus question and must be rewritten against a paper that is still present.

## Phases

| Phase | Change | Status |
|---|---|---|
| 0 | Text-only baseline and a hand-labelled eval set | blocked on first eval run |
| 0.5 | Public repo, CI, results in the README | - |
| 1 | Figure extraction, VLM captioning, multimodal retrieval | - |
| 2 | Eval as a CI gate, and cost per query | - |
| — | LoRA-tuned retriever | **parked, see below** |

Only retrieval changes between phases. The generator model, its prompt and the chunking config are frozen (`config.toml`, `src/figrag/generate.py`), and every result file is named by a hash of them.

### Why the LoRA retriever is parked

41 questions cannot support fine-tuning. Any improvement would sit inside the noise band of a test set that small (roughly ±15 points), so the result would be unverifiable. The only honest design — train on synthetic pairs, hold out the hand-labelled set entirely — is weeks of work for a number that still would not be defensible. Revisit only if the eval set passes ~200 questions.

## Models: free only (decided 2026-11-04)

No paid API key will be used. Researched against provider docs on 2026-11-04.

| Role | Proposed | Why |
|---|---|---|
| Generator | `openai/gpt-oss-120b` on Groq | Open weights, strict JSON schema, and Groq names it as the replacement for retired models. |
| Judge | Gemini on the AI Studio free tier: `gemini-3.8-flash` if quota allows, else `gemini-3.5-flash-lite` | Different vendor from the generator, so it does not grade itself. Both are stable with no announced shutdown. |

A full run is 81 LLM calls and about 174K input tokens: 41 generator calls (~2,240 tokens each, 92K
total) and about 40 judge calls (82K total). Measured from the 773 indexed chunks at `top_k = 5`,
not estimated. The generator role is 46% of Groq's documented 200K daily tokens, so a full run
fits in one day; at 8K tokens/min it paces out to about 11 minutes.

Rejected or held back:

- **Cohere trial key:** the best technical fit, and still rejected. 1,000 calls a month with no
  documented token cap, 20 RPM, 256K context on `command-a-03-2025`, and strict JSON schema the docs
  claim holds 100% of the time — 12 full runs a month, where Groq is token-bound to roughly one a day.
  Two reasons against. Cohere's terms prohibit access "for any other benchmarking or competitive
  purposes" and use "for personal, family or household purposes"; the clause is probably aimed at
  vendor-vs-vendor comparisons rather than at a retrieval benchmark holding the LLM fixed, but this
  repo publishes per-split accuracy from a named Cohere model and a judge-agreement number about it,
  which is an avoidable argument for a public portfolio project. More decisively, Command A is closed
  weights, and Cohere retired its whole older Command line on 2025-09-15; if the frozen generator is
  retired, Phase 0 can never be regenerated. `gpt-oss-120b` weights are downloadable, so a run stays
  reproducible on any hardware even if Groq drops the model. Keep Cohere as the fallback if Groq's
  real quota is tighter than documented, or as a judge in place of Gemini.
- **Cerebras:** the rate-limit page is generous, but the free trial is a one-time $5 credit that needs a payment method and expires after 30 days.
- **Mistral free tier:** reported to require opting in to data training and phone verification. Not checked against Mistral's docs.
- **Ollama (local):** the fallback. It has no quota and pins exactly, but the machine has 16 GB and gpt-oss-20b is about 13 GB.

Open points:

- **Open weights are the reason for the generator choice**, not throughput. The frozen generator is
  what makes phases comparable, so it has to stay runnable after any provider retires it.
- Quality has not been measured. None of these is assumed to match Opus. Retrieval metrics do not depend on the LLM, and the judge is checked by `spotcheck` → `agreement`.
- Free quotas are not fully confirmed. Groq's page shows 30 RPM, 1K RPD, 8K TPM and 200K TPD but labels
  them Developer-plan values, so check the account page. Google lists limits only inside AI Studio, and
  third-party figures for Flash range from 20 to 1,500 requests a day.
- Groq gave 17–60 days' notice on recent shutdowns, so a frozen generator can be retired. Open weights keep the runs reproducible elsewhere.
- Free-tier Gemini content is used to improve Google's products. The inputs here are public papers and our own questions, so nothing sensitive should go through it.

## Next steps, in order

1. **Fix the frozen generator prompt.** `SYSTEM_PROMPT` still says "computer vision research papers", but the corpus now includes Llama 2 and GPT-3. It has to change before results exist.
2. **Move `call_json` off the Anthropic SDK.** Add per-provider clients, pacing for rate limits, and retries. `tests/test_eval_subset.py` and the README also name `ANTHROPIC_API_KEY`.
3. **Run the eval.** Smoke-test the 10-question subset first, then freeze the models and run all 41.
4. **Make the repo public and add CI.** `pytest` and `figrag check-questions` need neither a key nor Postgres. Untrack the two committed `.DS_Store` files.
5. **Run the judge calibration.** `figrag spotcheck` 15 judgements, hand-label `human_supported`, run `figrag agreement`, and report the number in the README.
6. **Expand the figure split to 15–20 questions** using Llama 2 and GPT-3 raster plots, and reword the README, which still says "computer-vision papers" and lists the old corpus.
7. **Phase 1.** Extract figures, caption them with a VLM, embed and retrieve the captions, and report the before/after figure-split gap. This is the headline result. The captioning model needs a free option too.
8. **Results page.** A public page with the per-split table and the gap chart, regenerated by CI. For a benchmark this matters more than a chat UI.

## Known weaknesses

State these before someone else finds them:

- **Figure split is n=5.** Any claim about the figure gap is directional at best until step 6 lands.
- **Small corpus.** Seven papers, with Llama 2, GPT-3 and CLIP about 70% of the pages.
- **Single annotator.** No inter-annotator agreement, so label quality is unverified.
- **Judge uncalibrated.** The `spotcheck` path exists but has never been run, so faithfulness numbers carry no error bar yet.
