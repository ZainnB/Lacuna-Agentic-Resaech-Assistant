# E3 gold labels (extraction quality)

Blueprint section 11, E3: 30 hand-labelled papers, precision/recall/F1 per entity type. The labels are
written by the labeller, never generated or pre-filled by a model (decisions.md, 2026-11-14).

## Files

- `todo/<paper_id>.json` — templates from `lacuna evals e3-template <topic_id> --n 30`. They hold the
  title and abstract and **empty** entity lists. Fill one in, then move it to `labels/`.
- `labels/<paper_id>.json` — finished labels. Unfinished files (no `labeller`, or all lists empty) are ignored.
- `split.json` — `{paper_id: "tuning" | "reporting"}`, half each, fixed once written. Tune prompts and
  the alias table on `tuning` only. Quote results from `reporting` only.

## Label format

```json
{
  "paper_id": "W1234567890",
  "title": "...",
  "labeller": "your name or initials",
  "labelled_on": "2026-11-20",
  "scope": "abstract",
  "methods": ["ViT"],
  "datasets": ["ImageNet"],
  "tasks": ["image classification"],
  "metrics": ["top-1 accuracy"],
  "notes": "free text, not scored"
}
```

`paper_id` is the OpenAlex work ID. `scope` is `abstract` (you read only the abstract) or `fulltext`
(you read the paper). Each list is a **set** of names; order and repeats do not matter. The template also
carries `abstract` and `split`; they are ignored when scoring.

## Guidelines

Label what the **text states**, not what you know about the paper or the field.

- **Methods**: named models, algorithms, architectures, training techniques and frameworks the text refers
  to by name. Proposed versus used does not matter in E3 v1. Generic descriptions ("a neural network",
  "our approach") are not methods.
- **Tasks**: the problem being solved or the benchmark category ("image classification", "machine
  translation"). If it is something a method *does*, it is a task. If it is something a method *is*, it is
  a method.
- **Datasets**: only if the text names them ("ImageNet", "WMT14"). A described but unnamed dataset ("a
  large collection of web images") is not labelled.
- **Metrics**: named evaluation measures ("top-1 accuracy", "BLEU", "F1"). Raw numbers are not metrics.
- **Variants**: label the base name. "ViT-B/16" becomes `ViT`, "ResNet-50" becomes `ResNet`,
  "ImageNet-1k" becomes `ImageNet`. Scoring also strips size and version suffixes, but the base name
  keeps your labels clean.
- **Names**: use the form the text uses, spelled out or abbreviated, once. Do not list both "Vision
  Transformer" and "ViT"; alias merging is the resolver's job and is part of what E3 measures.
- **Scope**: if you read only the abstract, set `scope: "abstract"` and label only what the abstract states.
- When unsure whether something counts, leave it out and note it in `notes`.

## Worked examples (made-up text, not corpus papers)

**Example 1.** "We introduce Foldnet, a graph transformer for predicting protein stability. Trained on the
StabBench dataset and evaluated with Spearman correlation, Foldnet outperforms the LSTM baseline."

```json
{"methods": ["Foldnet", "graph transformer", "LSTM"], "datasets": ["StabBench"],
 "tasks": ["protein stability prediction"], "metrics": ["Spearman correlation"]}
```

`graph transformer` is a named architecture class; "baseline" is not a name. The dataset is named, so it is labelled.

**Example 2.** "Lumen-L/32 reaches 91.2 accuracy on a large private image collection and improves
calibration, using self-distillation during training."

```json
{"methods": ["Lumen", "self-distillation"], "datasets": [],
 "tasks": [], "metrics": ["accuracy", "calibration"]}
```

`Lumen-L/32` becomes `Lumen`. The collection is unnamed, so no dataset. No task is stated, so none is
invented. "91.2" is a number, not a metric.
