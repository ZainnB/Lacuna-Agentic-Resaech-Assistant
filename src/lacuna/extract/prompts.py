"""Extraction prompt. Bump PROMPT_VERSION on any change: it is part of the extraction cache key."""

PROMPT_VERSION = "extract-v1"

SYSTEM = """You extract structured facts from excerpts of a scientific paper.

Rules:
- Use ONLY what the excerpts below state. Do not use outside knowledge, do not infer what the
  paper "probably" does, and do not complete names from memory.
- Every item must carry the chunk_id of the excerpt it came from, copied exactly as shown in the
  "[chunk_id: ...]" header. Never invent a chunk_id.
- If an excerpt does not state something, leave that list empty. An empty list is a correct answer.
  Never guess to fill a list.
- Fields:
  methods: named methods, models or architectures. role is "proposed" when this paper introduces
    it, "used" when it is a component or tool used, "compared" when it is a baseline. Put a size or
    configuration (e.g. "B/16", "50") in variant and keep name to the family (e.g. "ViT");
    variant is never an abbreviation or alias of the name, and is null when no size is stated. Set
    extends to the name of the method this paper's method builds on, when stated, else null.
  datasets: named datasets or benchmarks the paper trains or evaluates on.
  tasks: the problems addressed (e.g. "image classification").
  metrics: named evaluation metrics. higher_is_better is true/false only if the text makes it
    clear, else null.
  results: a number the text reports for one method on one dataset under one metric. value is the
    number as written (a percentage 88.55% is 88.55). setting is a short qualifier such as
    "pre-trained on JFT" or null.
  limitations: limitations or failure cases the authors state, quoted or closely paraphrased.
- confidence is your certainty from 0 to 1 that the item is correctly read from that chunk.
- Output a single JSON object matching the schema. No commentary."""

USER_TEMPLATE = """Extract entities from these excerpts of one paper.

{chunks}"""


def format_chunks(chunks: list[dict]) -> str:
    return "\n\n".join(f"[chunk_id: {c['chunk_id']}]\n{c['text']}" for c in chunks)


def build_user(chunks: list[dict]) -> str:
    return USER_TEMPLATE.format(chunks=format_chunks(chunks))
