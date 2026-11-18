"""Lacuna MCP server (Mode A): the host is the planner and writer, Lacuna is the tool server.

Over stdio nothing but the protocol may reach stdout, so progress bars are disabled here, before
anything heavy (huggingface, sentence-transformers, tqdm) is imported.
"""

import os

for _k, _v in {
    "TQDM_DISABLE": "1",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "TRANSFORMERS_VERBOSITY": "error",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
}.items():
    os.environ.setdefault(_k, _v)
