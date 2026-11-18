"""Loads config/llm.toml: role -> fallback chain, plus optional per-provider limit overrides."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from lacuna.settings import ROOT

CONFIG_PATH = ROOT / "config" / "llm.toml"
ROLES = ("default", "generator", "judge", "extractor", "screener")


@dataclass
class LLMConfig:
    chains: dict[str, list[str]] = field(default_factory=dict)
    limits: dict[str, dict] = field(default_factory=dict)

    def chain_for(self, role: str) -> list[str]:
        chain = self.chains.get(role) or self.chains.get("default")
        if not chain:
            raise ValueError(f"no chain configured for role {role!r} and no 'default' chain")
        return list(chain)


def load_config(path: Path | None = None) -> LLMConfig:
    path = path or CONFIG_PATH
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return LLMConfig(
        chains={k: list(v) for k, v in raw.get("chains", {}).items()}, limits=raw.get("limits", {})
    )
