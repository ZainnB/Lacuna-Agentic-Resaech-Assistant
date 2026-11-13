"""Scout: seed search -> depth-1 citation snowball -> ranked, cited paper set. No LLM in Phase 1."""

from lacuna.scout.core import Ranked, ScoutConfig, ScoutResult, scout

__all__ = ["Ranked", "ScoutConfig", "ScoutResult", "scout"]
