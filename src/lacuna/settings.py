"""Runtime settings, read from the environment and from `.env` at the repo root.

Secrets never live in code or in tracked files. Copy `.env.example` to `.env` and fill it in.
Every field here maps to an environment variable of the same name in upper case.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # Blueprint section 2. 0 = build, 1 = personal use, 2 = paid. Paid providers refuse below 2.
    spend_stage: int = 0

    postgres_dsn: str = "postgresql://lacuna:lacuna@localhost:5433/lacuna"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "lacuna-dev"

    # Paper APIs. OpenAlex requires a free key and meters usage in dollars ($1/day free):
    # ID and DOI lookups are free, list/filter calls cost $0.0001, searches $0.001.
    openalex_api_key: str | None = None
    openalex_daily_budget_usd: float = 0.90  # hard stop enforced in code, under the free allowance
    s2_api_key: str | None = None  # 1 request/s with a key, a shared pool without one
    contact_email: str | None = None  # sent to OpenAlex and arXiv as a courtesy, only if set

    # Free-tier LLM providers. A missing key means the provider is skipped in the chain.
    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    cerebras_api_key: str | None = None
    openrouter_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434/v1"

    data_dir: Path = ROOT / "data"

    @field_validator("*", mode="before")
    @classmethod
    def _blank_is_none(cls, v):
        # `.env.example` ships with empty values; an empty string must not count as a key.
        return None if isinstance(v, str) and v.strip() == "" else v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
