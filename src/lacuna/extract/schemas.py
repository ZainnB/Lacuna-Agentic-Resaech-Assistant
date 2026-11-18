"""The extraction schema, as Pydantic models and as a strict-compatible JSON schema.

Every item cites the `chunk_id` it came from. The JSON schema is hand-built (not
`model_json_schema()`) so it stays inside the strict subset providers accept: every property
required, nullable via ["type", "null"], `additionalProperties: false` everywhere, and none of
minimum/maximum/format/default. Range checks (confidence 0-1) live in Pydantic, which clamps.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Role = Literal["proposed", "used", "compared"]


class _Item(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class _Cited(_Item):
    chunk_id: str = Field(min_length=1)
    confidence: float = 0.5

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, v):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("confidence must be a number")
        return min(1.0, max(0.0, float(v)))


class MethodItem(_Cited):
    name: str = Field(min_length=1)
    variant: str | None = None
    role: Role
    extends: str | None = None


class DatasetItem(_Cited):
    name: str = Field(min_length=1)


class TaskItem(_Cited):
    name: str = Field(min_length=1)


class MetricItem(_Cited):
    name: str = Field(min_length=1)
    higher_is_better: bool | None = None


class ResultItem(_Cited):
    method: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    value: float
    setting: str | None = None


class LimitationItem(_Item):
    text: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)


class Extraction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    methods: list[MethodItem] = []
    datasets: list[DatasetItem] = []
    tasks: list[TaskItem] = []
    metrics: list[MetricItem] = []
    results: list[ResultItem] = []
    limitations: list[LimitationItem] = []

    LISTS: ClassVar[tuple[str, ...]] = ("methods", "datasets", "tasks", "metrics", "results", "limitations")

    def is_empty(self) -> bool:
        return not any(getattr(self, k) for k in self.LISTS)

    def cited_chunk_ids(self) -> set[str]:
        return {i.chunk_id for k in self.LISTS for i in getattr(self, k)}

    def counts(self) -> dict[str, int]:
        return {k: len(getattr(self, k)) for k in self.LISTS}


def _s(nullable: bool = False) -> dict:
    return {"type": ["string", "null"] if nullable else "string"}


def _obj(props: dict) -> dict:
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


def extraction_json_schema() -> dict:
    cited = {"chunk_id": _s(), "confidence": {"type": "number"}}
    return _obj(
        {
            "methods": {
                "type": "array",
                "items": _obj(
                    {
                        "name": _s(),
                        "variant": _s(True),
                        "role": {"type": "string", "enum": ["proposed", "used", "compared"]},
                        "extends": _s(True),
                        **cited,
                    }
                ),
            },
            "datasets": {"type": "array", "items": _obj({"name": _s(), **cited})},
            "tasks": {"type": "array", "items": _obj({"name": _s(), **cited})},
            "metrics": {
                "type": "array",
                "items": _obj({"name": _s(), "higher_is_better": {"type": ["boolean", "null"]}, **cited}),
            },
            "results": {
                "type": "array",
                "items": _obj(
                    {
                        "method": _s(),
                        "dataset": _s(),
                        "metric": _s(),
                        "value": {"type": "number"},
                        "setting": _s(True),
                        **cited,
                    }
                ),
            },
            "limitations": {"type": "array", "items": _obj({"text": _s(), "chunk_id": _s()})},
        }
    )
