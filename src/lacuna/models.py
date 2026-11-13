"""Records shared between ingest, scout, graph and evals.

`paper_id` is always the OpenAlex work ID, e.g. "W2741809807" (blueprint 7.1). Results from
Semantic Scholar or arXiv are `SearchHit`s until they are resolved to an OpenAlex ID through a
free ID/DOI lookup; only then do they become a `PaperRecord`.
"""

from datetime import date

from pydantic import BaseModel, Field


class AuthorRecord(BaseModel):
    author_id: str  # OpenAlex author ID ("A...") when known, else "name:<normalised name>"
    name: str
    position: int | None = None  # 0-based author order


class PaperRecord(BaseModel):
    paper_id: str
    title: str
    doi: str | None = None  # bare DOI, no "https://doi.org/" prefix
    arxiv_id: str | None = None  # bare, without version: "2010.11929"
    s2_id: str | None = None
    year: int | None = None
    publication_date: date | None = None
    venue: str | None = None
    citation_count: int | None = None
    abstract: str | None = None
    authors: list[AuthorRecord] = Field(default_factory=list)
    referenced_ids: list[str] = Field(default_factory=list)  # OpenAlex IDs; most are outside the set
    source: str = "openalex"  # which API produced the record


class SearchHit(BaseModel):
    """One raw result from a paper API, before resolution to an OpenAlex ID."""

    source: str  # "openalex" | "s2" | "arxiv"
    source_id: str
    title: str
    doi: str | None = None
    arxiv_id: str | None = None
    year: int | None = None
    citation_count: int | None = None
    score: float | None = None  # source relevance score when the API gives one
