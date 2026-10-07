from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class DocumentInput(StrictModel):
    source_key: str = Field(min_length=1, max_length=120, pattern=r"^[a-zA-Z0-9_.:-]+$")
    title: str = Field(min_length=3, max_length=200)
    text: str = Field(min_length=20, max_length=100_000)
    source_url: HttpUrl | None = None


class DocumentInfo(StrictModel):
    id: UUID
    source_key: str
    title: str
    source_url: str | None
    version: int
    content_hash: str
    chunk_count: int
    created_at: datetime
    updated_at: datetime


class IngestResult(StrictModel):
    document: DocumentInfo
    changed: bool


class LeadContext(StrictModel):
    reference: str | None = Field(default=None, max_length=100)
    message: str = Field(min_length=1, max_length=6000)
    products: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("products")
    @classmethod
    def bounded_products(cls, values: list[str]) -> list[str]:
        if any(not v.strip() or len(v) > 200 for v in values):
            raise ValueError("Each product must contain 1–200 characters")
        return values


class DraftRequest(StrictModel):
    question: str = Field(min_length=5, max_length=2000)
    lead: LeadContext | None = None


class SearchRequest(StrictModel):
    query: str = Field(min_length=3, max_length=8000)
    mode: Literal["lexical", "vector", "hybrid"] | None = None
    limit: int = Field(default=5, ge=1, le=10)


class Source(StrictModel):
    id: str
    chunk_id: UUID
    document_id: UUID
    title: str
    source_key: str
    source_url: str | None
    version: int
    page: int | None
    text: str
    score: float


class AnswerBlock(StrictModel):
    text: str = Field(min_length=1, max_length=1600)
    source_ids: list[str] = Field(min_length=1, max_length=8)


class ModelAnswer(StrictModel):
    answerable: bool
    blocks: list[AnswerBlock] = Field(default_factory=list, max_length=6)
    clarifying_questions: list[str] = Field(default_factory=list, max_length=5)
    missing_information: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("clarifying_questions", "missing_information")
    @classmethod
    def bounded_items(cls, values: list[str]) -> list[str]:
        if any(not v.strip() or len(v) > 500 for v in values):
            raise ValueError("Items must contain 1–500 characters")
        return values


class DraftResponse(StrictModel):
    request_id: str
    status: Literal["ready", "insufficient_evidence"]
    generation_mode: Literal["ollama", "extractive"]
    draft: str
    clarifying_questions: list[str]
    missing_information: list[str]
    sources: list[Source]
    timings_ms: dict[str, float]
