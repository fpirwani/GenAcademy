"""Topic-oriented extraction and batch review contracts."""

from typing import Literal

from pydantic import BaseModel, Field

from comparison.schemas import Citation


class Topics(BaseModel):
    topics: list[str] = Field(
        min_length=1,
        max_length=24,
        description="Required substantive field names and section headings discovered in the passages; not values or blank signature fields",
    )


class FieldValue(BaseModel):
    topic: str
    status: Literal["found", "not_found", "not_stated", "not_applicable"]
    value: str = Field(
        max_length=2400,
        description="Required: concise extracted value for found; empty only if no value is available",
    )
    evidence: list[Citation] = Field(
        max_length=36,
        description="Required: exact source refs and quotations for every found value; empty only for not_found",
    )


class Fields(BaseModel):
    fields: list[FieldValue] = Field(default_factory=list, max_length=5)


class FieldDecision(BaseModel):
    topic: str
    supported: bool


class FieldReview(BaseModel):
    decisions: list[FieldDecision] = Field(default_factory=list, max_length=5)
