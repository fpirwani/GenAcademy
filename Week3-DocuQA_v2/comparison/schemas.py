from pydantic import BaseModel, Field


class Compatibility(BaseModel):
    comparable: bool
    document_one_type: str = Field(min_length=1, max_length=200)
    document_two_type: str = Field(min_length=1, max_length=200)
    reason: str = Field(min_length=1, max_length=1000)


class Citation(BaseModel):
    ref: str
    quote: str = Field(description="Exact contiguous source quotation, at least 15 characters")
