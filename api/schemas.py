from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    doc_id: str
    score: float
    url: str
    title: str
    token_count: int


class SearchResponse(BaseModel):
    query: str
    tokens: list[str]
    missing: list[str]
    mode: str
    total_hits: int
    results: list[SearchResult]
    elapsed_ms: float


class HealthResponse(BaseModel):
    status: str
    total_docs: int
    vocab_size: int
    avg_doc_length: float
    total_tokens: int


class ErrorResponse(BaseModel):
    detail: str