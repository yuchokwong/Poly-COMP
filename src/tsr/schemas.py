"""Pydantic response schemas for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    rank: int
    id: int
    image_id: str
    score: float
    vector_score: float | None = None
    keyword_score: float | None = None
    url: str
    captions: list[str] = Field(default_factory=list)
    category: str | None = None
    split: str | None = None
    width: int | None = None
    height: int | None = None


class SearchResponse(BaseModel):
    query: str
    expanded_queries: list[str] = Field(default_factory=list)
    hybrid: bool = False
    latency_ms: float
    count: int
    results: list[SearchResult]


class StatsResponse(BaseModel):
    num_images: int
    index_size: int
    index_type: str
    model: str
    pretrained: str
    embedding_dim: int
    device: str
    ready: bool
    data_dir: str


class HealthResponse(BaseModel):
    status: str
    ready: bool
