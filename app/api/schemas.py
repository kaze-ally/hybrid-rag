from pydantic import BaseModel
from typing import Optional


class IngestRequest(BaseModel):
    source: str


class IngestResponse(BaseModel):
    message: str
    chunks_created: int
    source: str


class JobResponse(BaseModel):
    job_id: str
    status: str        # pending | processing | done | error
    message: str
    chunks_created: Optional[int] = None
    source: Optional[str] = None


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    rerank_top_k: Optional[int] = 3


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    chunks_used: int
    model: str
    retrieval_scores: list[dict]


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    message: str
    qdrant_mode: Optional[str] = None


class DocumentSummary(BaseModel):
    source: str
    chunks: int