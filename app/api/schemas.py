from pydantic import BaseModel
from typing import Optional

class IngestRequest(BaseModel):
    source: str  # file path or raw text

class IngestResponse(BaseModel):
    message: str
    chunks_created: int
    source: str

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5        # chunks to retrieve before reranking
    rerank_top_k: Optional[int] = 3  # final chunks after reranking

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    chunks_used: int
    model: str
    retrieval_scores: list[dict]  # for observability / demo purposes

class HealthResponse(BaseModel):
    status: str
    qdrant: str
    message: str