from fastapi import APIRouter, HTTPException
from app.api.schemas import (
    IngestRequest, IngestResponse,
    QueryRequest, QueryResponse, HealthResponse
)
from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import embed_chunks
from app.retrieval.vector_store import store_chunks, vector_search, get_client
from app.retrieval.bm25_store import build_bm25_index, bm25_search, load_bm25_from_qdrant
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.reranker import rerank_documents
from app.generation.chain import generate_answer
from app.config import settings
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health_check():
    try:
        client = get_client()
        client.get_collections()
        qdrant_status = "connected"
    except Exception as e:
        qdrant_status = f"error: {str(e)}"

    return HealthResponse(
        status="ok",
        qdrant=qdrant_status,
        message="Hybrid RAG API is running"
    )

@router.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    try:
        # Load
        docs = load_documents(request.source)

        # Chunk
        chunks = chunk_documents(docs)
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks created from source")

        # Embed + store in Qdrant
        chunks, vectors = embed_chunks(chunks)
        store_chunks(chunks, vectors)

        # Rebuild BM25 index from full Qdrant collection
        load_bm25_from_qdrant()

        logger.info(f"Ingested {len(chunks)} chunks from {request.source}")

        return IngestResponse(
            message="Ingestion successful",
            chunks_created=len(chunks),
            source=request.source
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        # Vector search
        vector_results = vector_search(request.query, top_k=request.top_k)

        # BM25 search
        bm25_results = bm25_search(request.query, top_k=request.top_k)

        if not vector_results and not bm25_results:
            raise HTTPException(
                status_code=404,
                detail="No documents found. Please ingest documents first."
            )

        # Hybrid fusion
        fused = reciprocal_rank_fusion(vector_results, bm25_results)

        # Re-rank
        reranked = rerank_documents(request.query, fused, top_k=request.rerank_top_k)

        # Generate answer
        result = generate_answer(request.query, reranked)

        # Build retrieval scores for observability
        retrieval_scores = [
            {
                "chunk": doc.page_content[:120],
                "rerank_score": doc.metadata.get("rerank_score", 0),
                "hybrid_score": doc.metadata.get("hybrid_score", 0),
            }
            for doc in reranked
        ]

        return QueryResponse(
            query=request.query,
            answer=result["answer"],
            sources=result["sources"],
            chunks_used=result["chunks_used"],
            model=result["model"],
            retrieval_scores=retrieval_scores
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))