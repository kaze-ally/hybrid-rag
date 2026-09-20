from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from app.api.schemas import (
    IngestRequest, IngestResponse, JobResponse,
    QueryRequest, QueryResponse, HealthResponse
    , DocumentSummary
)
from app.api.jobs import create_job, update_job, get_job
from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import embed_chunks
from app.retrieval.vector_store import store_chunks, vector_search, get_client
from app.retrieval.bm25_store import bm25_search, load_bm25_from_qdrant
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.reranker import rerank_documents
from app.generation.chain import generate_answer
from app.config import settings
import tempfile
import os
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


@router.get("/documents", response_model=list[DocumentSummary])
def list_documents():
    """Return persisted document sources and their chunk counts."""
    client = get_client()
    results, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        limit=10000,
        with_payload=True,
        with_vectors=False
    )
    counts: dict[str, int] = {}
    for point in results:
        source = (point.payload or {}).get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1
    return [DocumentSummary(source=source, chunks=chunks) for source, chunks in sorted(counts.items())]


# ── Background ingestion logic ──────────────────────────────────────────
def _run_ingestion(job_id: str, source: str, filename: str, tmp_path: str | None = None):
    """Runs in background. Updates job store as it progresses."""
    try:
        update_job(job_id, status="processing", message="Loading document...")
        docs = load_documents(source)
        for doc in docs:
            doc.metadata["source"] = filename

        update_job(job_id, message="Chunking document semantically...")
        chunks = chunk_documents(docs)
        if not chunks:
            update_job(job_id, status="error", message="No content extracted from document")
            return

        update_job(job_id, message=f"Embedding {len(chunks)} chunks (this may take a minute)...")
        chunks, vectors = embed_chunks(chunks)

        update_job(job_id, message="Storing in Qdrant...")
        store_chunks(chunks, vectors)
        try:
            load_bm25_from_qdrant()
        except Exception as e:
            logger.warning(f"BM25 refresh skipped after storage: {e}")

        update_job(
            job_id,
            status="done",
            message="Ingestion complete",
            chunks_created=len(chunks),
            source=filename
        )
        logger.info(f"Job {job_id} complete: {len(chunks)} chunks from {filename}")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        update_job(job_id, status="error", message=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── File upload ──────────────────────────────────────────────────────────
@router.post("/ingest/upload", response_model=JobResponse)
async def ingest_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in [".pdf", ".txt", ".docx"]:
        raise HTTPException(status_code=400, detail=f"Unsupported: {ext}. Use .pdf, .txt, .docx")

    # Save to temp file synchronously (fast)
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    job_id = create_job()
    background_tasks.add_task(_run_ingestion, job_id, tmp_path, filename, tmp_path)

    return JobResponse(
        job_id=job_id,
        status="pending",
        message="Ingestion started in background",
        source=filename
    )


# ── URL ingest ───────────────────────────────────────────────────────────
@router.post("/ingest", response_model=JobResponse)
def ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    job_id = create_job()
    background_tasks.add_task(_run_ingestion, job_id, request.source, request.source, None)
    return JobResponse(
        job_id=job_id,
        status="pending",
        message="Ingestion started in background",
        source=request.source
    )


# ── Job status ───────────────────────────────────────────────────────────
@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(job_id=job_id, **job)


# ── Query ────────────────────────────────────────────────────────────────
@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        vector_results = vector_search(request.query, top_k=request.top_k)
        bm25_results = bm25_search(request.query, top_k=request.top_k)

        if not vector_results and not bm25_results:
            raise HTTPException(
                status_code=404,
                detail="No documents found. Please ingest documents first."
            )

        fused = reciprocal_rank_fusion(vector_results, bm25_results)
        reranked = rerank_documents(request.query, fused, top_k=request.rerank_top_k)
        result = generate_answer(request.query, reranked)

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