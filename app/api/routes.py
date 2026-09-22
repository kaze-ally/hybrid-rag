from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from app.api.schemas import (
    IngestRequest, IngestResponse, JobResponse,
    QueryRequest, QueryResponse, HealthResponse,
    DocumentSummary
)
from app.api.jobs import create_job, update_job, get_job
from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import embed_chunks
from app.retrieval.vector_store import store_chunks, vector_search, get_client, get_qdrant_status
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

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit


@router.get("/health", response_model=HealthResponse)
def health_check():
    try:
        status_info = get_qdrant_status()
        if status_info["status"] == "uninitialized":
            # Attempt initialization to determine true state
            get_client()
            status_info = get_qdrant_status()

        mode_suffix = f" ({status_info['mode']})" if status_info["mode"] else ""
        if status_info["status"] == "connected":
            qdrant_status = f"connected{mode_suffix}"
            app_status = "ok"
        else:
            qdrant_status = f"error: {status_info.get('error', 'unknown')}"
            app_status = "degraded"
        qdrant_mode = status_info.get("mode")
    except Exception as e:
        qdrant_status = f"error: {str(e)}"
        app_status = "degraded"
        qdrant_mode = "error"

    return HealthResponse(
        status=app_status,
        qdrant=qdrant_status,
        message="Hybrid RAG API is running",
        qdrant_mode=qdrant_mode
    )


@router.get("/documents", response_model=list[DocumentSummary])
def list_documents():
    """Return persisted document sources and their chunk counts without raising 500 errors."""
    try:
        client = get_client()
        if not client.collection_exists(settings.qdrant_collection):
            return []

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
    except Exception as e:
        logger.warning(f"Failed to scroll documents from Qdrant: {e}")
        return []


# ── Background ingestion logic ──────────────────────────────────────────
def _run_ingestion(job_id: str, source: str, filename: str, tmp_path: str | None = None):
    """Runs in background. Updates job store as it progresses."""
    try:
        update_job(job_id, status="processing", message="Loading document...")
        docs = load_documents(source)
        if not docs:
            update_job(job_id, status="error", message="No text content could be extracted from document.")
            return

        for doc in docs:
            doc.metadata["source"] = filename

        update_job(job_id, message="Chunking document...")
        chunks = chunk_documents(docs)
        if not chunks:
            update_job(job_id, status="error", message="Document produced no chunks after filtering.")
            return

        update_job(job_id, message=f"Embedding {len(chunks)} chunk(s) via Gemini...")
        chunks, vectors = embed_chunks(chunks)
        if not vectors:
            update_job(job_id, status="error", message="Embedding generation failed.")
            return

        update_job(job_id, message="Storing chunks in vector database...")
        store_chunks(chunks, vectors)

        update_job(job_id, message="Updating BM25 search index...")
        try:
            load_bm25_from_qdrant()
        except Exception as e:
            logger.warning(f"BM25 index refresh skipped: {e}")

        update_job(
            job_id,
            status="done",
            message=f"Successfully ingested {len(chunks)} chunks from {filename}",
            chunks_created=len(chunks),
            source=filename
        )
        logger.info(f"Job {job_id} complete: {len(chunks)} chunks from {filename}")

    except Exception as e:
        err_msg = str(e) or "An unexpected error occurred during ingestion."
        logger.error(f"Job {job_id} failed: {err_msg}", exc_info=True)
        update_job(job_id, status="error", message=err_msg)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {tmp_path}: {e}")


# ── File upload ──────────────────────────────────────────────────────────
@router.post("/ingest/upload", response_model=JobResponse)
async def ingest_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    filename = file.filename or "upload"
    # Basic sanitize
    filename = os.path.basename(filename)
    ext = os.path.splitext(filename)[1].lower()

    if ext not in [".pdf", ".txt", ".docx"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Only .pdf, .txt, and .docx are supported."
        )

    tmp_path = None
    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES // (1024*1024)}MB."
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        job_id = create_job()
        background_tasks.add_task(_run_ingestion, job_id, tmp_path, filename, tmp_path)

        return JobResponse(
            job_id=job_id,
            status="pending",
            message="Upload received. Ingestion queued in background.",
            source=filename
        )
    except HTTPException:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        logger.error(f"Failed to receive upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process upload: {str(e)}")


# ── URL ingest ───────────────────────────────────────────────────────────
@router.post("/ingest", response_model=JobResponse)
def ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    source = request.source.strip()
    if not source:
        raise HTTPException(status_code=400, detail="Source URL or text cannot be empty.")

    job_id = create_job()
    background_tasks.add_task(_run_ingestion, job_id, source, source, None)
    return JobResponse(
        job_id=job_id,
        status="pending",
        message="Ingestion queued in background.",
        source=source
    )


# ── Job status ───────────────────────────────────────────────────────────
@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job_status(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job ID not found or expired.")
    return JobResponse(job_id=job_id, **job)


# ── Query ────────────────────────────────────────────────────────────────
@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    query_text = request.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text cannot be empty.")

    try:
        vector_results = []
        try:
            vector_results = vector_search(query_text, top_k=request.top_k or 5)
        except Exception as e:
            logger.warning(f"Vector search failed during query: {e}")

        bm25_results = []
        try:
            bm25_results = bm25_search(query_text, top_k=request.top_k or 5)
        except Exception as e:
            logger.warning(f"BM25 search failed during query: {e}")

        if not vector_results and not bm25_results:
            raise HTTPException(
                status_code=404,
                detail="No matching documents found. Please ensure documents have been ingested."
            )

        fused = reciprocal_rank_fusion(vector_results, bm25_results)
        reranked = rerank_documents(query_text, fused, top_k=request.rerank_top_k or 3)
        result = generate_answer(query_text, reranked)

        retrieval_scores = [
            {
                "chunk": doc.page_content[:120],
                "rerank_score": doc.metadata.get("rerank_score", 0),
                "hybrid_score": doc.metadata.get("hybrid_score", 0),
            }
            for doc in reranked
        ]

        return QueryResponse(
            query=query_text,
            answer=result["answer"],
            sources=result["sources"],
            chunks_used=result["chunks_used"],
            model=result["model"],
            retrieval_scores=retrieval_scores
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")