from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.api.routes import router
from app.retrieval.bm25_store import load_bm25_from_qdrant
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load BM25 index from existing Qdrant data
    logger.info("Starting up — loading BM25 index from Qdrant...")
    try:
        load_bm25_from_qdrant()
        logger.info("BM25 index ready")
    except Exception as e:
        logger.warning(f"BM25 load skipped (no data yet): {e}")
    yield
    logger.info("Shutting down")

app = FastAPI(
    title="Enterprise Hybrid RAG",
    description="Hybrid Search + Semantic Chunking + Cross-Encoder Re-ranking",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router, prefix="/api/v1")

@app.get("/")
def root():
    import os
    static_path = "static/index.html"
    abs_path = os.path.abspath(static_path)
    exists = os.path.exists(static_path)
    logger.info(f"Looking for static file at: {abs_path}, exists: {exists}")
    if exists:
        return FileResponse(static_path)
    return {"error": f"static/index.html not found at {abs_path}"}
