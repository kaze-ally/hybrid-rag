from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from app.api.routes import router
from pathlib import Path
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up...")
    logger.info(f"BASE_DIR: {BASE_DIR}")
    logger.info(f"static/index.html exists: {(BASE_DIR / 'static' / 'index.html').exists()}")
    logger.info("BM25 index will load after the first successful ingestion")
    yield
    logger.info("Shutting down")

app = FastAPI(
    title="Enterprise Hybrid RAG",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(router, prefix="/api/v1")

# Mount static files
static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
def root():
    static_file = BASE_DIR / "static" / "index.html"
    logger.info(f"Serving static file from: {static_file}, exists: {static_file.exists()}")
    if static_file.exists():
        return FileResponse(str(static_file))
    return {"project": "Enterprise Hybrid RAG", "docs": "/docs", "health": "/api/v1/health"}