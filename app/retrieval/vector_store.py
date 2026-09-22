from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_core.documents import Document
from app.config import settings
from app.ingestion.embedder import get_embedder
from pathlib import Path
import uuid
import logging

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 3072  # gemini-embedding-001 output dimension

_client: QdrantClient | None = None
_client_mode: str = "uninitialized"  # "remote", "embedded", "fallback"
_client_error: str | None = None


def _normalize_qdrant_url(url: str) -> str:
    """Normalize Qdrant URL to prevent common connection reset / protocol errors."""
    clean = url.strip()
    # If using Qdrant Cloud, HTTPS is required. Connecting via HTTP causes connection reset by peer.
    if ".qdrant.io" in clean:
        if clean.startswith("http://"):
            clean = "https://" + clean[7:]
        elif not clean.startswith("https://"):
            clean = "https://" + clean
    elif not clean.startswith("http://") and not clean.startswith("https://"):
        clean = "http://" + clean
    return clean


def get_qdrant_status() -> dict:
    """Return health details of current Qdrant connection."""
    return {
        "status": "connected" if _client is not None else ("error" if _client_error else "uninitialized"),
        "mode": _client_mode,
        "error": _client_error,
        "collection": settings.qdrant_collection,
    }


def get_client() -> QdrantClient:
    """
    Get or initialize Qdrant client.
    Supports remote servers (with automatic protocol normalization) and
    falls back cleanly to embedded local storage if the remote server is unreachable
    (e.g., when deployed on Render without an external Qdrant instance).
    """
    global _client, _client_mode, _client_error

    if _client is not None:
        return _client

    url = (settings.qdrant_url or "").strip()
    is_explicit_local = url.lower() in ("", "local", ":memory:", "embedded")

    if is_explicit_local:
        try:
            storage_path = Path(settings.qdrant_storage_path)
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            _client = QdrantClient(path=str(storage_path))
            _client_mode = "embedded"
            _client_error = None
            logger.info(f"Initialized embedded Qdrant storage at: {storage_path}")
            ensure_collection(_client)
            return _client
        except Exception as e:
            _client_error = str(e)
            logger.error(f"Failed to initialize embedded Qdrant: {e}")
            raise

    # Attempt remote connection
    normalized_url = _normalize_qdrant_url(url)
    try:
        logger.info(f"Connecting to remote Qdrant at: {normalized_url}")
        client_candidate = QdrantClient(
            url=normalized_url,
            api_key=settings.qdrant_api_key or None,
            timeout=10,
            check_compatibility=False
        )
        # Verify connectivity by checking collections
        client_candidate.get_collections()
        _client = client_candidate
        _client_mode = "remote"
        _client_error = None
        logger.info(f"Successfully connected to remote Qdrant ({normalized_url})")
        ensure_collection(_client)
        return _client
    except Exception as remote_err:
        _client_error = str(remote_err)
        logger.warning(
            f"Unable to connect to remote Qdrant at {normalized_url} ({remote_err})."
        )

        if settings.qdrant_auto_fallback:
            logger.info(
                f"Auto-fallback enabled. Switching to embedded Qdrant storage at {settings.qdrant_storage_path}"
            )
            try:
                storage_path = Path(settings.qdrant_storage_path)
                storage_path.parent.mkdir(parents=True, exist_ok=True)
                _client = QdrantClient(path=str(storage_path))
                _client_mode = "embedded (fallback)"
                logger.info(f"Fallback embedded Qdrant initialized successfully at: {storage_path}")
                ensure_collection(_client)
                return _client
            except Exception as embed_err:
                logger.error(f"Failed to initialize fallback embedded Qdrant: {embed_err}")
                _client = None
                raise RuntimeError(
                    f"Remote Qdrant failed ({remote_err}) and embedded fallback also failed ({embed_err})"
                ) from embed_err

        # If fallback not enabled, re-raise
        _client = None
        raise


def ensure_collection(client: QdrantClient | None = None):
    """Ensure the target collection exists with expected dimensions."""
    if client is None:
        client = get_client()

    try:
        exists = client.collection_exists(settings.qdrant_collection)
        if not exists:
            client.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
            )
            logger.info(f"Created Qdrant collection: {settings.qdrant_collection}")
        else:
            logger.info(f"Qdrant collection exists: {settings.qdrant_collection}")
    except Exception as e:
        logger.error(f"Error checking/creating collection '{settings.qdrant_collection}': {e}")
        raise


def store_chunks(chunks: list[Document], vectors: list[list[float]], batch_size: int = 50):
    """Store chunks and embeddings in Qdrant in safe batches."""
    if not chunks or not vectors:
        logger.warning("store_chunks called with empty chunks or vectors; skipping.")
        return

    if len(chunks) != len(vectors):
        raise ValueError(
            f"Chunk count ({len(chunks)}) does not match vector count ({len(vectors)})"
        )

    client = get_client()
    ensure_collection(client)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": chunk.page_content,
                "source": chunk.metadata.get("source", "unknown"),
                "chunk_id": chunk.metadata.get("chunk_id", 0),
            }
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    total = len(points)
    for i in range(0, total, batch_size):
        batch = points[i:i + batch_size]
        client.upsert(collection_name=settings.qdrant_collection, points=batch)
        logger.info(f"Upserted batch {i // batch_size + 1} ({len(batch)} points) into Qdrant")

    logger.info(f"Stored {total} chunks in Qdrant successfully ({_client_mode} mode)")


def vector_search(query: str, top_k: int = 10) -> list[Document]:
    """Search for relevant chunks via vector similarity with graceful error recovery."""
    try:
        client = get_client()
        if not client.collection_exists(settings.qdrant_collection):
            logger.warning(f"Collection '{settings.qdrant_collection}' does not exist yet.")
            return []

        embedder = get_embedder()
        query_vector = embedder.embed_query(query)

        search_result = client.query_points(
            collection_name=settings.qdrant_collection,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )

        results = search_result.points if hasattr(search_result, "points") else search_result
        docs = [
            Document(
                page_content=r.payload.get("text", ""),
                metadata={
                    "source": r.payload.get("source", "unknown"),
                    "score": getattr(r, "score", 0.0),
                    "retrieval_type": "vector"
                }
            )
            for r in results
            if r.payload and "text" in r.payload
        ]
        logger.info(f"Vector search returned {len(docs)} results")
        return docs
    except Exception as e:
        logger.warning(f"Vector search encountered an error: {e}. Falling back gracefully.")
        return []