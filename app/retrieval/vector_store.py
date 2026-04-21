from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_core.documents import Document
from app.config import settings
from app.ingestion.embedder import get_embedder
import uuid
import logging

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 3072  # gemini-embedding-001 output dimension

_client = None

def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None
        )
        logger.info("Connected to Qdrant")
    return _client

def ensure_collection():
    client = get_client()
    collections = [c.name for c in client.get_collections().collections]
    if settings.qdrant_collection not in collections:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
        logger.info(f"Created collection: {settings.qdrant_collection}")
    else:
        logger.info(f"Collection exists: {settings.qdrant_collection}")

def store_chunks(chunks: list[Document], vectors: list[list[float]]):
    client = get_client()
    ensure_collection()
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
    client.upsert(collection_name=settings.qdrant_collection, points=points)
    logger.info(f"Stored {len(points)} chunks in Qdrant")

def vector_search(query: str, top_k: int = 10) -> list[Document]:
    client = get_client()
    embedder = get_embedder()
    query_vector = embedder.embed_query(query)
    
    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points

    docs = [
        Document(
            page_content=r.payload["text"],
            metadata={
                "source": r.payload.get("source", "unknown"),
                "score": r.score,
                "retrieval_type": "vector"
            }
        )
        for r in results
    ]
    logger.info(f"Vector search returned {len(docs)} results")
    return docs