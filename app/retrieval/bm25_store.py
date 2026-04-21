from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from app.retrieval.vector_store import get_client
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Module-level cache
_bm25_index = None
_bm25_corpus: list[Document] = []

def _tokenize(text: str) -> list[str]:
    return text.lower().split()

def build_bm25_index(chunks: list[Document]):
    """Build BM25 index from a list of chunks."""
    global _bm25_index, _bm25_corpus
    _bm25_corpus = chunks
    tokenized = [_tokenize(chunk.page_content) for chunk in chunks]
    _bm25_index = BM25Okapi(tokenized)
    logger.info(f"BM25 index built with {len(chunks)} chunks")

def load_bm25_from_qdrant():
    """Rebuild BM25 index by loading all chunks from Qdrant."""
    client = get_client()
    results, _ = client.scroll(
        collection_name=settings.qdrant_collection,
        limit=10000,
        with_payload=True,
        with_vectors=False
    )
    chunks = [
        Document(
            page_content=r.payload["text"],
            metadata={"source": r.payload.get("source", "unknown")}
        )
        for r in results
    ]
    if chunks:
        build_bm25_index(chunks)
        logger.info(f"BM25 index loaded from Qdrant: {len(chunks)} chunks")
    else:
        logger.warning("No chunks found in Qdrant to build BM25 index")

def bm25_search(query: str, top_k: int = 10) -> list[Document]:
    """BM25 keyword search. Returns top_k chunks."""
    if _bm25_index is None:
        logger.warning("BM25 index not built yet. Call load_bm25_from_qdrant() first.")
        return []

    tokenized_query = _tokenize(query)
    scores = _bm25_index.get_scores(tokenized_query)

    # Get top_k indices sorted by score
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for i in top_indices:
        if scores[i] > 0:  # Only include if there's a keyword match
            doc = Document(
                page_content=_bm25_corpus[i].page_content,
                metadata={
                    **_bm25_corpus[i].metadata,
                    "score": float(scores[i]),
                    "retrieval_type": "bm25"
                }
            )
            results.append(doc)

    logger.info(f"BM25 search returned {len(results)} results")
    return results