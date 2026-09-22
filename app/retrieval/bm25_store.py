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
    return [w for w in text.lower().split() if w]


def build_bm25_index(chunks: list[Document]):
    """Build BM25 index from a list of chunks."""
    global _bm25_index, _bm25_corpus
    if not chunks:
        logger.warning("build_bm25_index called with empty chunks list")
        return

    _bm25_corpus = chunks
    tokenized = [_tokenize(chunk.page_content) for chunk in chunks]
    # Filter empty token lists if any
    tokenized = [t if t else ["<empty>"] for t in tokenized]
    _bm25_index = BM25Okapi(tokenized)
    logger.info(f"BM25 index built with {len(chunks)} chunks")


def load_bm25_from_qdrant():
    """Rebuild BM25 index by loading all chunks from Qdrant with error handling."""
    try:
        client = get_client()
        if not client.collection_exists(settings.qdrant_collection):
            logger.info(f"Collection '{settings.qdrant_collection}' does not exist yet. BM25 index empty.")
            return

        results, _ = client.scroll(
            collection_name=settings.qdrant_collection,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        chunks = [
            Document(
                page_content=r.payload.get("text", ""),
                metadata={"source": r.payload.get("source", "unknown")}
            )
            for r in results
            if r.payload and r.payload.get("text")
        ]
        if chunks:
            build_bm25_index(chunks)
            logger.info(f"BM25 index loaded from Qdrant: {len(chunks)} chunks")
        else:
            logger.info("No chunks found in Qdrant to build BM25 index")
    except Exception as e:
        logger.warning(f"Could not load BM25 index from Qdrant: {e}")


def bm25_search(query: str, top_k: int = 10) -> list[Document]:
    """BM25 keyword search. Returns top_k chunks."""
    global _bm25_index, _bm25_corpus

    if not query or not query.strip():
        return []

    # Attempt lazy load if not built yet
    if _bm25_index is None:
        load_bm25_from_qdrant()

    if _bm25_index is None or not _bm25_corpus:
        logger.info("BM25 index is not populated; skipping keyword search.")
        return []

    try:
        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []

        scores = _bm25_index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        token_set = set(tokenized_query)
        results = []
        for i in top_indices:
            doc_tokens = set(_tokenize(_bm25_corpus[i].page_content))
            has_term_match = bool(token_set & doc_tokens)
            if (scores[i] > 0 or (scores[i] >= 0 and has_term_match)) and i < len(_bm25_corpus):
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
    except Exception as e:
        logger.warning(f"BM25 search failed: {e}")
        return []