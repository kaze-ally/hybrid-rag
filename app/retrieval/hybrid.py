from langchain_core.documents import Document
import logging

logger = logging.getLogger(__name__)

def reciprocal_rank_fusion(
    vector_results: list[Document],
    bm25_results: list[Document],
    k: int = 60,
    vector_weight: float = 0.7,
    bm25_weight: float = 0.3
) -> list[Document]:
    """
    Combine vector and BM25 results using Reciprocal Rank Fusion.
    k=60 is the standard constant that controls rank sensitivity.
    Higher vector_weight favors semantic similarity.
    """
    scores: dict[str, float] = {}
    docs_map: dict[str, Document] = {}

    # Score vector results
    for rank, doc in enumerate(vector_results):
        key = doc.page_content[:100]  # use content snippet as unique key
        rrf_score = vector_weight * (1 / (k + rank + 1))
        scores[key] = scores.get(key, 0) + rrf_score
        docs_map[key] = doc

    # Score BM25 results
    for rank, doc in enumerate(bm25_results):
        key = doc.page_content[:100]
        rrf_score = bm25_weight * (1 / (k + rank + 1))
        scores[key] = scores.get(key, 0) + rrf_score
        if key not in docs_map:
            docs_map[key] = doc

    # Sort by combined score
    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)

    fused = []
    for key in sorted_keys:
        doc = docs_map[key]
        doc.metadata["hybrid_score"] = round(scores[key], 6)
        doc.metadata["retrieval_type"] = "hybrid"
        fused.append(doc)

    logger.info(f"Hybrid fusion produced {len(fused)} unique results")
    return fused