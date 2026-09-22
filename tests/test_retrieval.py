import pytest
from langchain_core.documents import Document
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.bm25_store import build_bm25_index, bm25_search


def test_reciprocal_rank_fusion():
    v_docs = [
        Document(page_content="Common doc", metadata={"source": "doc1"}),
        Document(page_content="Vector only", metadata={"source": "doc2"}),
    ]
    b_docs = [
        Document(page_content="BM25 only", metadata={"source": "doc3"}),
        Document(page_content="Common doc", metadata={"source": "doc1"}),
    ]
    fused = reciprocal_rank_fusion(v_docs, b_docs)
    assert len(fused) == 3
    # Common doc should have highest rank because it appears in both
    assert fused[0].page_content == "Common doc"
    assert "hybrid_score" in fused[0].metadata


def test_bm25_search():
    chunks = [
        Document(page_content="Machine learning is fascinating and powerful.", metadata={"source": "ml.txt"}),
        Document(page_content="Cooking pasta requires boiling water and salt.", metadata={"source": "cook.txt"}),
    ]
    build_bm25_index(chunks)
    results = bm25_search("machine learning", top_k=2)
    assert len(results) >= 1
    assert "machine learning" in results[0].page_content.lower()