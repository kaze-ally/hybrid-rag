import pytest
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_store import BM25Store
from app.retrieval.hybrid import HybridRetrieval

@pytest.fixture
def setup_retrieval():
    vector_store = VectorStore()
    bm25_store = BM25Store()
    hybrid_retrieval = HybridRetrieval(vector_store, bm25_store)
    return hybrid_retrieval

def test_hybrid_retrieval(setup_retrieval):
    hybrid = setup_retrieval
    query = "Sample query for testing"
    
    results = hybrid.retrieve(query)
    
    assert results is not None
    assert isinstance(results, list)
    assert len(results) > 0

def test_vector_store_integration(setup_retrieval):
    hybrid = setup_retrieval
    query = "Test vector store integration"
    
    vector_results = hybrid.vector_store.retrieve(query)
    
    assert vector_results is not None
    assert isinstance(vector_results, list)

def test_bm25_store_integration(setup_retrieval):
    hybrid = setup_retrieval
    query = "Test BM25 store integration"
    
    bm25_results = hybrid.bm25_store.retrieve(query)
    
    assert bm25_results is not None
    assert isinstance(bm25_results, list)