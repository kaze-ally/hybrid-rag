# test_ingestion.py
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"  # disable tracing for this test

from app.ingestion.loader import load_documents
from app.ingestion.chunker import chunk_documents
from app.ingestion.embedder import embed_chunks
from app.retrieval.vector_store import store_chunks, vector_search, ensure_collection
from app.retrieval.bm25_store import build_bm25_index, bm25_search

# Step 1: Load
docs = load_documents("data/sample.txt")
print(f"✅ Loaded: {len(docs)} doc(s)")

# Step 2: Chunk
chunks = chunk_documents(docs)
print(f"✅ Chunks: {len(chunks)}")
for i, c in enumerate(chunks):
    print(f"   Chunk {i}: {c.page_content[:80]}...")

# Step 3: Embed + Store
chunks, vectors = embed_chunks(chunks)
print(f"✅ Embeddings generated. Dim: {len(vectors[0])}")
store_chunks(chunks, vectors)
print(f"✅ Stored in Qdrant")

# Step 4: BM25
build_bm25_index(chunks)
print(f"✅ BM25 index built with {len(chunks)} chunks")

# Step 5: Test searches
query = "how does hybrid search improve retrieval?"

v_results = vector_search(query, top_k=3)
print(f"\n🔍 Vector results ({len(v_results)}):")
for r in v_results:
    print(f"  score={r.metadata['score']:.3f} | {r.page_content[:100]}")

b_results = bm25_search(query, top_k=3)
print(f"\n🔍 BM25 results ({len(b_results)}):")
for r in b_results:
    print(f"  score={r.metadata['score']:.3f} | {r.page_content[:100]}")

# test hybrid + reranking
from app.retrieval.hybrid import reciprocal_rank_fusion
from app.retrieval.reranker import rerank_documents

print("\n🔀 Hybrid Fusion:")
fused = reciprocal_rank_fusion(v_results, b_results)
for doc in fused:
    print(f"  hybrid_score={doc.metadata['hybrid_score']} | {doc.page_content[:100]}")

print("\n🏆 After Re-ranking:")
reranked = rerank_documents(query, fused, top_k=2)
for doc in reranked:
    print(f"  rerank_score={doc.metadata['rerank_score']} | {doc.page_content[:100]}")


from app.generation.chain import generate_answer

print("\n💬 Final Answer:")
result = generate_answer(query, reranked)
print(f"  Answer: {result['answer']}")
print(f"  Sources: {result['sources']}")
print(f"  Chunks used: {result['chunks_used']}")