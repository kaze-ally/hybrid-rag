import pytest
from langchain_core.documents import Document
from app.generation.chain import build_context, generate_answer


def test_build_context():
    docs = [
        Document(page_content="Content 1", metadata={"source": "doc1.txt", "rerank_score": 0.9}),
        Document(page_content="Content 2", metadata={"source": "doc2.txt", "rerank_score": 0.8}),
    ]
    context = build_context(docs)
    assert "Content 1" in context
    assert "Content 2" in context
    assert "doc1.txt" in context


def test_generate_answer_empty():
    result = generate_answer("test query", [])
    assert "answer" in result
    assert result["chunks_used"] == 0
    assert result["sources"] == []