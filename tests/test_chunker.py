import pytest
from langchain_core.documents import Document
from app.ingestion.chunker import chunk_documents


def test_chunk_documents():
    doc = Document(
        page_content="This is a test document that should be chunked into meaningful segments for retrieval.",
        metadata={"source": "test.txt"}
    )
    chunks = chunk_documents([doc])
    assert len(chunks) > 0
    assert chunks[0].page_content
    assert chunks[0].metadata["source"] == "test.txt"


def test_chunk_documents_empty():
    chunks = chunk_documents([])
    assert chunks == []


def test_chunk_documents_whitespace():
    doc = Document(page_content="   \n\n   ", metadata={"source": "empty.txt"})
    chunks = chunk_documents([doc])
    assert chunks == []