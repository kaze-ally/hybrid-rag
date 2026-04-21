import pytest
from app.ingestion.chunker import chunk_text

def test_chunk_text():
    text = "This is a sample document that needs to be chunked into smaller pieces."
    expected_chunks = [
        "This is a sample document",
        "that needs to be chunked",
        "into smaller pieces."
    ]
    
    chunks = chunk_text(text, chunk_size=10)
    
    assert chunks == expected_chunks

def test_chunk_text_empty():
    text = ""
    expected_chunks = []
    
    chunks = chunk_text(text, chunk_size=10)
    
    assert chunks == expected_chunks

def test_chunk_text_long():
    text = "This is a long document that should be chunked into multiple parts based on the specified size."
    expected_chunks = [
        "This is a long document",
        "that should be chunked",
        "into multiple parts",
        "based on the specified",
        "size."
    ]
    
    chunks = chunk_text(text, chunk_size=10)
    
    assert chunks == expected_chunks