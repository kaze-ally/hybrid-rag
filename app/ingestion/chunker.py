from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from app.ingestion.embedder import LocalEmbedder
import logging

logger = logging.getLogger(__name__)

MAX_CHARS = 50_000  # Use fixed-size chunks for web pages and other long text.

class GeminiEmbeddingsAdapter(Embeddings):
    def __init__(self):
        self._embedder = LocalEmbedder()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)

def get_semantic_chunker():
    return SemanticChunker(
        embeddings=GeminiEmbeddingsAdapter(),
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=90
    )

def get_fallback_chunker():
    """Fast fixed-size chunker for very large documents."""
    return RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

def chunk_documents(docs: list[Document]) -> list[Document]:
    all_chunks = []

    for doc in docs:
        content = doc.page_content
        content_len = len(content)
        logger.info(f"Document size: {content_len:,} characters")

        # Keep chunking local; Gemini is called once per embedding batch below.
        if content_len > MAX_CHARS:
            logger.warning(f"Large document ({content_len:,} chars) — using fixed-size chunker")
        else:
            logger.info("Using fixed-size chunker")
        chunker = get_fallback_chunker()

        chunks = chunker.create_documents(
            texts=[content],
            metadatas=[doc.metadata]
        )

        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "total_chunks": len(chunks),
                "source": doc.metadata.get("source", "unknown")
            })

        all_chunks.extend(chunks)
        logger.info(f"Created {len(chunks)} chunks from document")

    logger.info(f"Total chunks: {len(all_chunks)}")
    # Filter out empty or whitespace-only chunks
    all_chunks = [c for c in all_chunks if c.page_content.strip()]
    logger.info(f"Total chunks after filtering: {len(all_chunks)}")
    return all_chunks