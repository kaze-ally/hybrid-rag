from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from app.ingestion.embedder import GeminiEmbedder
import logging

logger = logging.getLogger(__name__)

class GeminiEmbeddingsAdapter(Embeddings):
    """Adapts GeminiEmbedder to LangChain Embeddings interface."""
    def __init__(self):
        self._embedder = GeminiEmbedder()

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

def chunk_documents(docs: list[Document]) -> list[Document]:
    chunker = get_semantic_chunker()
    all_chunks = []

    for doc in docs:
        chunks = chunker.create_documents(
            texts=[doc.page_content],
            metadatas=[doc.metadata]
        )
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "total_chunks": len(chunks),
                "source": doc.metadata.get("source", "unknown")
            })
        all_chunks.extend(chunks)
        logger.info(f"Document split into {len(chunks)} semantic chunks")

    logger.info(f"Total chunks: {len(all_chunks)}")
    return all_chunks