from google import genai
from google.genai import types
from langchain_core.documents import Document
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class GeminiEmbedder:
    """Wrapper around new google.genai SDK for embeddings."""
    def __init__(self):
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.embedding_model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=texts
        )
        return [e.values for e in response.embeddings]

    def embed_query(self, text: str) -> list[float]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=[text]
        )
        return response.embeddings[0].values

def get_embedder() -> GeminiEmbedder:
    return GeminiEmbedder()

def embed_chunks(chunks: list[Document]) -> tuple[list[Document], list[list[float]]]:
    embedder = get_embedder()
    texts = [chunk.page_content for chunk in chunks]
    logger.info(f"Embedding {len(texts)} chunks via Gemini...")
    vectors = embedder.embed_documents(texts)
    logger.info(f"Done. Dimension: {len(vectors[0])}")
    return chunks, vectors