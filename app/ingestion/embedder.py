from langchain_core.documents import Document
from app.config import settings
import logging

logger = logging.getLogger(__name__)

BATCH_SIZE = 32

class LocalEmbedder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(settings.embedding_model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        clean_texts = [text.strip() or "empty" for text in texts]
        vectors = self.model.encode(
            clean_texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )[0].tolist()

def get_embedder() -> LocalEmbedder:
    return LocalEmbedder()

def embed_chunks(chunks: list[Document]) -> tuple[list[Document], list[list[float]]]:
    embedder = get_embedder()
    texts = [chunk.page_content for chunk in chunks]
    logger.info(f"Embedding {len(texts)} chunks...")
    vectors = embedder.embed_documents(texts)
    logger.info(f"Done. Dimension: {len(vectors[0])}")
    return chunks, vectors