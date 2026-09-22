from google import genai
from langchain_core.documents import Document
from app.config import settings
import logging
import time

logger = logging.getLogger(__name__)

BATCH_SIZE = 40  # Keep within Gemini API per-request content limits
MAX_RETRIES = 3


class GeminiEmbedder:
    """Generate embeddings through Gemini without loading a local model."""

    def __init__(self):
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY in your environment."
            )
        try:
            self.client = genai.Client(api_key=settings.gemini_api_key)
            self.model = settings.embedding_model
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise RuntimeError(f"Could not initialize Gemini embedder client: {e}") from e

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self.client.models.embed_content(
                    model=self.model,
                    contents=batch
                )
                return [embedding.values for embedding in response.embeddings]
            except Exception as e:
                err_msg = str(e)
                logger.warning(
                    f"Gemini embedding attempt {attempt}/{MAX_RETRIES} failed: {err_msg}"
                )
                if attempt < MAX_RETRIES:
                    # Exponential backoff for rate limits or network glitches
                    sleep_time = 2 ** attempt
                    time.sleep(sleep_time)
                else:
                    raise RuntimeError(
                        f"Failed to generate embeddings after {MAX_RETRIES} attempts: {err_msg}"
                    ) from e
        return []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        clean_texts = [text.strip() or "empty" for text in texts]
        all_embeddings: list[list[float]] = []

        # Process in batches to prevent API batch limit errors
        for i in range(0, len(clean_texts), BATCH_SIZE):
            batch = clean_texts[i:i + BATCH_SIZE]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        cleaned = text.strip() or "empty"
        results = self._embed_batch([cleaned])
        if not results:
            raise RuntimeError("Gemini returned empty embedding for query.")
        return results[0]


def get_embedder() -> GeminiEmbedder:
    return GeminiEmbedder()


def embed_chunks(chunks: list[Document]) -> tuple[list[Document], list[list[float]]]:
    if not chunks:
        return [], []

    embedder = get_embedder()
    texts = [chunk.page_content for chunk in chunks]
    logger.info(f"Embedding {len(texts)} chunks via Gemini in batches of {BATCH_SIZE}...")
    vectors = embedder.embed_documents(texts)
    if vectors:
        logger.info(f"Done embedding. Dimension: {len(vectors[0])}")
    return chunks, vectors