from groq import Groq
from langchain_core.documents import Document
from app.config import settings
import json
import re
import logging

logger = logging.getLogger(__name__)

RERANK_PROMPT = """You are a relevance scoring engine.
Given a query and a text chunk, return ONLY a JSON object with a single key "score" 
containing a float between 0.0 and 1.0 representing how relevant the chunk is to the query.
1.0 = perfectly relevant, 0.0 = completely irrelevant.
Return ONLY the JSON. No explanation. No markdown.

Query: {query}
Chunk: {chunk}"""


def get_groq_client() -> Groq | None:
    if not settings.groq_api_key:
        return None
    try:
        return Groq(api_key=settings.groq_api_key)
    except Exception as e:
        logger.warning(f"Failed to initialize Groq client for reranking: {e}")
        return None


def _parse_rerank_score(raw_text: str) -> float:
    """Safely extract float relevance score from LLM output, handling markdown fences and prose."""
    cleaned = raw_text.strip()
    # Strip markdown fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "score" in parsed:
            return float(parsed["score"])
    except Exception:
        pass

    # Regex search for {"score": <number>}
    match = re.search(r'["\']?score["\']?\s*:\s*([0-9]*\.?[0-9]+)', cleaned)
    if match:
        try:
            val = float(match.group(1))
            return max(0.0, min(1.0, val))
        except ValueError:
            pass

    # Fallback search for any isolated float between 0 and 1
    match_num = re.search(r"\b(0\.\d+|1\.0|0|1)\b", cleaned)
    if match_num:
        try:
            return float(match_num.group(1))
        except ValueError:
            pass

    return 0.0


def rerank_documents(
    query: str,
    docs: list[Document],
    top_k: int = 3
) -> list[Document]:
    """
    Re-rank documents using Groq LLM as a cross-encoder.
    Scores each chunk individually, then returns top_k by score.
    Falls back gracefully to preserving original order if re-ranking fails.
    """
    if not docs:
        return []

    client = get_groq_client()
    if not client:
        logger.warning("Groq client not available; returning unranked top documents.")
        return docs[:top_k]

    scored = []
    for doc in docs:
        try:
            response = client.chat.completions.create(
                model=settings.reranker_model,
                messages=[{
                    "role": "user",
                    "content": RERANK_PROMPT.format(
                        query=query,
                        chunk=doc.page_content[:500]  # limit chunk size
                    )
                }],
                max_tokens=20,
                temperature=0.0
            )
            raw = response.choices[0].message.content or ""
            score = _parse_rerank_score(raw)
        except Exception as e:
            logger.warning(f"Rerank scoring failed for chunk, defaulting to 0: {e}")
            score = 0.0

        doc.metadata["rerank_score"] = round(score, 4)
        scored.append((score, doc))

    # Sort by rerank score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    reranked = [doc for _, doc in scored[:top_k]]

    logger.info(f"Re-ranking complete. Top {top_k} chunks selected from {len(docs)}")
    return reranked