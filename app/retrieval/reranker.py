from groq import Groq
from langchain_core.documents import Document
from app.config import settings
import json
import logging

logger = logging.getLogger(__name__)

client = Groq(api_key=settings.groq_api_key)

RERANK_PROMPT = """You are a relevance scoring engine.
Given a query and a text chunk, return ONLY a JSON object with a single key "score" 
containing a float between 0.0 and 1.0 representing how relevant the chunk is to the query.
1.0 = perfectly relevant, 0.0 = completely irrelevant.
Return ONLY the JSON. No explanation. No markdown.

Query: {query}
Chunk: {chunk}"""

def rerank_documents(
    query: str,
    docs: list[Document],
    top_k: int = 3
) -> list[Document]:
    """
    Re-rank documents using Groq LLM as a cross-encoder.
    Scores each chunk individually, then returns top_k by score.
    """
    if not docs:
        return []

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
            raw = response.choices[0].message.content.strip()
            score = json.loads(raw).get("score", 0.0)
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