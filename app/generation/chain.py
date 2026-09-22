from groq import Groq
from langchain_core.documents import Document
from app.config import settings
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based strictly on the provided context.

Rules:
- Answer ONLY using the context provided below
- If the context doesn't contain enough information, say "I don't have enough information in the provided documents to answer this."
- Be concise and accurate
- Cite which part of the context supports your answer"""


def get_groq_client() -> Groq:
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY is not configured.")
    return Groq(api_key=settings.groq_api_key)


def build_context(docs: list[Document]) -> str:
    """Format reranked docs into a context string for the LLM."""
    context_parts = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        score = doc.metadata.get("rerank_score", 0)
        context_parts.append(
            f"[Chunk {i+1} | source: {source} | relevance: {score}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(context_parts)


def generate_answer(query: str, reranked_docs: list[Document]) -> dict:
    """
    Generate answer from reranked docs using Groq LLM.
    Returns answer + metadata about what was used with robust fallback on API error.
    """
    if not reranked_docs:
        return {
            "answer": "No relevant documents found to answer your query.",
            "sources": [],
            "chunks_used": 0,
            "model": settings.llm_model
        }

    sources = list(set(
        doc.metadata.get("source", "unknown") for doc in reranked_docs
    ))

    try:
        client = get_groq_client()
        context = build_context(reranked_docs)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]

        logger.info(f"Generating answer with {len(reranked_docs)} chunks using model {settings.llm_model}...")

        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=messages,
            max_tokens=1024,
            temperature=0.1
        )

        answer = response.choices[0].message.content or "No response generated."
        logger.info("Answer generated successfully")

        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(reranked_docs),
            "model": settings.llm_model
        }

    except Exception as e:
        logger.error(f"Error generating answer with Groq: {e}")
        return {
            "answer": f"Unable to generate response from model ({e}). However, {len(reranked_docs)} relevant context chunk(s) were found.",
            "sources": sources,
            "chunks_used": len(reranked_docs),
            "model": settings.llm_model
        }