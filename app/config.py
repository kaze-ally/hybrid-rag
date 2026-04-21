from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    qdrant_api_key: str = ""
    groq_api_key: str
    gemini_api_key: str
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "hybrid_rag"
    langsmith_api_key: str
    langsmith_project: str = "hybrid-rag"
    langchain_tracing_v2: str = "true"
    app_env: str = "development"

    embedding_model: str = "models/gemini-embedding-001"
    reranker_model: str = "llama-3.1-8b-instant"
    llm_model: str = "llama-3.3-70b-versatile"

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()