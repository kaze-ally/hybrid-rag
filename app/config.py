from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    groq_api_key: str
    gemini_api_key: str = ""
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "hybrid_rag_local"
    langsmith_api_key: str = ""
    langsmith_project: str = "hybrid-rag"
    langchain_tracing_v2: str = "true"
    app_env: str = "development"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model: str = "openai/gpt-oss-120b"
    llm_model: str = "openai/gpt-oss-120b"

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()