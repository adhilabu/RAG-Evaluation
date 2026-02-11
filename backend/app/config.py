"""Application configuration management."""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # OpenAI
    openai_api_key: str
    
    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    
    # Neo4j (for future use)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""
    
    # Application
    environment: str = "development"
    log_level: str = "INFO"
    max_upload_size_mb: int = 50
    
    # Chunking Configuration
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 100
    summary_chunk_size: int = 15000
    summary_chunk_overlap: int = 500
    
    # Model Configuration
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
