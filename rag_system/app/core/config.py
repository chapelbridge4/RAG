from pydantic_settings import BaseSettings
from typing import Optional, List
import os

class Settings(BaseSettings):
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = False
    
    # LLM Configuration
    MAIN_LLM_MODEL: str = "llama3.2:3b"
    PREPROCESSING_LLM_MODEL: str = "gemma2:2b"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    
    # Vector Database Configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "rag_documents"
    
    # Embedding Configuration
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    EMBEDDING_DIMENSION: int = 1024
    
    # Chunking Configuration
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 200
    CHUNKING_STRATEGY: str = "semantic"  # or "recursive"
    
    # Retrieval Configuration
    RETRIEVAL_K: int = 20  # Initial retrieval
    RERANK_K: int = 8      # Final chunks for LLM
    USE_HYBRID_SEARCH: bool = True
    
    # Performance Configuration
    ENABLE_CACHING: bool = True
    CACHE_TTL: int = 3600
    BATCH_SIZE: int = 16
    
    # Monitoring Configuration
    ENABLE_MONITORING: bool = True
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"

settings = Settings()