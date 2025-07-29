from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional
import asyncio
import hashlib
import json
from functools import lru_cache

class EmbeddingService:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model = SentenceTransformer(model_name)
        self.cache = {}  # In-memory cache, use Redis for production
        
    @lru_cache(maxsize=1000)  # Cache for repeated queries
    def _get_embedding_cached(self, text_hash: str, text: str) -> np.ndarray:
        """Cached embedding computation"""
        return self.model.encode(text, normalize_embeddings=True)
    
    def encode(self, texts: List[str], batch_size: int = 16) -> List[np.ndarray]:
        """Batch encode texts with caching"""
        embeddings = []
        
        for text in texts:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            embedding = self._get_embedding_cached(text_hash, text)
            embeddings.append(embedding)
            
        return embeddings
    
    async def encode_async(self, texts: List[str]) -> List[np.ndarray]:
        """Async wrapper for embedding computation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.encode, texts)