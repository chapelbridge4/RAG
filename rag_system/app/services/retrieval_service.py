from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SearchParams, PointStruct
from sentence_transformers import CrossEncoder
import numpy as np
from typing import List, Dict, Any, Optional
import asyncio
import uuid

class HybridRetrievalService:
    def __init__(self, qdrant_host: str, qdrant_port: int, collection_name: str):
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        
        # Cross-encoder for reranking
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
    def create_collection(self, vector_size: int):
        """Create Qdrant collection with optimization"""
        try:
            # Check if collection already exists
            collections = self.client.get_collections().collections
            if any(c.name == self.collection_name for c in collections):
                print(f"Collection '{self.collection_name}' already exists")
                return True
                
            # Create new collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                    hnsw_config={
                        "m": 16,
                        "ef_construct": 200
                    }
                )
            )
            print(f"Created collection '{self.collection_name}' successfully")
            return True
        except Exception as e:
            print(f"Error creating collection: {e}")
            return False
    
    def collection_exists(self) -> bool:
        """Check if the collection exists"""
        try:
            collections = self.client.get_collections().collections
            return any(c.name == self.collection_name for c in collections)
        except Exception as e:
            print(f"Error checking collection existence: {e}")
            return False
    
    async def ensure_collection_exists(self, vector_size: int) -> bool:
        """Ensure collection exists, create if not"""
        if not self.collection_exists():
            print(f"Collection '{self.collection_name}' doesn't exist, creating...")
            return self.create_collection(vector_size)
        return True

    async def add_documents(self, documents: List[Dict[str, Any]], embeddings: List[np.ndarray]):
        """Add documents with embeddings to Qdrant collection"""
        try:
            # Check if collection exists, create if not
            if not self.collection_exists():
                print(f"Collection '{self.collection_name}' doesn't exist, creating...")
                vector_size = len(embeddings[0]) if embeddings else 1024
                success = self.create_collection(vector_size)
                if not success:
                    print("Failed to create collection")
                    return False
            
            points = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding.tolist(),
                    payload={
                        "content": doc.get("content", ""),
                        "raw_content": doc.get("raw_content", ""),
                        "metadata": doc.get("metadata", {}),
                        "chunk_id": doc.get("chunk_id", i)
                    }
                )
                points.append(point)
            
            # Upload points in batches
            batch_size = 100
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
            
            print(f"Successfully added {len(documents)} documents to collection")
            return True
            
        except Exception as e:
            print(f"Error adding documents: {e}")
            return False
    
    async def search_with_reranking(
        self, 
        query_embedding: np.ndarray, 
        query_text: str,
        k: int = 20,
        rerank_k: int = 8
    ) -> List[Dict[str, Any]]:
        """Hybrid search with cross-encoder reranking"""
        
        # Check if collection exists
        if not self.collection_exists():
            print(f"Warning: Collection '{self.collection_name}' doesn't exist. No documents to search.")
            return []
        
        try:
            # Step 1: Dense vector search
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=k,
                search_params=SearchParams(hnsw_ef=128)
            )
        except Exception as e:
            print(f"Error during vector search: {e}")
            return []
        
        # Step 2: Cross-encoder reranking
        if search_result:
            query_doc_pairs = [(query_text, hit.payload['content']) for hit in search_result]
            scores = self.reranker.predict(query_doc_pairs)
            
            # Combine with original scores
            for i, hit in enumerate(search_result):
                hit.score = scores[i]  # Replace with cross-encoder score
            
            # Re-sort by cross-encoder scores
            search_result.sort(key=lambda x: x.score, reverse=True)
            
        return search_result[:rerank_k]