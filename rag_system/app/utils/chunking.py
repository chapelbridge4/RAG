from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np

class SentenceTransformerEmbedding:
    """Wrapper for SentenceTransformer to work with LangChain"""
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode([text])
        return embedding[0].tolist()

class AdvancedChunker:
    def __init__(self, strategy: str = "semantic", chunk_size: int = 800, overlap: int = 200):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        
        if strategy == "semantic":
            try:
                embedding_model = SentenceTransformerEmbedding("BAAI/bge-small-en-v1.5")
                self.chunker = SemanticChunker(
                    embedding_model,
                    breakpoint_threshold_type="percentile"
                )
            except Exception as e:
                print(f"Warning: Failed to initialize semantic chunker: {e}")
                print("Falling back to recursive chunker")
                self.strategy = "recursive"
                self.chunker = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=overlap,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
        else:
            self.chunker = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
    
    def chunk_document(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Chunk document and preserve metadata"""
        if self.strategy == "semantic":
            chunks = self.chunker.split_text(text)
        else:
            chunks = self.chunker.split_text(text)
        
        # Add context to chunks for better standalone understanding
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            # Add contextual information
            context = f"Document: {metadata.get('title', 'Unknown')}\n"
            if i > 0:
                context += f"Previous context: ...{chunks[i-1][-100:]}...\n"
            
            enhanced_chunk = {
                "content": context + chunk,
                "raw_content": chunk,
                "chunk_id": i,
                "metadata": {
                    **(metadata or {}),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            }
            enhanced_chunks.append(enhanced_chunk)
        
        return enhanced_chunks