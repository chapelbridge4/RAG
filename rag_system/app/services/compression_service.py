import ollama
from typing import List, Dict, Any
import asyncio

class ContextualCompressionService:
    def __init__(self, model: str = "gemma2:2b"):
        self.model = model
    
    async def compress_documents(
        self, 
        query: str, 
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract and compress relevant information from retrieved documents"""
        
        compressed_docs = []
        
        for doc in documents:
            compression_prompt = f"""Query: {query}

Document content: {doc['content']}

Task: Extract ONLY the information from this document that is relevant to answering the query. Remove irrelevant details but keep the essential context. If the document doesn't contain relevant information, respond with "NOT_RELEVANT".

Relevant information:"""
            
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": compression_prompt}],
                options={"temperature": 0.1}
            )
            
            compressed_content = response['message']['content'].strip()
            
            if compressed_content != "NOT_RELEVANT":
                compressed_doc = {
                    **doc,
                    "compressed_content": compressed_content,
                    "compression_ratio": len(compressed_content) / len(doc['content'])
                }
                compressed_docs.append(compressed_doc)
        
        return compressed_docs