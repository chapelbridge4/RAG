import ollama
from typing import Dict, Any
import asyncio

class HyDEService:
    def __init__(self, model: str = "gemma2:2b"):
        self.model = model
        
    async def generate_hypothetical_document(self, query: str) -> str:
        """Generate hypothetical document for better retrieval"""
        prompt = f"""Generate a hypothetical document that would perfectly answer this question: {query}

The document should be factual, detailed, and comprehensive. Write as if you're creating the ideal source document that contains the answer.

Hypothetical document:"""
        
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": 0.3,
                "max_tokens": 300
            }
        )
        
        return response['message']['content']
    
    async def hybrid_query_expansion(self, query: str) -> Dict[str, str]:
        """Generate multiple query variations"""
        variations_prompt = f"""Generate 3 different variations of this query, each focusing on different aspects:

Original query: {query}

Provide:
1. A more specific version
2. A broader contextual version  
3. A different perspective version

Format as:
Specific: [variation]
Broad: [variation]
Alternative: [variation]"""

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": variations_prompt}]
        )
        
        # Parse response (simplified - implement robust parsing)
        content = response['message']['content']
        variations = {
            "original": query,
            "hypothetical": await self.generate_hypothetical_document(query),
            "variations": content
        }
        
        return variations