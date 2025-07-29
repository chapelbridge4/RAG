import ollama
from typing import Dict, Any, List, Tuple
import json
import asyncio

class SelfValidationService:
    def __init__(self, main_model: str = "llama3.2:3b", validator_model: str = "gemma2:2b"):
        self.main_model = main_model
        self.validator_model = validator_model
    
    async def validate_response(
        self, 
        query: str, 
        response: str, 
        context: List[str]
    ) -> Dict[str, Any]:
        """Comprehensive response validation using CRAG principles"""
        
        # 1. Groundedness check
        groundedness_score = await self._check_groundedness(response, context)
        
        # 2. Relevance assessment
        relevance_score = await self._assess_relevance(query, response)
        
        # 3. Factual accuracy validation
        accuracy_score = await self._validate_accuracy(response, context)
        
        # 4. Overall quality assessment
        overall_quality = (groundedness_score + relevance_score + accuracy_score) / 3
        
        validation_result = {
            "groundedness_score": groundedness_score,
            "relevance_score": relevance_score,
            "accuracy_score": accuracy_score,
            "overall_quality": overall_quality,
            "needs_correction": overall_quality < 0.7,
            "confidence_level": "high" if overall_quality > 0.8 else "medium" if overall_quality > 0.6 else "low"
        }
        
        return validation_result
    
    async def _check_groundedness(self, response: str, context: List[str]) -> float:
        """Check if response is grounded in provided context"""
        context_text = "\n\n".join(context)
        
        prompt = f"""Context: {context_text}

Response: {response}

Task: Evaluate if the response is fully supported by the provided context. Score from 0.0 to 1.0 where:
- 1.0 = Completely supported by context
- 0.5 = Partially supported  
- 0.0 = Not supported or contradicts context

Provide only a single number between 0.0 and 1.0:"""
        
        result = ollama.chat(
            model=self.validator_model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            score = float(result['message']['content'].strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5  # Default moderate score on parsing error
    
    async def _assess_relevance(self, query: str, response: str) -> float:
        """Assess response relevance to query"""
        prompt = f"""Query: {query}

Response: {response}

Task: Rate how well the response answers the query. Score from 0.0 to 1.0 where:
- 1.0 = Perfectly answers the query
- 0.5 = Partially answers the query
- 0.0 = Does not answer the query

Provide only a single number between 0.0 and 1.0:"""
        
        result = ollama.chat(
            model=self.validator_model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            score = float(result['message']['content'].strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    async def _validate_accuracy(self, response: str, context: List[str]) -> float:
        """Validate factual accuracy of response"""
        context_text = "\n\n".join(context)
        
        prompt = f"""Context: {context_text}

Response: {response}

Task: Check for factual errors or hallucinations in the response. Score from 0.0 to 1.0 where:
- 1.0 = No factual errors, all claims supported
- 0.5 = Minor inaccuracies or unsupported claims
- 0.0 = Significant factual errors or hallucinations

Provide only a single number between 0.0 and 1.0:"""
        
        result = ollama.chat(
            model=self.validator_model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        try:
            score = float(result['message']['content'].strip())
            return max(0.0, min(1.0, score))
        except:
            return 0.5
    
    async def auto_correct_response(
        self, 
        query: str, 
        original_response: str, 
        context: List[str],
        validation_result: Dict[str, Any]
    ) -> str:
        """Auto-correct response based on validation feedback"""
        
        if not validation_result["needs_correction"]:
            return original_response
        
        context_text = "\n\n".join(context)
        
        correction_prompt = f"""Query: {query}

Context: {context_text}

Original Response: {original_response}

Issues Found:
- Groundedness Score: {validation_result['groundedness_score']}
- Relevance Score: {validation_result['relevance_score']}
- Accuracy Score: {validation_result['accuracy_score']}

Task: Provide a corrected response that:
1. Is fully grounded in the provided context
2. Directly answers the query  
3. Contains no factual errors or hallucinations
4. Is clear and concise

Corrected Response:"""
        
        result = ollama.chat(
            model=self.main_model,
            messages=[{"role": "user", "content": correction_prompt}]
        )
        
        return result['message']['content']