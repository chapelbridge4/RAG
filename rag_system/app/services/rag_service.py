from .embedding_service import EmbeddingService
from .retrieval_service import HybridRetrievalService
from .hyde_service import HyDEService
from .compression_service import ContextualCompressionService
from .validation_service import SelfValidationService
from ..utils.rag_logger import rag_logger
from ..utils.prometheus_metrics import prometheus_metrics
import ollama
import asyncio
import time
from typing import Dict, Any, List

class AdvancedRAGService:
    def __init__(self, config):
        self.config = config
        self.embedding_service = EmbeddingService(config.EMBEDDING_MODEL)
        self.retrieval_service = HybridRetrievalService(
            config.QDRANT_HOST, 
            config.QDRANT_PORT, 
            config.QDRANT_COLLECTION_NAME
        )
        self.hyde_service = HyDEService(config.PREPROCESSING_LLM_MODEL)
        self.compression_service = ContextualCompressionService(config.PREPROCESSING_LLM_MODEL)
        self.validation_service = SelfValidationService(config.MAIN_LLM_MODEL, config.PREPROCESSING_LLM_MODEL)
    
    async def process_query(self, query: str, use_hyde: bool = True, max_corrections: int = 2) -> Dict[str, Any]:
        """Main RAG pipeline with advanced features"""
        
        # Start query tracking
        query_id = rag_logger.start_query(query)
        query_start_time = time.time()
        prometheus_metrics.start_query()
        
        try:
            # Step 1: Query expansion with HyDE
            step_start = time.time()
            if use_hyde:
                query_variations = await self.hyde_service.hybrid_query_expansion(query)
                search_query = query_variations["hypothetical"]
                rag_logger.log_hyde_expansion(query, search_query, query_variations)
            else:
                search_query = query
                rag_logger.log_step("QUERY_DIRECT", {"original_query": query})
            
            rag_logger.log_step("HYDE_EXPANSION", duration=time.time() - step_start)
            
            # Step 2: Embedding and retrieval
            step_start = time.time()
            query_embedding = await self.embedding_service.encode_async([search_query])
            embedding_duration = time.time() - step_start
            rag_logger.log_step("EMBEDDING_GENERATION", {"embedding_shape": query_embedding[0].shape}, embedding_duration)
            
            step_start = time.time()
            retrieved_docs = await self.retrieval_service.search_with_reranking(
                query_embedding[0], 
                search_query,
                k=self.config.RETRIEVAL_K,
                rerank_k=self.config.RERANK_K
            )
            retrieval_duration = time.time() - step_start
            rag_logger.log_retrieval(query_embedding[0].shape, len(retrieved_docs), min(len(retrieved_docs), self.config.RERANK_K))
            rag_logger.log_step("RETRIEVAL", duration=retrieval_duration)
            
            # Record Prometheus metrics for retrieval
            prometheus_metrics.record_retrieval(len(retrieved_docs))
            
            # Step 3: Contextual compression
            step_start = time.time()
            if retrieved_docs:
                doc_contents = [{"content": doc.payload["content"]} for doc in retrieved_docs]
                compressed_docs = await self.compression_service.compress_documents(query, doc_contents)
                context = [doc["compressed_content"] for doc in compressed_docs]
                
                # Calculate compression ratio
                total_original = sum(len(doc["content"]) for doc in doc_contents)
                total_compressed = sum(len(content) for content in context)
                compression_ratio = total_compressed / total_original if total_original > 0 else 0
                
                rag_logger.log_compression(len(doc_contents), len(context), compression_ratio)
            else:
                context = []
                rag_logger.log_step("COMPRESSION", {"result": "no_documents_to_compress"})
            
            compression_duration = time.time() - step_start
            rag_logger.log_step("COMPRESSION", duration=compression_duration)
            
            # Step 4: Generate initial response
            step_start = time.time()
            response = await self._generate_response(query, context)
            generation_duration = time.time() - step_start
            
            context_length = sum(len(c) for c in context)
            rag_logger.log_generation(context_length, len(response), self.config.MAIN_LLM_MODEL)
            rag_logger.log_step("GENERATION", duration=generation_duration)
            
            # Step 5: Self-validation and correction loop
            corrections_made = 0
            validation_result = None
            
            while corrections_made < max_corrections:
                step_start = time.time()
                validation_result = await self.validation_service.validate_response(query, response, context)
                validation_duration = time.time() - step_start
                
                rag_logger.log_validation(validation_result, corrections_made)
                
                if not validation_result["needs_correction"]:
                    rag_logger.log_step("VALIDATION_PASSED", duration=validation_duration)
                    break
                    
                # Auto-correct response
                step_start = time.time()
                corrected_response = await self.validation_service.auto_correct_response(
                    query, response, context, validation_result
                )
                correction_duration = time.time() - step_start
                
                # Check if correction improved quality
                new_validation = await self.validation_service.validate_response(query, corrected_response, context)
                if new_validation["overall_quality"] > validation_result["overall_quality"]:
                    response = corrected_response
                    validation_result = new_validation
                    rag_logger.log_step("CORRECTION_APPLIED", 
                                      {"improvement": new_validation["overall_quality"] - validation_result["overall_quality"]}, 
                                      correction_duration)
                else:
                    rag_logger.log_step("CORRECTION_REJECTED", 
                                      {"quality_diff": new_validation["overall_quality"] - validation_result["overall_quality"]})
                
                corrections_made += 1
            
            # Prepare final result
            final_result = {
                "query": query,
                "response": response,
                "context_used": context,
                "validation_scores": validation_result,
                "corrections_made": corrections_made,
                "retrieval_metadata": {
                    "documents_retrieved": len(retrieved_docs),
                    "documents_after_compression": len(context),
                    "hyde_used": use_hyde
                }
            }
            
            # Record Prometheus metrics
            total_duration = time.time() - query_start_time
            prometheus_metrics.end_query(total_duration, "success")
            prometheus_metrics.record_validation_scores(validation_result)
            prometheus_metrics.record_corrections(corrections_made)
            
            # End query tracking
            rag_logger.end_query(response, corrections_made, final_result["retrieval_metadata"])
            
            return final_result
            
        except Exception as e:
            # Record error in Prometheus
            total_duration = time.time() - query_start_time
            prometheus_metrics.end_query(total_duration, "error")
            
            rag_logger.log_error(e, "process_query")
            raise
    
    async def _generate_response(self, query: str, context: List[str]) -> str:
        """Generate response using main LLM"""
        context_text = "\n\n".join(context) if context else "No relevant context found."
        
        prompt = f"""Context: {context_text}

Question: {query}

Instructions: Answer the question based on the provided context. If the context doesn't contain enough information, clearly state this limitation. Be concise but comprehensive.

Answer:"""
        
        response = ollama.chat(
            model=self.config.MAIN_LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1}
        )
        
        return response['message']['content']