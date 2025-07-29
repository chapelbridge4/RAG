"""
Sistema di logging avanzato per il RAG system
"""
import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
import uuid

class RAGLogger:
    """Logger specializzato per il sistema RAG con tracking dettagliato"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup logger principale
        self.logger = logging.getLogger("RAG_System")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # File handler per log dettagliati
        detailed_handler = logging.FileHandler(
            self.log_dir / "rag_detailed.log",
            encoding='utf-8'
        )
        detailed_handler.setLevel(logging.INFO)
        
        # File handler per query specifiche
        query_handler = logging.FileHandler(
            self.log_dir / "queries.log",
            encoding='utf-8'
        )
        query_handler.setLevel(logging.INFO)
        
        # Console handler per output immediato
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        detailed_handler.setFormatter(detailed_formatter)
        query_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(console_formatter)
        
        self.logger.addHandler(detailed_handler)
        self.logger.addHandler(query_handler)
        self.logger.addHandler(console_handler)
        
        # Query tracking
        self.current_query_id = None
        self.query_start_time = None
        
    def start_query(self, query: str, query_id: Optional[str] = None) -> str:
        """Inizia il tracking di una nuova query"""
        self.current_query_id = query_id or str(uuid.uuid4())[:8]
        self.query_start_time = time.time()
        
        self.logger.info(
            f"🔍 QUERY_START [{self.current_query_id}] | Query: '{query[:100]}{'...' if len(query) > 100 else ''}'"
        )
        
        return self.current_query_id
    
    def log_step(self, step: str, details: Dict[str, Any] = None, duration: float = None):
        """Log di un singolo step del processo RAG"""
        if not self.current_query_id:
            return
            
        elapsed = time.time() - self.query_start_time if self.query_start_time else 0
        duration_str = f" | Duration: {duration:.2f}s" if duration else ""
        details_str = f" | Details: {json.dumps(details, ensure_ascii=False)}" if details else ""
        
        self.logger.info(
            f"⚡ STEP [{self.current_query_id}] | {step} | Elapsed: {elapsed:.2f}s{duration_str}{details_str}"
        )
    
    def log_hyde_expansion(self, original_query: str, hypothetical_doc: str, variations: Dict[str, str]):
        """Log specifico per HyDE expansion"""
        self.log_step("HyDE_EXPANSION", {
            "original_query": original_query,
            "hypothetical_doc_length": len(hypothetical_doc),
            "variations_count": len(variations)
        })
    
    def log_retrieval(self, query_embedding_shape: tuple, retrieved_count: int, reranked_count: int):
        """Log specifico per retrieval"""
        self.log_step("RETRIEVAL", {
            "embedding_shape": query_embedding_shape,
            "retrieved_docs": retrieved_count,
            "reranked_docs": reranked_count
        })
    
    def log_compression(self, original_docs: int, compressed_docs: int, avg_compression_ratio: float):
        """Log specifico per contextual compression"""
        self.log_step("COMPRESSION", {
            "original_docs": original_docs,
            "compressed_docs": compressed_docs,
            "avg_compression_ratio": round(avg_compression_ratio, 3)
        })
    
    def log_generation(self, context_length: int, response_length: int, model: str):
        """Log specifico per generazione LLM"""
        self.log_step("GENERATION", {
            "context_length": context_length,
            "response_length": response_length,
            "model": model
        })
    
    def log_validation(self, scores: Dict[str, Any], corrections_made: int):
        """Log specifico per validazione"""
        self.log_step("VALIDATION", {
            "groundedness": round(scores.get('groundedness_score', 0), 3),
            "relevance": round(scores.get('relevance_score', 0), 3),
            "accuracy": round(scores.get('accuracy_score', 0), 3),
            "overall_quality": round(scores.get('overall_quality', 0), 3),
            "needs_correction": scores.get('needs_correction', False),
            "corrections_made": corrections_made
        })
    
    def end_query(self, final_response: str, total_corrections: int, metadata: Dict[str, Any]):
        """Conclude il tracking della query"""
        if not self.current_query_id or not self.query_start_time:
            return
            
        total_time = time.time() - self.query_start_time
        
        # Log finale con riepilogo completo
        summary = {
            "query_id": self.current_query_id,
            "total_time": round(total_time, 2),
            "response_length": len(final_response),
            "corrections_made": total_corrections,
            "metadata": metadata
        }
        
        self.logger.info(
            f"✅ QUERY_END [{self.current_query_id}] | Total: {total_time:.2f}s | "
            f"Corrections: {total_corrections} | Response: {len(final_response)} chars | "
            f"Retrieved: {metadata.get('documents_retrieved', 0)} docs"
        )
        
        # Salva query completa in file JSON per analisi
        self._save_query_json(summary, final_response)
        
        # Reset tracking
        self.current_query_id = None
        self.query_start_time = None
    
    def log_document_upload(self, filename: str, chunks_created: int, processing_time: float):
        """Log per upload documenti"""
        self.logger.info(
            f"📄 DOCUMENT_UPLOAD | File: {filename} | "
            f"Chunks: {chunks_created} | Time: {processing_time:.2f}s"
        )
    
    def log_error(self, error: Exception, context: str = ""):
        """Log errori con contesto"""
        context_str = f" | Context: {context}" if context else ""
        self.logger.error(
            f"❌ ERROR [{self.current_query_id or 'SYSTEM'}] | {type(error).__name__}: {str(error)}{context_str}"
        )
    
    def log_warning(self, message: str, details: Dict[str, Any] = None):
        """Log warning"""
        details_str = f" | Details: {json.dumps(details)}" if details else ""
        self.logger.warning(
            f"⚠️  WARNING [{self.current_query_id or 'SYSTEM'}] | {message}{details_str}"
        )
    
    def _save_query_json(self, summary: Dict[str, Any], response: str):
        """Salva query completa in formato JSON per analisi"""
        query_data = {
            **summary,
            "timestamp": datetime.now().isoformat(),
            "response_preview": response[:200] + "..." if len(response) > 200 else response
        }
        
        queries_file = self.log_dir / "queries_detailed.json"
        
        # Carica queries esistenti o crea nuovo file
        try:
            if queries_file.exists():
                with open(queries_file, 'r', encoding='utf-8') as f:
                    queries = json.load(f)
            else:
                queries = []
        except:
            queries = []
        
        queries.append(query_data)
        
        # Mantieni solo le ultime 1000 queries per evitare file troppo grandi
        if len(queries) > 1000:
            queries = queries[-1000:]
        
        with open(queries_file, 'w', encoding='utf-8') as f:
            json.dump(queries, f, ensure_ascii=False, indent=2)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Analizza i log per statistiche performance"""
        queries_file = self.log_dir / "queries_detailed.json"
        
        if not queries_file.exists():
            return {"message": "No query data available"}
        
        try:
            with open(queries_file, 'r', encoding='utf-8') as f:
                queries = json.load(f)
            
            if not queries:
                return {"message": "No query data available"}
            
            # Calcola statistiche
            times = [q.get('total_time', 0) for q in queries]
            corrections = [q.get('corrections_made', 0) for q in queries]
            
            stats = {
                "total_queries": len(queries),
                "avg_response_time": round(sum(times) / len(times), 2) if times else 0,
                "min_response_time": round(min(times), 2) if times else 0,
                "max_response_time": round(max(times), 2) if times else 0,
                "avg_corrections": round(sum(corrections) / len(corrections), 2) if corrections else 0,
                "queries_with_corrections": sum(1 for c in corrections if c > 0),
                "correction_rate": round(sum(1 for c in corrections if c > 0) / len(corrections) * 100, 1) if corrections else 0
            }
            
            return stats
            
        except Exception as e:
            return {"error": f"Failed to analyze logs: {str(e)}"}

# Istanza globale del logger
rag_logger = RAGLogger()