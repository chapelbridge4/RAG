"""
Prometheus metrics per il RAG system
"""
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time
from typing import Dict, Any

# Metriche Prometheus
rag_queries_total = Counter(
    'rag_queries_total', 
    'Total number of RAG queries processed',
    ['status']  # success, error
)

rag_query_duration_seconds = Histogram(
    'rag_query_duration_seconds', 
    'RAG query processing time in seconds',
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, float('inf')]
)

rag_validation_scores = Histogram(
    'rag_validation_scores', 
    'RAG validation scores distribution',
    ['metric_type'],  # groundedness, relevance, accuracy, overall
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

rag_active_queries = Gauge(
    'rag_active_queries', 
    'Currently active RAG queries'
)

rag_corrections_total = Counter(
    'rag_corrections_total', 
    'Total number of response corrections made'
)

rag_documents_uploaded_total = Counter(
    'rag_documents_uploaded_total',
    'Total number of documents uploaded',
    ['status']  # success, error
)

rag_document_chunks_total = Counter(
    'rag_document_chunks_total',
    'Total number of document chunks created'
)

rag_embeddings_generated_total = Counter(
    'rag_embeddings_generated_total',
    'Total number of embeddings generated'
)

rag_retrieval_documents_found = Histogram(
    'rag_retrieval_documents_found',
    'Number of documents found during retrieval',
    buckets=[0, 1, 2, 5, 10, 20, 50, 100, float('inf')]
)

class PrometheusMetrics:
    """Classe per gestire metriche Prometheus"""
    
    def __init__(self):
        self.active_queries_count = 0
    
    def start_query(self):
        """Incrementa contatore query attive"""
        self.active_queries_count += 1
        rag_active_queries.set(self.active_queries_count)
    
    def end_query(self, duration: float, status: str = "success"):
        """Registra fine query con metriche"""
        self.active_queries_count = max(0, self.active_queries_count - 1)
        rag_active_queries.set(self.active_queries_count)
        
        rag_queries_total.labels(status=status).inc()
        rag_query_duration_seconds.observe(duration)
    
    def record_validation_scores(self, scores: Dict[str, Any]):
        """Registra punteggi di validazione"""
        rag_validation_scores.labels(metric_type='groundedness').observe(
            scores.get('groundedness_score', 0)
        )
        rag_validation_scores.labels(metric_type='relevance').observe(
            scores.get('relevance_score', 0)
        )
        rag_validation_scores.labels(metric_type='accuracy').observe(
            scores.get('accuracy_score', 0)
        )
        rag_validation_scores.labels(metric_type='overall').observe(
            scores.get('overall_quality', 0)
        )
    
    def record_corrections(self, count: int):
        """Registra correzioni fatte"""
        rag_corrections_total.inc(count)
    
    def record_document_upload(self, status: str, chunks_count: int = 0, embeddings_count: int = 0):
        """Registra upload documento"""
        rag_documents_uploaded_total.labels(status=status).inc()
        if chunks_count > 0:
            rag_document_chunks_total.inc(chunks_count)
        if embeddings_count > 0:
            rag_embeddings_generated_total.inc(embeddings_count)
    
    def record_retrieval(self, documents_found: int):
        """Registra retrieval documenti"""
        rag_retrieval_documents_found.observe(documents_found)
    
    def get_metrics(self) -> str:
        """Ottieni metriche in formato Prometheus"""
        return generate_latest()
    
    def get_content_type(self) -> str:
        """Content type per endpoint metriche"""
        return CONTENT_TYPE_LATEST

# Istanza globale
prometheus_metrics = PrometheusMetrics()