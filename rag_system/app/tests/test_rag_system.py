import pytest
import asyncio
from app.services.rag_service import AdvancedRAGService
from app.core.config import Settings

@pytest.fixture
def test_config():
    return Settings(
        MAIN_LLM_MODEL="llama3.2:3b",
        PREPROCESSING_LLM_MODEL="gemma2:2b",
        QDRANT_COLLECTION_NAME="test_collection"
    )

@pytest.fixture
async def rag_service(test_config):
    service = AdvancedRAGService(test_config)
    yield service
    # Cleanup

@pytest.mark.asyncio
async def test_basic_query_processing(rag_service):
    """Test basic RAG query processing"""
    result = await rag_service.process_query("What is machine learning?")
    
    assert result["query"] == "What is machine learning?"
    assert len(result["response"]) > 0
    assert isinstance(result["validation_scores"], dict)
    assert result["corrections_made"] >= 0

@pytest.mark.asyncio
async def test_hyde_query_expansion(rag_service):
    """Test HyDE query expansion"""
    result = await rag_service.process_query("ML basics", use_hyde=True)
    
    assert result["retrieval_metadata"]["hyde_used"] is True
    assert len(result["response"]) > 0

@pytest.mark.asyncio
async def test_self_validation_system(rag_service):
    """Test self-validation and correction"""
    # This test would ideally use a query that triggers correction
    result = await rag_service.process_query("Complex technical query")
    
    assert "validation_scores" in result
    assert "overall_quality" in result["validation_scores"]
    assert result["validation_scores"]["overall_quality"] >= 0.0

class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.mark.asyncio
    async def test_response_time_benchmark(self, rag_service):
        """Test response time is under acceptable threshold"""
        import time
        
        start = time.time()
        result = await rag_service.process_query("Quick test query")
        duration = time.time() - start
        
        assert duration < 10.0  # Response under 10 seconds
        assert result["processing_time"] < 15.0
    
    @pytest.mark.asyncio
    async def test_concurrent_queries(self, rag_service):
        """Test handling multiple concurrent queries"""
        queries = [f"Test query {i}" for i in range(3)]
        
        tasks = [rag_service.process_query(query) for query in queries]
        results = await asyncio.gather(*tasks)
        
        assert len(results) == 3
        for result in results:
            assert len(result["response"]) > 0