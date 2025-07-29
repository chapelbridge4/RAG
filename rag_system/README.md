# Advanced RAG System

🚧 **Work in Progress** - An advanced Retrieval Augmented Generation system featuring:
- **Llama3.2:3b** for main generation
- **Gemma2:2b** for preprocessing and validation  
- **Qdrant** vector database
- **FastAPI** + async processing
- **Self-validation** with auto-correction
- **Latest 2024/2025 RAG techniques** (HyDE, contextual compression, etc.)

## Quick Start

1. **Setup Environment**
   ```bash
   python -m venv rag_env
   source rag_env/bin/activate  # Linux/Mac
   # rag_env\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. **Install Ollama and Models**
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull llama3.2:3b
   ollama pull gemma2:2b
   ```

3. **Start Services**
   ```bash
   # If Docker is available
   cd docker && docker compose up -d
   
   # Or manually start Qdrant
   docker run -d --name qdrant-rag -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

4. **Initialize System**
   ```bash
   python scripts/setup_system.py
   ```

5. **Run API**
   ```bash
   cd rag_system
   python -m app.main
   ```

## Architecture Overview

```
Query → HyDE Expansion → Embedding → Hybrid Retrieval → 
Contextual Compression → LLM Generation → Self-Validation → Response
```

## Advanced Features

- **HyDE Pattern**: Hypothetical document generation for better retrieval
- **Multi-Query Expansion**: Generate query variations for comprehensive search
- **Contextual Compression**: Extract only relevant information from retrieved documents
- **Self-Validation**: Automatic quality assessment and correction
- **Hybrid Search**: Combines dense and sparse retrieval methods
- **Real-time Monitoring**: Prometheus metrics and Grafana dashboards

## API Usage

```python
import requests

response = requests.post("http://localhost:8000/query", json={
    "query": "What are the main features?",
    "use_hyde": True,
    "max_corrections": 2
})

result = response.json()
print(f"Response: {result['response']}")
print(f"Quality Score: {result['validation_scores']['overall_quality']}")
```

## Project Structure

```
rag_system/
├── app/
│   ├── core/                 # Configuration and models
│   ├── services/             # RAG components
│   ├── api/                  # FastAPI routes
│   ├── utils/                # Utilities (chunking, etc.)
│   └── tests/                # Test suite
├── data/                     # Documents and embeddings
├── docker/                   # Docker configuration
├── scripts/                  # Setup and utility scripts
└── config/                   # Configuration files
```

## Testing

```bash
pytest app/tests/ -v
```

## Monitoring

- **API Health**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs
- **Qdrant**: http://localhost:6333
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

## Configuration

Copy `.env.example` to `.env` and adjust settings:

```env
# LLM Configuration
MAIN_LLM_MODEL=llama3.2:3b
PREPROCESSING_LLM_MODEL=gemma2:2b

# Vector Database
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Performance
CHUNK_SIZE=800
RETRIEVAL_K=20
RERANK_K=8
```

## Key Components

### 1. Embedding Service (`app/services/embedding_service.py`)
- Sentence-transformers with caching
- Batch processing optimization
- Async support

### 2. Retrieval Service (`app/services/retrieval_service.py`)
- Hybrid vector + cross-encoder reranking
- HNSW optimization
- Configurable search parameters

### 3. Self-Validation (`app/services/validation_service.py`)
- Groundedness checking
- Relevance assessment
- Automatic correction
- Quality scoring

### 4. HyDE Service (`app/services/hyde_service.py`)
- Hypothetical document generation
- Query expansion strategies
- Multiple search perspectives

## Current Status & Benchmarks

🚧 **In Development**: Performance metrics being collected and optimized

- **Response Time**: Variable (optimization in progress)
- **Accuracy**: Testing on evaluation datasets
- **Throughput**: Benchmarking in progress  
- **Self-Correction**: Validation system implemented

## Troubleshooting

### Common Issues

1. **Ollama Connection Error**
   - Ensure Ollama is running: `ollama serve`
   - Check model availability: `ollama list`

2. **Qdrant Connection Error**
   - Verify Docker container: `docker ps`
   - Check port availability: `netstat -an | grep 6333`

3. **Memory Issues**
   - Reduce batch size in config
   - Use smaller embedding models
   - Enable model offloading

### Performance Optimization

1. **Embedding Caching**: Enable Redis for production
2. **Model Quantization**: Use quantized models for faster inference
3. **Batch Processing**: Optimize batch sizes for your hardware
4. **Vector Index**: Tune HNSW parameters for your data

## Contributing

🚧 **This is a work-in-progress project**

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Ensure tests pass: `pytest`
5. Submit pull request

**Note**: The system is actively being developed. Some features may be incomplete or under optimization.

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Llama 3.2 by Meta
- Gemma 2 by Google
- Qdrant vector database
- LangChain framework
- FastAPI framework