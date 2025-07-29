from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import asyncio
from .core.config import settings
from .services.rag_service import AdvancedRAGService
from .core.models import QueryRequest, QueryResponse, DocumentUpload
from .utils.document_processor import DocumentProcessor
from .utils.chunking import AdvancedChunker
from .utils.warnings_filter import suppress_model_warnings, configure_logging_level
from .utils.rag_logger import rag_logger
from .utils.prometheus_metrics import prometheus_metrics
import time
import json

# Suppress model warnings for cleaner logs
suppress_model_warnings()
configure_logging_level()

# Global service instances
rag_service = None
document_processor = None
chunker = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global rag_service, document_processor, chunker
    
    # Startup: Initialize RAG components
    print("Initializing RAG system...")
    rag_service = AdvancedRAGService(settings)
    document_processor = DocumentProcessor()
    chunker = AdvancedChunker(
        strategy=settings.CHUNKING_STRATEGY,
        chunk_size=settings.CHUNK_SIZE,
        overlap=settings.CHUNK_OVERLAP
    )
    
    # Check/Initialize Qdrant collection
    print("Checking Qdrant collection...")
    if rag_service.retrieval_service.collection_exists():
        print("Qdrant collection already exists and ready!")
    else:
        print("Creating Qdrant collection...")
        collection_created = rag_service.retrieval_service.create_collection(settings.EMBEDDING_DIMENSION)
        if collection_created:
            print("Qdrant collection created successfully!")
        else:
            print("Warning: Failed to create Qdrant collection")
    
    print("RAG system ready!")
    yield
    
    # Shutdown: Cleanup resources
    print("Shutting down RAG system...")

app = FastAPI(
    title="Advanced RAG System",
    description="Production-ready RAG with Llama3.2:3b + Gemma2:2b",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process RAG query with advanced features"""
    try:
        start_time = time.time()
        
        result = await rag_service.process_query(
            query=request.query,
            use_hyde=request.use_hyde,
            max_corrections=request.max_corrections or 2
        )
        
        processing_time = time.time() - start_time
        
        return QueryResponse(
            query=result["query"],
            response=result["response"],
            processing_time=processing_time,
            validation_scores=result["validation_scores"],
            corrections_made=result["corrections_made"],
            metadata=result["retrieval_metadata"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document for RAG"""
    upload_start = time.time()
    
    try:
        rag_logger.logger.info(f"📤 UPLOAD_START | File: {file.filename} | Size: {file.size if hasattr(file, 'size') else 'unknown'}")
        
        # Read file content
        file_content = await file.read()
        rag_logger.logger.info(f"📖 FILE_READ | Size: {len(file_content)} bytes")
        
        # Validate file
        validation = await document_processor.validate_file(
            filename=file.filename,
            file_size=len(file_content)
        )
        
        if not validation['valid']:
            rag_logger.log_error(Exception(f"Validation failed: {validation['errors']}"), "file_upload")
            raise HTTPException(
                status_code=400,
                detail=f"File validation failed: {'; '.join(validation['errors'])}"
            )
        
        # Process document
        process_start = time.time()
        processed_doc = await document_processor.process_file(
            file_content=file_content,
            filename=file.filename,
            metadata={
                'uploaded_at': time.time(),
                'original_filename': file.filename,
                'content_type': file.content_type
            }
        )
        process_duration = time.time() - process_start
        
        if processed_doc['processing_status'] != 'success':
            rag_logger.log_error(Exception(f"Processing failed: {processed_doc.get('error')}"), "document_processing")
            raise HTTPException(
                status_code=500,
                detail=f"Document processing failed: {processed_doc.get('error', 'Unknown error')}"
            )
        
        rag_logger.logger.info(f"📄 DOCUMENT_PROCESSED | Duration: {process_duration:.2f}s | Content length: {len(processed_doc['content'])}")
        
        # Chunk the document
        chunk_start = time.time()
        chunks = chunker.chunk_document(
            text=processed_doc['content'],
            metadata=processed_doc['metadata']
        )
        chunk_duration = time.time() - chunk_start
        
        if not chunks:
            rag_logger.log_error(Exception("No chunks created"), "chunking")
            raise HTTPException(
                status_code=400,
                detail="No content could be extracted from the document"
            )
        
        rag_logger.logger.info(f"✂️ CHUNKS_CREATED | Count: {len(chunks)} | Duration: {chunk_duration:.2f}s")
        
        # Generate embeddings for chunks
        embedding_start = time.time()
        chunk_texts = [chunk['content'] for chunk in chunks]
        embeddings = await rag_service.embedding_service.encode_async(chunk_texts)
        embedding_duration = time.time() - embedding_start
        
        rag_logger.logger.info(f"🔢 EMBEDDINGS_GENERATED | Count: {len(embeddings)} | Duration: {embedding_duration:.2f}s")
        
        # Add to Qdrant
        storage_start = time.time()
        success = await rag_service.retrieval_service.add_documents(chunks, embeddings)
        storage_duration = time.time() - storage_start
        
        if not success:
            rag_logger.log_error(Exception("Failed to store in Qdrant"), "vector_storage")
            raise HTTPException(
                status_code=500,
                detail="Failed to add document to vector database"
            )
        
        total_duration = time.time() - upload_start
        
        # Record Prometheus metrics
        prometheus_metrics.record_document_upload("success", len(chunks), len(embeddings))
        
        rag_logger.log_document_upload(file.filename, len(chunks), total_duration)
        rag_logger.logger.info(f"💾 STORAGE_COMPLETE | Duration: {storage_duration:.2f}s | Total: {total_duration:.2f}s")
        
        return {
            "message": "Document uploaded and processed successfully",
            "filename": file.filename,
            "chunks_created": len(chunks),
            "content_hash": processed_doc['content_hash'],
            "processing_details": {
                "mime_type": processed_doc['mime_type'],
                "size": processed_doc['size'],
                "chunk_strategy": settings.CHUNKING_STRATEGY,
                "timings": {
                    "total": round(total_duration, 2),
                    "processing": round(process_duration, 2),
                    "chunking": round(chunk_duration, 2),
                    "embedding": round(embedding_duration, 2),
                    "storage": round(storage_duration, 2)
                }
            }
        }
        
    except HTTPException:
        prometheus_metrics.record_document_upload("error")
        rag_logger.log_error(Exception("HTTP error during upload"), "upload_endpoint")
        raise
    except Exception as e:
        prometheus_metrics.record_document_upload("error")
        rag_logger.log_error(e, "upload_endpoint")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": ["llama3.2:3b", "gemma2:2b"],
        "services": ["qdrant", "embedding", "rag"]
    }

@app.get("/stats")
async def get_performance_stats():
    """Get performance statistics from logs"""
    try:
        stats = rag_logger.get_performance_stats()
        return {
            "status": "success",
            "statistics": stats,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": time.time()
        }

@app.get("/metrics")
async def get_prometheus_metrics():
    """Endpoint per metriche Prometheus"""
    metrics_data = prometheus_metrics.get_metrics()
    return Response(
        content=metrics_data,
        media_type=prometheus_metrics.get_content_type()
    )

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD
    )