#!/usr/bin/env python3
"""
Setup script for Advanced RAG System
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run shell command with error handling"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}: {e.stderr}")
        return None

def check_ollama():
    """Check if Ollama is installed and running"""
    try:
        result = subprocess.run("ollama --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Ollama is installed")
            return True
        else:
            print("❌ Ollama not found. Please install from https://ollama.ai")
            return False
    except:
        print("❌ Ollama not found. Please install from https://ollama.ai")
        return False

def install_ollama_models():
    """Install required Ollama models"""
    models = ["llama3.2:3b", "gemma2:2b"]
    
    for model in models:
        print(f"📥 Installing {model}...")
        result = run_command(f"ollama pull {model}", f"Installing {model}")
        if result is None:
            print(f"❌ Failed to install {model}")
            return False
    
    return True

def setup_qdrant():
    """Setup Qdrant vector database"""
    print("🗄️ Setting up Qdrant...")
    
    # Check if Docker is available
    docker_available = run_command("docker --version", "Checking Docker") is not None
    
    if docker_available:
        # Start Qdrant with Docker
        qdrant_command = """
        docker run -d --name qdrant-rag \
          -p 6333:6333 -p 6334:6334 \
          -v $(pwd)/qdrant_storage:/qdrant/storage:z \
          qdrant/qdrant
        """
        run_command(qdrant_command, "Starting Qdrant container")
    else:
        print("❌ Docker not found. Please install Docker to run Qdrant")
        print("📝 Alternative: Install Qdrant locally or use Qdrant Cloud")

def create_env_file():
    """Create environment configuration file"""
    env_content = """# RAG System Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=True

# LLM Configuration
MAIN_LLM_MODEL=llama3.2:3b
PREPROCESSING_LLM_MODEL=gemma2:2b
OLLAMA_BASE_URL=http://localhost:11434

# Vector Database Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=rag_documents

# Embedding Configuration
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
EMBEDDING_DIMENSION=1024

# Performance Configuration
ENABLE_CACHING=True
CACHE_TTL=3600
BATCH_SIZE=16

# Monitoring Configuration
ENABLE_MONITORING=True
LOG_LEVEL=INFO
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ Created .env configuration file")

def main():
    """Main setup function"""
    print("🚀 Setting up Advanced RAG System")
    print("="*50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Check if we're in virtual environment
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Warning: Not in virtual environment")
        print("📝 Recommendation: Activate your virtual environment first")
    
    # Create environment file
    create_env_file()
    
    # Check Ollama
    if not check_ollama():
        print("📝 Please install Ollama first, then run this script again")
        return
    
    # Install Ollama models
    if not install_ollama_models():
        print("❌ Failed to install required models")
        return
    
    # Setup Qdrant
    setup_qdrant()
    
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Ensure Docker containers are running (if using Docker)")
    print("2. Run: python -m app.main")
    print("3. Access API at http://localhost:8000")
    print("4. View API docs at http://localhost:8000/docs")
    
    print("\n🔍 Health check URLs:")
    print("- API Health: http://localhost:8000/health")
    print("- Qdrant: http://localhost:6333")
    print("- Grafana (if running): http://localhost:3000")

if __name__ == "__main__":
    main()