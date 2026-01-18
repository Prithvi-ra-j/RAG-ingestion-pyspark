# Runbook

## Quick Start (Enhanced Minimal System)

**For immediate testing with your documents:**

```bash
# 1. Add PDFs/TXT files to data/raw/
# 2. Install PDF support
pip install PyPDF2

# 3. Run enhanced system
python enhanced_minimal_rag.py

# 4. Open browser: http://localhost:8001
# 5. Query: "control system safety procedures"
```

**Result**: Working RAG system in <2 minutes, no Docker required.

## Full PySpark Pipeline

### Prerequisites

```bash
# Install Java 17+ (required for PySpark)
# Install Docker Desktop (for containerized deployment)
# Ensure 8GB+ RAM available
```

### Method 1: PowerShell Script (Recommended)

```powershell
# Set OpenRouter API key (optional, for LLM generation)
$env:OPENROUTER_API_KEY="sk-or-v1-your-key-here"

# Run complete pipeline
.\run_pipeline.ps1

# Or run specific stages
.\run_pipeline.ps1 -Stage ingest
.\run_pipeline.ps1 -Stage chunk  
.\run_pipeline.ps1 -Stage embed
.\run_pipeline.ps1 -Stage api
```

### Method 2: Docker Compose

```bash
# Build containers
docker-compose -f docker/docker-compose.yml build

# Run pipeline stages
docker-compose -f docker/docker-compose.yml run --rm spark-ingestion \
  python spark/ingest.py --input /data/raw --output /data/processed/docs

docker-compose -f docker/docker-compose.yml run --rm spark-chunking \
  python spark/chunk.py --input /data/processed/docs --output /data/processed/chunks

docker-compose -f docker/docker-compose.yml run --rm embedding-worker \
  python spark/embed.py --input /data/processed/chunks --output /data/embeddings

# Build vector store (local Python)
python rag/vector_store.py --embeddings data/embeddings --chunks data/processed/chunks --output data/vector_store

# Start API
docker-compose -f docker/docker-compose.yml up rag-api
```

### Method 3: Manual Steps

```bash
# 1. Ingestion
python spark/ingest.py --input data/raw --output data/processed/docs

# 2. Chunking  
python spark/chunk.py --input data/processed/docs --output data/processed/chunks --chunk-size 512 --overlap 50

# 3. Embedding
python spark/embed.py --input data/processed/chunks --output data/embeddings --model all-MiniLM-L6-v2 --batch-size 16

# 4. Vector Store
python rag/vector_store.py --embeddings data/embeddings --chunks data/processed/chunks --output data/vector_store

# 5. API
python rag/api.py
```

## Configuration

### Environment Variables

```bash
# Required for LLM generation
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Optional configurations
EMBEDDING_MODEL=all-MiniLM-L6-v2        # or paraphrase-MiniLM-L3-v2
CHUNK_SIZE=512                          # characters
CHUNK_OVERLAP=50                        # characters  
BATCH_SIZE=16                           # embedding batch size
VECTOR_STORE_PATH=./data/vector_store   # index location
PORT=8000                               # API port
```

### Model Options

| Model | Dimensions | Speed | Memory | Use Case |
|-------|-----------|-------|--------|----------|
| all-MiniLM-L6-v2 | 384 | 400 chunks/sec | 1.2GB | **Default** |
| paraphrase-MiniLM-L3-v2 | 384 | 500 chunks/sec | 0.8GB | Memory-constrained |
| all-mpnet-base-v2 | 768 | 150 chunks/sec | 2.1GB | Higher quality |

### Index Types

| Type | Build Time | Query Time | Memory | Use Case |
|------|-----------|------------|--------|----------|
| Flat | Fast | Slow (>100k) | High | **<100k vectors** |
| IVF | Medium | Fast | Medium | 100k-1M vectors |
| HNSW | Slow | Fastest | Highest | >1M vectors |

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "vector_store_loaded": true,
  "total_chunks": 706,
  "embedding_model": "all-MiniLM-L6-v2"
}
```

### Query Documents

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "control system safety procedures",
    "top_k": 5
  }'
```

```json
{
  "query": "control system safety procedures",
  "answer": "According to the safety procedures...",
  "sources": [
    {
      "chunk_text": "Safety shutdown procedures require...",
      "doc_id": "uuid-123",
      "similarity_score": 0.847,
      "rank": 1
    }
  ],
  "metrics": {
    "retrieval_ms": 42,
    "generation_ms": 1180,
    "total_ms": 1222
  }
}
```

### Query with Filters

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "startup procedures",
    "top_k": 3,
    "filters": {
      "doc_type": "manual",
      "source": "simatic"
    }
  }'
```

## Monitoring

### Check Processing Metrics

```bash
# Ingestion metrics
cat data/processed/docs/metrics.json

# Chunking metrics  
cat data/processed/chunks/metrics.json

# Embedding metrics
cat data/embeddings/metrics.json

# Vector store metrics
cat data/vector_store/metrics.json
```

### Performance Analysis

```bash
# Run comprehensive performance analysis
python experiments/performance_metrics.py

# Output: performance_report.csv + console summary
```

### Log Files

```bash
# Application logs
tail -f logs/app.log

# Docker container logs
docker-compose -f docker/docker-compose.yml logs -f rag-api
```

## Troubleshooting

### Common Issues

**"Java heap space" error:**
```bash
export SPARK_DRIVER_MEMORY=8g
export SPARK_EXECUTOR_MEMORY=8g
```

**"Vector store not loaded" (503 error):**
```bash
# Check if vector store exists
ls -la data/vector_store/

# Rebuild if missing
python rag/vector_store.py --embeddings data/embeddings --chunks data/processed/chunks --output data/vector_store
```

**"Out of memory" during embedding:**
```bash
# Reduce batch size
python spark/embed.py --batch-size 8  # or 4
```

**Docker port conflict:**
```bash
# Check what's using port 8000
netstat -ano | findstr :8000

# Use different port
docker-compose -f docker/docker-compose.yml run -p 8001:8000 rag-api
```

### Recovery Procedures

**Resume failed embedding:**
```bash
# Embeddings auto-resume from last checkpoint
python spark/embed.py --input data/processed/chunks --output data/embeddings
```

**Rebuild corrupted index:**
```bash
# Delete corrupted index
rm -rf data/vector_store/

# Rebuild from embeddings
python rag/vector_store.py --embeddings data/embeddings --chunks data/processed/chunks --output data/vector_store
```

**Clean restart:**
```bash
# Remove all processed data
rm -rf data/processed/ data/embeddings/ data/vector_store/

# Re-run pipeline from scratch
.\run_pipeline.ps1
```

## Data Management

### Directory Structure

```
data/
├── raw/                    # Input documents (PDFs, TXT)
├── processed/
│   ├── docs/              # Ingested documents (Parquet)
│   └── chunks/            # Document chunks (Parquet)
├── embeddings/            # Vector embeddings (NumPy)
└── vector_store/          # FAISS index + metadata
```

### Backup Strategy

```bash
# Backup processed data (avoids re-ingestion)
tar -czf backup_$(date +%Y%m%d).tar.gz data/processed/ data/embeddings/ data/vector_store/

# Restore from backup
tar -xzf backup_20240115.tar.gz
```

### Adding New Documents

```bash
# 1. Add new files to data/raw/
# 2. Re-run ingestion (will skip existing documents)
python spark/ingest.py --input data/raw --output data/processed/docs

# 3. Re-run remaining pipeline
python spark/chunk.py --input data/processed/docs --output data/processed/chunks
python spark/embed.py --input data/processed/chunks --output data/embeddings  
python rag/vector_store.py --embeddings data/embeddings --chunks data/processed/chunks --output data/vector_store
```

## Production Deployment

### Resource Requirements

**Minimum**:
- 8GB RAM
- 4 CPU cores  
- 20GB disk space
- Java 17+

**Recommended**:
- 16GB RAM
- 8 CPU cores
- 50GB SSD
- GPU (optional, 10x speedup)

### Security Considerations

- **Air-gapped deployment**: No internet required after setup
- **Local storage**: All data stays on-premises
- **API keys**: Store in environment variables, not code
- **Network**: API runs on localhost by default

### Scaling Guidelines

- **<10k documents**: Single machine sufficient
- **10k-100k documents**: Consider GPU for embedding
- **100k-500k documents**: Upgrade to 32GB RAM
- **>500k documents**: Implement distributed embedding