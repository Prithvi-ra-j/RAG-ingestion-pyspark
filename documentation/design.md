# Design Document

## Overview

This system implements a scalable, on-premises RAG ingestion pipeline using PySpark for distributed document processing, decoupled from LLM inference. The architecture prioritizes batch efficiency, operational simplicity, and measurable performance over real-time responsiveness.

**Core Design Principle**: Separate data preparation (Spark) from inference (single-node) to optimize each independently and enable data reuse across multiple RAG experiments.

## Architecture

### System Context

```
┌─────────────────────────────────────────────────────────────┐
│                     Enterprise Environment                   │
│                                                              │
│  ┌──────────────┐         ┌─────────────────┐              │
│  │   Document   │────────▶│  Ingestion Job  │              │
│  │   Corpus     │         │   (PySpark)     │              │
│  │ (10k-1M PDFs)│         └────────┬────────┘              │
│  └──────────────┘                  │                        │
│                                    │                        │
│                          ┌─────────▼────────┐              │
│                          │  Parquet Store   │              │
│                          │  (Processed Docs)│              │
│                          └─────────┬────────┘              │
│                                    │                        │
│                          ┌─────────▼────────┐              │
│                          │  Chunking Job    │              │
│                          │   (Spark UDFs)   │              │
│                          └─────────┬────────┘              │
│                                    │                        │
│                          ┌─────────▼────────┐              │
│                          │  Embedding Job   │              │
│                          │ (Single Node +   │              │
│                          │  HF Transformers)│              │
│                          └─────────┬────────┘              │
│                                    │                        │
│                    ┌───────────────┴───────────────┐       │
│                    │                                │       │
│          ┌─────────▼────────┐          ┌──────────▼──────┐│
│          │   Vector Store   │          │  Metadata Store ││
│          │  (FAISS/Chroma)  │          │   (Parquet)     ││
│          └─────────┬────────┘          └──────────┬──────┘│
│                    │                                │       │
│                    └───────────────┬────────────────┘       │
│                                    │                        │
│                          ┌─────────▼────────┐              │
│                          │    RAG Layer     │              │
│                          │   (FastAPI)      │              │
│                          └─────────┬────────┘              │
│                                    │                        │
│                          ┌─────────▼────────┐              │
│                          │   LLM Client     │              │
│                          │  (LangChain)     │              │
│                          └──────────────────┘              │
└─────────────────────────────────────────────────────────────┘
```

### Component Separation Rationale

**Why Spark doesn't touch inference:**
1. Spark excels at data-parallel operations (parsing, chunking) but adds overhead for model inference
2. Embedding models benefit from GPU batching on a single node, not distributed CPU execution
3. Decoupling allows independent scaling: add Spark workers for ingestion, add GPUs for embedding
4. Enables experimentation with different embedding models without re-running Spark jobs

## Components and Interfaces

### 1. Ingestion System (PySpark)

**Responsibility**: Transform raw documents into clean, deduplicated, structured text

**Input**: 
- Directory path containing PDF/TXT/LOG files
- Configuration: file patterns, deduplication strategy

**Output**:
- Parquet files with schema: `[doc_id: string, source_path: string, text: string, hash: string, ingestion_timestamp: timestamp]`

**Performance Target**: 1000+ documents/minute on 4-node Spark cluster

### 2. Chunking Engine (Spark UDFs)

**Responsibility**: Split documents into retrieval-optimized segments with metadata preservation

**Input**:
- Parquet files from Ingestion System
- Configuration: chunk_size, overlap, strategy (fixed/semantic)

**Output**:
- Parquet files with schema: `[chunk_id: string, doc_id: string, chunk_text: string, chunk_index: int, char_start: int, char_end: int]`

**Performance Target**: 5000+ chunks/second on 4-node cluster

### 3. Embedding Pipeline (Single Node)

**Responsibility**: Generate vector representations of chunks using local models

**Input**:
- Parquet files from Chunking Engine
- Configuration: model_name, batch_size, device (cpu/cuda)

**Output**:
- NumPy arrays: `embeddings.npy` (shape: [N, embedding_dim])
- Parquet metadata: `[chunk_id: string, embedding_index: int]`

**Performance Target**: 200-500 chunks/second on 8GB RAM CPU (batch_size=16)

### 4. Vector Store (FAISS/Chroma)

**Responsibility**: Index embeddings for fast similarity search with metadata filtering

**Performance Target**: <50ms retrieval latency for top-10 results

### 5. RAG Layer (FastAPI)

**Responsibility**: Expose retrieval and generation interface, orchestrate query flow

**Performance Target**: <2s end-to-end latency (retrieval + generation)

## Hardware Constraints and Adaptations

This implementation is optimized for **8GB RAM CPU environments without GPU**, making it accessible to most developers and realistic for resource-constrained deployments.

**Key Adaptations**:

1. **Embedding Model Selection**: Use smaller models (384-dim) like `all-MiniLM-L6-v2` or `paraphrase-MiniLM-L3-v2` instead of larger 768-dim models
2. **Batch Size**: Reduce from 64 to 16 for embedding generation to fit in memory
3. **Checkpointing**: More frequent checkpoints (every 5k instead of 10k) to reduce re-work on OOM failures
4. **Memory-Mapped Arrays**: Use `np.memmap` for embeddings to avoid loading entire arrays into RAM
5. **LLM Integration**: Use OpenRouter API instead of local models to avoid memory overhead of model serving

## Data Models

### Document Schema (Post-Ingestion)

```python
Document = {
    "doc_id": str,           # UUID
    "source_path": str,      # Original file path
    "text": str,             # Extracted text
    "hash": str,             # SHA256 for deduplication
    "file_type": str,        # pdf, txt, log
    "file_size_bytes": int,
    "page_count": int,       # For PDFs
    "ingestion_timestamp": datetime,
    "parser_version": str    # For reproducibility
}
```

### Chunk Schema

```python
Chunk = {
    "chunk_id": str,         # UUID
    "doc_id": str,           # Foreign key to Document
    "chunk_text": str,       # Actual text content
    "chunk_index": int,      # Position in document
    "char_start": int,       # Character offset in original doc
    "char_end": int,
    "token_count": int,      # Approximate tokens
    "chunking_strategy": str # fixed, semantic
}
```

## Error Handling

### Ingestion Errors

| Error Type | Handling Strategy | Recovery |
|------------|------------------|----------|
| Corrupted PDF | Log to failed_docs.json, continue | Manual review, re-ingest |
| Encoding errors | Try UTF-8, Latin-1, fallback to ignore | Best-effort text extraction |
| Out of memory | Reduce Spark partition size | Automatic retry with smaller batches |
| Disk full | Halt processing, alert | Manual intervention required |

### Embedding Errors

| Error Type | Handling Strategy | Recovery |
|------------|------------------|----------|
| OOM during batch | Reduce batch size by 50%, retry | Automatic |
| Model load failure | Fail fast with clear error | Check model path, re-download |
| CUDA errors | Fall back to CPU | Automatic |

### Query Errors

| Error Type | Handling Strategy | Recovery |
|------------|------------------|----------|
| Index not found | Return 503 with rebuild instructions | Manual index rebuild |
| Timeout | Return partial results if available | Client retry |
| Invalid query | Return 400 with validation errors | Client fix |

## Deployment Architecture

### Docker Composition

```
enterprise-rag-pyspark/
├── docker/
│   ├── Dockerfile.spark       # PySpark + dependencies
│   ├── Dockerfile.embedding   # Python + HF transformers
│   ├── Dockerfile.api         # FastAPI + LangChain
│   └── docker-compose.yml     # Orchestration
```

### Container Responsibilities

**spark-ingestion**:
- Runs ingestion and chunking jobs
- Mounts: `/data/raw` (input), `/data/processed` (output)
- Resources: 4 CPU, 8GB RAM

**embedding-worker**:
- Runs embedding pipeline
- Mounts: `/data/processed` (input), `/data/embeddings` (output)
- Resources: 8 CPU, 16GB RAM (or 1 GPU)

**rag-api**:
- Serves FastAPI application
- Mounts: `/data/embeddings` (read-only)
- Resources: 2 CPU, 4GB RAM
- Ports: 8000

## Performance Optimization

### Spark Tuning

```python
spark_config = {
    "spark.executor.memory": "4g",
    "spark.executor.cores": "2",
    "spark.sql.shuffle.partitions": "200",  # Adjust based on data size
    "spark.sql.adaptive.enabled": "true",   # Adaptive query execution
    "spark.serializer": "org.apache.spark.serializer.KryoSerializer"
}
```

### Embedding Optimization

```python
# Use mixed precision for GPU
model = SentenceTransformer(model_name)
model.half()  # FP16 for 2x speedup

# Batch size tuning
batch_size = 64 if torch.cuda.is_available() else 16

# Multi-processing for CPU
pool = model.start_multi_process_pool()
embeddings = model.encode_multi_process(texts, pool)
```

## Future Enhancements

### Phase 2: Incremental Updates

**Problem**: Full re-ingestion is expensive for large corpora

**Solution**: 
- Track document hashes in metadata store
- Only process new/changed documents
- Merge new embeddings into existing index

**Complexity**: Medium (requires change detection logic)

### Phase 3: Distributed Embedding

**Problem**: Single-node embedding is a bottleneck at 500k+ documents

**Solution**:
- Use Ray or Spark UDFs with model broadcasting
- Partition chunks across GPU nodes
- Aggregate embeddings

**Complexity**: High (model serialization, GPU scheduling)

### Phase 4: Multi-Modal Support

**Problem**: PDFs contain images, tables, diagrams

**Solution**:
- Extract images with pdfplumber
- Use CLIP for image embeddings
- Hybrid retrieval (text + image)

**Complexity**: High (requires vision models, hybrid indexing)

## Success Metrics

### Quantitative

| Metric | Target | Measurement |
|--------|--------|-------------|
| Ingestion throughput | 1000 docs/min | Spark job logs |
| Chunking throughput | 5000 chunks/sec | Spark job logs |
| Embedding throughput | 500 chunks/sec (CPU) | Python profiling |
| Retrieval latency | <50ms | API metrics |
| End-to-end latency | <2s | API metrics |
| Recall@10 | >0.7 | Quality test suite |

### Qualitative

- **Operational simplicity**: Can a new engineer run the pipeline in <1 hour?
- **Debuggability**: Are failures easy to diagnose from logs?
- **Reproducibility**: Do repeated runs produce identical results?
- **Documentation quality**: Can the system be understood without reading code?