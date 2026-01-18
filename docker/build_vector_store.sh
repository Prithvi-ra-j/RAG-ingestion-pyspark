#!/bin/bash
# Build vector store from embeddings

set -e

echo "=== Building Vector Store ==="
echo "Embeddings: data/embeddings"
echo "Chunks: data/processed/chunks"
echo "Output: data/vector_store"
echo ""

# Build vector store using Python directly (not in container for simplicity)
python -c "
import sys
sys.path.append('.')
from rag.vector_store import build_vector_store_from_embeddings

build_vector_store_from_embeddings(
    embeddings_path='data/embeddings',
    chunks_path='data/processed/chunks',
    output_path='data/vector_store',
    index_type='Flat',
    use_gpu=False
)
"

echo ""
echo "=== Vector Store Built ==="
echo "Check data/vector_store for output"
echo "Check data/vector_store/metrics.json for statistics"
