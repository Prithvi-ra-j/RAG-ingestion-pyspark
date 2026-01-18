#!/bin/bash
# Run complete RAG pipeline from ingestion to API

set -e

echo "========================================="
echo "  Enterprise RAG Pipeline - Full Run"
echo "========================================="
echo ""

# Step 1: Ingestion
echo "Step 1/5: Document Ingestion"
./docker/run_ingestion.sh

echo ""
echo "========================================="
echo ""

# Step 2: Chunking
echo "Step 2/5: Document Chunking"
./docker/run_chunking.sh

echo ""
echo "========================================="
echo ""

# Step 3: Embedding
echo "Step 3/5: Embedding Generation"
./docker/run_embedding.sh

echo ""
echo "========================================="
echo ""

# Step 4: Vector Store
echo "Step 4/5: Vector Store Building"
./docker/build_vector_store.sh

echo ""
echo "========================================="
echo ""

# Step 5: Start API
echo "Step 5/5: Starting API"
echo "Press Ctrl+C to stop the API server"
echo ""
./docker/start_api.sh
