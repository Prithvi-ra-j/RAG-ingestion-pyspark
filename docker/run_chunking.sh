#!/bin/bash
# Run document chunking pipeline

set -e

echo "=== Starting Document Chunking ==="
echo "Input: data/processed/docs"
echo "Output: data/processed/chunks"
echo ""

# Run chunking container
docker-compose run --rm spark-chunking python spark/chunk.py \
  --input /data/processed/docs \
  --output /data/processed/chunks \
  --chunk-size 512 \
  --overlap 50 \
  --strategy fixed

echo ""
echo "=== Chunking Complete ==="
echo "Check data/processed/chunks for output"
echo "Check data/processed/chunks/metrics.json for statistics"
