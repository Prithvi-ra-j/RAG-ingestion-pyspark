#!/bin/bash
# Run embedding generation pipeline

set -e

echo "=== Starting Embedding Generation ==="
echo "Input: data/processed/chunks"
echo "Output: data/embeddings"
echo "Model: all-MiniLM-L6-v2"
echo "Batch Size: 16 (optimized for 8GB RAM)"
echo ""

# Run embedding container
docker-compose run --rm embedding-worker python spark/embed.py \
  --input /data/processed/chunks \
  --output /data/embeddings \
  --model all-MiniLM-L6-v2 \
  --batch-size 16 \
  --checkpoint-interval 5000

echo ""
echo "=== Embedding Generation Complete ==="
echo "Check data/embeddings for output"
echo "Check data/embeddings/metrics.json for statistics"
