#!/bin/bash
# Run document ingestion pipeline

set -e

echo "=== Starting Document Ingestion ==="
echo "Input: data/raw"
echo "Output: data/processed/docs"
echo ""

# Run ingestion container
docker-compose run --rm spark-ingestion python spark/ingest.py \
  --input /data/raw \
  --output /data/processed/docs

echo ""
echo "=== Ingestion Complete ==="
echo "Check data/processed/docs for output"
echo "Check data/processed/docs/metrics.json for statistics"
