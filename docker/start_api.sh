#!/bin/bash
# Start RAG API server

set -e

echo "=== Starting RAG API ==="
echo "Port: 8000"
echo "Vector Store: data/vector_store"
echo ""

# Check if OPENROUTER_API_KEY is set
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "WARNING: OPENROUTER_API_KEY not set. LLM generation will not work."
    echo "Set it with: export OPENROUTER_API_KEY=your_key_here"
    echo ""
fi

# Start API container
docker-compose up rag-api

echo ""
echo "=== API Started ==="
echo "Access API at: http://localhost:8000"
echo "API docs at: http://localhost:8000/docs"
