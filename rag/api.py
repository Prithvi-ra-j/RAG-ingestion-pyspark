"""
RAG API Layer (FastAPI)

This module implements the REST API for RAG queries with retrieval
and generation using OpenRouter for LLM inference.

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5
"""

import os
import time
import json
from typing import List, Dict, Optional
import logging

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

from vector_store import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enterprise RAG API",
    description="Retrieval-Augmented Generation API for enterprise document search",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
vector_store: Optional[VectorStore] = None
embedding_model: Optional[SentenceTransformer] = None
openrouter_api_key: Optional[str] = None
openrouter_model: str = "openai/gpt-3.5-turbo"


# Request/Response models
class Source(BaseModel):
    """Source document chunk."""
    chunk_id: str
    chunk_text: str
    doc_id: str
    source_path: Optional[str] = None
    similarity_score: float
    rank: int


class QueryMetrics(BaseModel):
    """Query performance metrics."""
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    num_chunks_retrieved: int


class QueryRequest(BaseModel):
    """Query request schema."""
    query: str = Field(..., description="User query text")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    filters: Optional[Dict] = Field(default=None, description="Metadata filters")
    model: Optional[str] = Field(default=None, description="LLM model to use (OpenRouter)")


class QueryResponse(BaseModel):
    """Query response schema."""
    answer: str
    sources: List[Source]
    metrics: QueryMetrics


# Startup/Shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize models and vector store on startup."""
    global vector_store, embedding_model, openrouter_api_key, openrouter_model
    
    logger.info("Starting RAG API...")
    
    # Load configuration from environment
    vector_store_path = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_model = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")
    
    if not openrouter_api_key:
        logger.warning("OPENROUTER_API_KEY not set. LLM generation will not work.")
    
    # Load vector store
    try:
        logger.info(f"Loading vector store from {vector_store_path}")
        vector_store = VectorStore()
        vector_store.load(vector_store_path)
        logger.info(f"Vector store loaded: {vector_store.metrics['total_vectors']} vectors")
    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        logger.error("API will start but queries will fail. Please build vector store first.")
    
    # Load embedding model
    try:
        logger.info(f"Loading embedding model: {embedding_model_name}")
        embedding_model = SentenceTransformer(embedding_model_name)
        logger.info("Embedding model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down RAG API...")


# API endpoints
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "vector_store_loaded": vector_store is not None,
        "embedding_model_loaded": embedding_model is not None,
        "openrouter_configured": openrouter_api_key is not None
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Process RAG query: retrieve relevant chunks and generate answer.
    
    Args:
        request: Query request with query text and parameters
        
    Returns:
        Query response with answer, sources, and metrics
    """
    start_time = time.time()
    
    # Validate vector store is loaded
    if vector_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not loaded. Please build and load vector store first."
        )
    
    if embedding_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding model not loaded."
        )
    
    try:
        # Step 1: Embed query
        retrieval_start = time.time()
        query_vector = embed_query(request.query)
        
        # Step 2: Retrieve relevant chunks
        results = vector_store.search(
            query_vector=query_vector,
            k=request.top_k,
            filters=request.filters
        )
        
        retrieval_time = (time.time() - retrieval_start) * 1000  # Convert to ms
        
        if not results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant documents found for query."
            )
        
        # Step 3: Format sources
        sources = [
            Source(
                chunk_id=r["chunk_id"],
                chunk_text=r.get("chunk_text", ""),
                doc_id=r.get("doc_id", ""),
                source_path=r.get("source_path"),
                similarity_score=r["similarity_score"],
                rank=r["rank"]
            )
            for r in results
        ]
        
        # Step 4: Generate answer using LLM
        generation_start = time.time()
        
        if openrouter_api_key:
            answer = generate_answer(
                query=request.query,
                chunks=results,
                model=request.model or openrouter_model
            )
        else:
            # Fallback: return concatenated chunks without LLM
            answer = "LLM not configured. Retrieved chunks:\n\n" + "\n\n".join(
                f"[{i+1}] {r.get('chunk_text', '')[:200]}..."
                for i, r in enumerate(results)
            )
        
        generation_time = (time.time() - generation_start) * 1000  # Convert to ms
        total_time = (time.time() - start_time) * 1000
        
        # Step 5: Return response
        return QueryResponse(
            answer=answer,
            sources=sources,
            metrics=QueryMetrics(
                retrieval_ms=round(retrieval_time, 2),
                generation_ms=round(generation_time, 2),
                total_ms=round(total_time, 2),
                num_chunks_retrieved=len(sources)
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )


def embed_query(query: str) -> np.ndarray:
    """
    Embed query text using the same model as document embedding.
    
    Args:
        query: Query text
        
    Returns:
        Query embedding vector
    """
    try:
        embedding = embedding_model.encode([query], convert_to_numpy=True)
        return embedding[0]
    except Exception as e:
        logger.error(f"Query embedding failed: {e}")
        raise


def generate_answer(query: str, chunks: List[Dict], model: str) -> str:
    """
    Generate answer using OpenRouter API.
    
    Args:
        query: User query
        chunks: Retrieved chunks with metadata
        model: OpenRouter model name
        
    Returns:
        Generated answer
    """
    # Format prompt with retrieved chunks
    prompt = format_prompt(query, chunks)
    
    # Call OpenRouter API
    try:
        response = call_openrouter(prompt, model)
        return response
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        # Return fallback response
        return f"Error generating answer: {str(e)}\n\nRetrieved information:\n\n" + "\n\n".join(
            f"[{i+1}] {c.get('chunk_text', '')[:300]}..."
            for i, c in enumerate(chunks[:3])
        )


def format_prompt(query: str, chunks: List[Dict]) -> str:
    """
    Format prompt for LLM with query and retrieved chunks.
    
    Args:
        query: User query
        chunks: Retrieved chunks
        
    Returns:
        Formatted prompt
    """
    context = "\n\n".join(
        f"[Document {i+1}] (Source: {c.get('source_path', 'Unknown')})\n{c.get('chunk_text', '')}"
        for i, c in enumerate(chunks)
    )
    
    prompt = f"""You are a helpful assistant answering questions based on provided documents.

Context from relevant documents:
{context}

User Question: {query}

Instructions:
- Answer the question based on the provided context
- If the context doesn't contain enough information, say so
- Cite sources by referring to document numbers (e.g., "According to Document 1...")
- Be concise and accurate

Answer:"""
    
    return prompt


def call_openrouter(prompt: str, model: str, max_retries: int = 3) -> str:
    """
    Call OpenRouter API with exponential backoff retry logic.
    
    Args:
        prompt: Formatted prompt
        model: Model name
        max_retries: Maximum number of retries
        
    Returns:
        Generated text
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.Timeout:
            logger.warning(f"OpenRouter API timeout (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise
        
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenRouter API error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    
    raise Exception("OpenRouter API call failed after retries")


# Additional utility endpoints
@app.get("/stats")
async def get_stats():
    """Get vector store statistics."""
    if vector_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not loaded."
        )
    
    return {
        "total_vectors": vector_store.metrics.get("total_vectors", 0),
        "index_type": vector_store.index_type,
        "embedding_dim": vector_store.embedding_dim,
        "use_gpu": vector_store.use_gpu
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
