"""
Vector Store with FAISS

This module implements vector indexing and similarity search using FAISS
with metadata integration for filtering and retrieval.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
"""

import os
import json
import pickle
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

import faiss
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-based vector store with metadata integration.
    
    Supports Flat and IVF index types with optional GPU acceleration.
    """
    
    def __init__(self, index_type: str = "Flat", use_gpu: bool = None):
        """
        Initialize the vector store.
        
        Args:
            index_type: Index type ('Flat', 'IVF', or 'HNSW')
            use_gpu: Use GPU if available (None for auto-detect)
        """
        self.index_type = index_type
        self.use_gpu = use_gpu if use_gpu is not None else torch.cuda.is_available()
        self.index = None
        self.metadata = {}  # chunk_id -> metadata dict
        self.id_to_index = {}  # chunk_id -> embedding_index
        self.index_to_id = {}  # embedding_index -> chunk_id
        self.embedding_dim = None
        self.metrics = {
            "total_vectors": 0,
            "index_type": index_type,
            "use_gpu": self.use_gpu,
            "build_time_seconds": 0.0
        }
    
    def build_index(self, embeddings: np.ndarray, index_type: Optional[str] = None) -> faiss.Index:
        """
        Create FAISS index from embeddings.
        
        Args:
            embeddings: NumPy array of embeddings (shape: [N, dim])
            index_type: Index type override
            
        Returns:
            FAISS index
        """
        if index_type:
            self.index_type = index_type
        
        n_vectors, dim = embeddings.shape
        self.embedding_dim = dim
        
        logger.info(f"Building {self.index_type} index for {n_vectors} vectors (dim={dim})")
        
        start_time = datetime.now()
        
        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')
        
        if self.index_type == "Flat":
            # Flat index: exact search, good for <100k vectors
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings)
            
        elif self.index_type == "IVF":
            # IVF index: approximate search, good for 100k-1M vectors
            # Number of clusters (nlist) should be ~sqrt(N) to 4*sqrt(N)
            nlist = min(int(np.sqrt(n_vectors) * 2), n_vectors // 39)
            nlist = max(nlist, 1)  # At least 1 cluster
            
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            
            # Train the index
            logger.info(f"Training IVF index with {nlist} clusters...")
            index.train(embeddings)
            index.add(embeddings)
            
            # Set search parameters (nprobe = number of clusters to search)
            index.nprobe = min(10, nlist)  # Search 10 clusters by default
            
        elif self.index_type == "HNSW":
            # HNSW index: fast approximate search, good for >1M vectors
            M = 32  # Number of connections per layer
            index = faiss.IndexHNSWFlat(dim, M)
            index.add(embeddings)
            
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
        
        # Transfer to GPU if requested
        if self.use_gpu and torch.cuda.is_available():
            logger.info("Transferring index to GPU...")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        build_time = (datetime.now() - start_time).total_seconds()
        
        self.index = index
        self.metrics["total_vectors"] = n_vectors
        self.metrics["build_time_seconds"] = round(build_time, 2)
        
        logger.info(f"Index built in {build_time:.2f}s")
        
        return index
    
    def add_metadata(self, chunk_ids: List[str], metadata_dict: Dict[str, Dict]):
        """
        Link chunk IDs to document metadata.
        
        Args:
            chunk_ids: List of chunk IDs (in same order as embeddings)
            metadata_dict: Dictionary mapping chunk_id to metadata
        """
        logger.info(f"Adding metadata for {len(chunk_ids)} chunks")
        
        # Create bidirectional mappings
        for idx, chunk_id in enumerate(chunk_ids):
            self.id_to_index[chunk_id] = idx
            self.index_to_id[idx] = chunk_id
            
            # Store metadata if provided
            if chunk_id in metadata_dict:
                self.metadata[chunk_id] = metadata_dict[chunk_id]
        
        logger.info(f"Metadata added for {len(self.metadata)} chunks")
    
    def search(self, query_vector: np.ndarray, k: int = 10, 
              filters: Optional[Dict] = None) -> List[Dict]:
        """
        Retrieve top-k similar vectors with optional metadata filtering.
        
        Args:
            query_vector: Query embedding (shape: [dim] or [1, dim])
            k: Number of results to return
            filters: Optional metadata filters (e.g., {"file_type": "pdf"})
            
        Returns:
            List of result dictionaries with chunk_id, score, and metadata
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Ensure query vector is 2D
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Ensure float32
        query_vector = query_vector.astype('float32')
        
        # Search index (retrieve more if filtering)
        search_k = k * 10 if filters else k
        distances, indices = self.index.search(query_vector, search_k)
        
        # Convert to results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty results
                continue
            
            chunk_id = self.index_to_id.get(idx)
            if chunk_id is None:
                continue
            
            # Get metadata
            chunk_metadata = self.metadata.get(chunk_id, {})
            
            # Apply filters
            if filters:
                if not self._matches_filters(chunk_metadata, filters):
                    continue
            
            result = {
                "chunk_id": chunk_id,
                "similarity_score": float(dist),
                "rank": len(results) + 1,
                **chunk_metadata
            }
            
            results.append(result)
            
            # Stop if we have enough results
            if len(results) >= k:
                break
        
        return results
    
    def _matches_filters(self, metadata: Dict, filters: Dict) -> bool:
        """
        Check if metadata matches filter criteria.
        
        Args:
            metadata: Chunk metadata
            filters: Filter criteria
            
        Returns:
            True if metadata matches all filters
        """
        for key, value in filters.items():
            if key not in metadata:
                return False
            
            if isinstance(value, list):
                # Match any value in list
                if metadata[key] not in value:
                    return False
            else:
                # Exact match
                if metadata[key] != value:
                    return False
        
        return True
    
    def persist(self, index_path: str):
        """
        Save index and metadata to disk.
        
        Args:
            index_path: Directory to save index files
        """
        os.makedirs(index_path, exist_ok=True)
        
        logger.info(f"Saving index to {index_path}")
        
        # Save FAISS index (transfer from GPU if needed)
        index_file = os.path.join(index_path, "vector_store.index")
        if self.use_gpu and torch.cuda.is_available():
            # Transfer back to CPU for saving
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, index_file)
        else:
            faiss.write_index(self.index, index_file)
        
        # Save metadata
        metadata_file = os.path.join(index_path, "metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump({
                "metadata": self.metadata,
                "id_to_index": self.id_to_index,
                "index_to_id": self.index_to_id,
                "embedding_dim": self.embedding_dim,
                "index_type": self.index_type
            }, f)
        
        # Save metrics
        metrics_file = os.path.join(index_path, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info("Index and metadata saved successfully")
    
    def load(self, index_path: str):
        """
        Load index and metadata from disk.
        
        Args:
            index_path: Directory containing index files
        """
        logger.info(f"Loading index from {index_path}")
        
        # Load FAISS index
        index_file = os.path.join(index_path, "vector_store.index")
        if not os.path.exists(index_file):
            raise FileNotFoundError(f"Index file not found: {index_file}")
        
        index = faiss.read_index(index_file)
        
        # Transfer to GPU if requested
        if self.use_gpu and torch.cuda.is_available():
            logger.info("Transferring index to GPU...")
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        
        self.index = index
        
        # Load metadata
        metadata_file = os.path.join(index_path, "metadata.pkl")
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'rb') as f:
            data = pickle.load(f)
            self.metadata = data["metadata"]
            self.id_to_index = data["id_to_index"]
            self.index_to_id = data["index_to_id"]
            self.embedding_dim = data["embedding_dim"]
            self.index_type = data["index_type"]
        
        # Load metrics
        metrics_file = os.path.join(index_path, "metrics.json")
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                self.metrics = json.load(f)
        
        logger.info(f"Loaded index with {len(self.metadata)} vectors")
    
    def validate_index(self) -> bool:
        """
        Validate index integrity.
        
        Returns:
            True if index is valid
        """
        if self.index is None:
            logger.error("Index is None")
            return False
        
        # Check index size matches metadata
        if self.index.ntotal != len(self.id_to_index):
            logger.error(f"Index size mismatch: {self.index.ntotal} != {len(self.id_to_index)}")
            return False
        
        # Test search
        try:
            test_vector = np.random.randn(1, self.embedding_dim).astype('float32')
            self.index.search(test_vector, 1)
        except Exception as e:
            logger.error(f"Index search test failed: {e}")
            return False
        
        logger.info("Index validation passed")
        return True


def build_vector_store_from_embeddings(embeddings_path: str, chunks_path: str, 
                                       output_path: str, index_type: str = "Flat",
                                       use_gpu: bool = None):
    """
    Build vector store from embeddings and chunk metadata.
    
    Args:
        embeddings_path: Path to embeddings directory
        chunks_path: Path to chunks Parquet directory
        output_path: Path to save vector store
        index_type: FAISS index type
        use_gpu: Use GPU if available
    """
    logger.info("Building vector store...")
    
    # Load embeddings
    embeddings_file = os.path.join(embeddings_path, "embeddings.npy")
    embeddings = np.load(embeddings_file, mmap_mode='r')
    logger.info(f"Loaded embeddings: {embeddings.shape}")
    
    # Load embedding metadata
    metadata_file = os.path.join(embeddings_path, "embedding_metadata.json")
    with open(metadata_file, 'r') as f:
        embedding_metadata = json.load(f)
    
    chunk_ids = [m["chunk_id"] for m in embedding_metadata]
    
    # Load chunk metadata from Parquet
    import pyarrow.parquet as pq
    table = pq.read_table(chunks_path)
    chunks_df = table.to_pandas()
    
    # Create metadata dictionary
    metadata_dict = {}
    for _, row in chunks_df.iterrows():
        metadata_dict[row["chunk_id"]] = {
            "chunk_text": row["chunk_text"],
            "doc_id": row["doc_id"],
            "source_path": row.get("source_path"),
            "file_type": row.get("file_type"),
            "chunk_index": row.get("chunk_index"),
            "token_count": row.get("token_count")
        }
    
    # Build vector store
    store = VectorStore(index_type=index_type, use_gpu=use_gpu)
    store.build_index(embeddings)
    store.add_metadata(chunk_ids, metadata_dict)
    
    # Validate
    if not store.validate_index():
        raise ValueError("Index validation failed")
    
    # Save
    store.persist(output_path)
    
    logger.info("Vector store built successfully")


def main():
    """Main entry point for vector store building."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Vector Store")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings directory")
    parser.add_argument("--chunks", required=True, help="Path to chunks Parquet directory")
    parser.add_argument("--output", required=True, help="Path to save vector store")
    parser.add_argument("--index-type", choices=["Flat", "IVF", "HNSW"], default="Flat",
                       help="FAISS index type (default: Flat)")
    parser.add_argument("--use-gpu", action="store_true", help="Use GPU if available")
    
    args = parser.parse_args()
    
    build_vector_store_from_embeddings(
        embeddings_path=args.embeddings,
        chunks_path=args.chunks,
        output_path=args.output,
        index_type=args.index_type,
        use_gpu=args.use_gpu
    )


if __name__ == "__main__":
    main()
