"""
Embedding Generation Pipeline

This module implements batch embedding generation using HuggingFace
SentenceTransformers with memory optimization for 8GB RAM environments.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5
"""

import os
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

from sentence_transformers import SentenceTransformer
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """
    Embedding generation pipeline optimized for 8GB RAM CPU environments.
    
    Uses smaller models (384-dim) with batch size 16 and memory-mapped
    arrays to avoid loading all embeddings into RAM.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        """
        Initialize the embedding pipeline.
        
        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2 for 8GB RAM)
            device: Device to use ('cpu', 'cuda', or None for auto-detect)
        """
        self.model_name = model_name
        self.device = device or self._detect_device()
        self.model = None
        self.embedding_dim = None
        self.metrics = {
            "total_embeddings": 0,
            "processing_time_seconds": 0.0,
            "chunks_per_second": 0.0,
            "model_name": model_name,
            "device": self.device,
            "start_time": None,
            "end_time": None
        }
    
    def _detect_device(self) -> str:
        """
        Detect available device (CUDA or CPU).
        
        Returns:
            Device string ('cuda' or 'cpu')
        """
        if torch.cuda.is_available():
            logger.info("CUDA available, using GPU")
            return "cuda"
        else:
            logger.info("CUDA not available, using CPU")
            return "cpu"
    
    def load_model(self) -> SentenceTransformer:
        """
        Load HuggingFace SentenceTransformer model.
        
        Returns:
            Loaded model
        """
        logger.info(f"Loading model: {self.model_name} on {self.device}")
        
        try:
            model = SentenceTransformer(self.model_name, device=self.device)
            
            # Get embedding dimension
            self.embedding_dim = model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            
            # Validate model
            test_embedding = model.encode(["test"], show_progress_bar=False)
            assert test_embedding.shape[1] == self.embedding_dim, "Model validation failed"
            
            self.model = model
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def load_chunks_from_parquet(self, input_path: str) -> Tuple[List[str], List[str]]:
        """
        Load chunks from Parquet files.
        
        Args:
            input_path: Path to Parquet files with chunks
            
        Returns:
            Tuple of (chunk_ids, chunk_texts)
        """
        logger.info(f"Loading chunks from {input_path}")
        
        try:
            # Use pyarrow to read Parquet (more memory efficient than Spark for this)
            import pyarrow.parquet as pq
            
            # Read Parquet table
            table = pq.read_table(input_path, columns=["chunk_id", "chunk_text"])
            
            # Convert to lists
            chunk_ids = table.column("chunk_id").to_pylist()
            chunk_texts = table.column("chunk_text").to_pylist()
            
            logger.info(f"Loaded {len(chunk_ids)} chunks")
            
            return chunk_ids, chunk_texts
            
        except Exception as e:
            logger.error(f"Failed to load chunks: {e}")
            raise
    
    def batch_embed(self, texts: List[str], batch_size: int = 16, 
                   show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings in batches with memory optimization.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size (default: 16 for 8GB RAM)
            show_progress: Show progress bar
            
        Returns:
            NumPy array of embeddings (shape: [N, embedding_dim])
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info(f"Generating embeddings for {len(texts)} texts with batch_size={batch_size}")
        
        try:
            # Generate embeddings with automatic batching
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=False  # Keep raw embeddings
            )
            
            logger.info(f"Generated embeddings with shape: {embeddings.shape}")
            
            return embeddings
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM error with batch_size={batch_size}, reducing...")
                
                # Reduce batch size and retry
                new_batch_size = max(batch_size // 2, 1)
                if new_batch_size < batch_size:
                    # Clear cache
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    
                    return self.batch_embed(texts, batch_size=new_batch_size, show_progress=show_progress)
                else:
                    raise
            else:
                raise
    
    def persist_embeddings(self, embeddings: np.ndarray, output_path: str, 
                          use_memmap: bool = True):
        """
        Save embeddings as memory-mapped NumPy array.
        
        Args:
            embeddings: NumPy array of embeddings
            output_path: Output directory path
            use_memmap: Use memory-mapped array (recommended for large datasets)
        """
        os.makedirs(output_path, exist_ok=True)
        embeddings_file = os.path.join(output_path, "embeddings.npy")
        
        logger.info(f"Saving embeddings to {embeddings_file}")
        
        if use_memmap:
            # Create memory-mapped array
            fp = np.memmap(
                embeddings_file,
                dtype='float32',
                mode='w+',
                shape=embeddings.shape
            )
            fp[:] = embeddings[:]
            del fp  # Flush to disk
        else:
            # Save as regular NumPy array
            np.save(embeddings_file, embeddings)
        
        logger.info(f"Saved {embeddings.shape[0]} embeddings")
    
    def create_index_mapping(self, chunk_ids: List[str], output_path: str):
        """
        Create mapping from chunk_id to embedding array index.
        
        Args:
            chunk_ids: List of chunk IDs
            output_path: Output directory path
        """
        os.makedirs(output_path, exist_ok=True)
        
        # Create metadata with index mapping
        metadata = []
        for idx, chunk_id in enumerate(chunk_ids):
            metadata.append({
                "chunk_id": chunk_id,
                "embedding_index": idx,
                "model_name": self.model_name,
                "embedding_dim": self.embedding_dim,
                "embedding_timestamp": datetime.now().isoformat()
            })
        
        # Save as JSON for simplicity (could use Parquet for larger datasets)
        metadata_file = os.path.join(output_path, "embedding_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata for {len(metadata)} embeddings")
    
    def checkpoint(self, embeddings: np.ndarray, chunk_ids: List[str], 
                  output_path: str, checkpoint_num: int):
        """
        Save checkpoint during embedding generation.
        
        Args:
            embeddings: Embeddings generated so far
            chunk_ids: Chunk IDs for embeddings
            output_path: Output directory path
            checkpoint_num: Checkpoint number
        """
        checkpoint_dir = os.path.join(output_path, f"checkpoint_{checkpoint_num}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save embeddings
        self.persist_embeddings(embeddings, checkpoint_dir, use_memmap=True)
        
        # Save metadata
        self.create_index_mapping(chunk_ids, checkpoint_dir)
        
        logger.info(f"Checkpoint {checkpoint_num} saved: {len(embeddings)} embeddings")
    
    def load_checkpoint(self, output_path: str) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        """
        Load the latest checkpoint if it exists.
        
        Args:
            output_path: Output directory path
            
        Returns:
            Tuple of (embeddings, chunk_ids) or (None, None) if no checkpoint
        """
        # Find latest checkpoint
        checkpoints = [d for d in os.listdir(output_path) if d.startswith("checkpoint_")]
        if not checkpoints:
            return None, None
        
        # Sort by checkpoint number
        checkpoints.sort(key=lambda x: int(x.split("_")[1]))
        latest_checkpoint = checkpoints[-1]
        checkpoint_dir = os.path.join(output_path, latest_checkpoint)
        
        logger.info(f"Loading checkpoint: {latest_checkpoint}")
        
        # Load embeddings
        embeddings_file = os.path.join(checkpoint_dir, "embeddings.npy")
        embeddings = np.load(embeddings_file, mmap_mode='r')
        
        # Load metadata
        metadata_file = os.path.join(checkpoint_dir, "embedding_metadata.json")
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        chunk_ids = [m["chunk_id"] for m in metadata]
        
        logger.info(f"Loaded checkpoint with {len(embeddings)} embeddings")
        
        return embeddings, chunk_ids
    
    def calculate_metrics(self, total_embeddings: int, processing_time: float):
        """
        Calculate and store embedding metrics.
        
        Args:
            total_embeddings: Total number of embeddings generated
            processing_time: Processing time in seconds
        """
        chunks_per_second = total_embeddings / processing_time if processing_time > 0 else 0
        
        self.metrics["total_embeddings"] = total_embeddings
        self.metrics["processing_time_seconds"] = round(processing_time, 2)
        self.metrics["chunks_per_second"] = round(chunks_per_second, 2)
        
        logger.info(f"Generated {total_embeddings} embeddings in {processing_time:.2f}s")
        logger.info(f"Throughput: {chunks_per_second:.2f} chunks/sec")
    
    def save_metrics(self, output_path: str):
        """
        Save embedding metrics to JSON file.
        
        Args:
            output_path: Directory to save metrics.json
        """
        metrics_path = os.path.join(output_path, "metrics.json")
        os.makedirs(output_path, exist_ok=True)
        
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Saved metrics to {metrics_path}")
    
    def run(self, input_path: str, output_path: str, batch_size: int = 16, 
            checkpoint_interval: int = 5000, resume: bool = True):
        """
        Run the complete embedding pipeline.
        
        Args:
            input_path: Directory containing chunks (Parquet)
            output_path: Directory to save embeddings
            batch_size: Batch size for embedding generation (default: 16 for 8GB RAM)
            checkpoint_interval: Save checkpoint every N embeddings (default: 5000)
            resume: Resume from checkpoint if available
        """
        self.metrics["start_time"] = datetime.now().isoformat()
        start_time = datetime.now()
        
        try:
            # Load model
            self.load_model()
            
            # Load chunks
            chunk_ids, chunk_texts = self.load_chunks_from_parquet(input_path)
            
            # Check for existing checkpoint
            start_idx = 0
            if resume and os.path.exists(output_path):
                checkpoint_embeddings, checkpoint_ids = self.load_checkpoint(output_path)
                if checkpoint_embeddings is not None:
                    start_idx = len(checkpoint_embeddings)
                    logger.info(f"Resuming from checkpoint: {start_idx} embeddings already processed")
            
            # Process remaining chunks
            if start_idx < len(chunk_texts):
                remaining_texts = chunk_texts[start_idx:]
                remaining_ids = chunk_ids[start_idx:]
                
                logger.info(f"Processing {len(remaining_texts)} remaining chunks")
                
                # Generate embeddings with checkpointing
                all_embeddings = []
                checkpoint_num = start_idx // checkpoint_interval
                
                for i in range(0, len(remaining_texts), checkpoint_interval):
                    batch_texts = remaining_texts[i:i+checkpoint_interval]
                    batch_ids = remaining_ids[i:i+checkpoint_interval]
                    
                    # Generate embeddings for this batch
                    batch_embeddings = self.batch_embed(batch_texts, batch_size=batch_size)
                    all_embeddings.append(batch_embeddings)
                    
                    # Save checkpoint
                    checkpoint_num += 1
                    combined_embeddings = np.vstack(all_embeddings)
                    combined_ids = chunk_ids[start_idx:start_idx+len(combined_embeddings)]
                    self.checkpoint(combined_embeddings, combined_ids, output_path, checkpoint_num)
                    
                    # Clear cache periodically
                    if self.device == "cuda":
                        torch.cuda.empty_cache()
                    
                    logger.info(f"Progress: {start_idx + len(combined_embeddings)}/{len(chunk_texts)} embeddings")
                
                # Combine all embeddings
                final_embeddings = np.vstack(all_embeddings)
                
                # If we had a checkpoint, combine with previous embeddings
                if start_idx > 0:
                    checkpoint_embeddings, _ = self.load_checkpoint(output_path)
                    final_embeddings = np.vstack([checkpoint_embeddings, final_embeddings])
            else:
                # All embeddings already generated
                final_embeddings, _ = self.load_checkpoint(output_path)
                logger.info("All embeddings already generated")
            
            # Save final embeddings
            self.persist_embeddings(final_embeddings, output_path, use_memmap=True)
            self.create_index_mapping(chunk_ids, output_path)
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            self.metrics["end_time"] = end_time.isoformat()
            
            # Calculate and save metrics
            self.calculate_metrics(len(final_embeddings), processing_time)
            self.save_metrics(output_path)
            
            logger.info("Embedding pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Embedding pipeline failed: {e}")
            raise


def main():
    """Main entry point for embedding pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Embedding Generation Pipeline")
    parser.add_argument("--input", required=True, help="Input directory with chunks (Parquet)")
    parser.add_argument("--output", required=True, help="Output directory for embeddings")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", 
                       help="HuggingFace model name (default: all-MiniLM-L6-v2)")
    parser.add_argument("--batch-size", type=int, default=16, 
                       help="Batch size for embedding generation (default: 16)")
    parser.add_argument("--checkpoint-interval", type=int, default=5000,
                       help="Save checkpoint every N embeddings (default: 5000)")
    parser.add_argument("--no-resume", action="store_true",
                       help="Don't resume from checkpoint")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None,
                       help="Device to use (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Run embedding pipeline
    pipeline = EmbeddingPipeline(model_name=args.model, device=args.device)
    pipeline.run(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
        checkpoint_interval=args.checkpoint_interval,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()
