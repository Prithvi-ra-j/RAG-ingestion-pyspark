"""
Distributed Chunking Engine (Spark UDFs)

This module implements document chunking strategies using PySpark UDFs
for distributed processing of large document corpora.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""

import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf, col, explode, lit, current_timestamp, size as array_size
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    ArrayType, TimestampType
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Chunk:
    """Represents a text chunk with metadata."""
    
    def __init__(self, chunk_id: str, doc_id: str, chunk_text: str, 
                 chunk_index: int, char_start: int, char_end: int):
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.chunk_text = chunk_text
        self.chunk_index = chunk_index
        self.char_start = char_start
        self.char_end = char_end
    
    def to_dict(self) -> Dict:
        """Convert chunk to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "chunk_text": self.chunk_text,
            "chunk_index": self.chunk_index,
            "char_start": self.char_start,
            "char_end": self.char_end
        }


class ChunkingEngine:
    """
    Distributed chunking engine using PySpark.
    
    Implements fixed-size and semantic chunking strategies with
    configurable parameters and metadata tracking.
    """
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """
        Initialize the chunking engine.
        
        Args:
            spark: Optional SparkSession. If None, creates a new session.
        """
        self.spark = spark or self._create_spark_session()
        self.metrics = {
            "total_chunks": 0,
            "total_documents": 0,
            "average_chunk_size": 0.0,
            "chunks_per_second": 0.0,
            "processing_time_seconds": 0.0,
            "start_time": None,
            "end_time": None
        }
    
    def _create_spark_session(self) -> SparkSession:
        """Create and configure SparkSession for chunking."""
        return (SparkSession.builder
                .appName("ChunkingEngine")
                .config("spark.executor.memory", "4g")
                .config("spark.executor.cores", "2")
                .config("spark.sql.shuffle.partitions", "200")
                .config("spark.sql.adaptive.enabled", "true")
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .getOrCreate())
    
    def chunk_fixed_size(self, text: str, doc_id: str, chunk_size: int = 512, 
                        overlap: int = 50) -> List[Dict]:
        """
        Split text into fixed-size chunks with overlap.
        
        Args:
            text: Text content to chunk
            doc_id: Document ID for metadata
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of chunk dictionaries with metadata
        """
        if not text or len(text) == 0:
            return []
        
        chunks = []
        text_length = len(text)
        chunk_index = 0
        start = 0
        
        while start < text_length:
            # Calculate end position
            end = min(start + chunk_size, text_length)
            
            # Extract chunk text
            chunk_text = text[start:end]
            
            # Skip empty chunks
            if chunk_text.strip():
                # Generate unique chunk ID
                chunk_id = str(uuid.uuid4())
                
                # Create chunk object
                chunk = Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    chunk_text=chunk_text,
                    chunk_index=chunk_index,
                    char_start=start,
                    char_end=end
                )
                
                chunks.append(chunk.to_dict())
                chunk_index += 1
            
            # Move to next chunk with overlap
            # If we're at the end, break to avoid infinite loop
            if end >= text_length:
                break
            
            start = end - overlap
            
            # Ensure we make progress
            if start <= chunks[-1]["char_start"] if chunks else False:
                start = end
        
        return chunks
    
    def create_fixed_size_chunking_udf(self, chunk_size: int = 512, 
                                       overlap: int = 50):
        """
        Create Spark UDF for fixed-size chunking.
        
        Args:
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            
        Returns:
            Spark UDF for chunking
        """
        # Define the schema for chunk output
        chunk_schema = ArrayType(
            StructType([
                StructField("chunk_id", StringType(), False),
                StructField("doc_id", StringType(), False),
                StructField("chunk_text", StringType(), False),
                StructField("chunk_index", IntegerType(), False),
                StructField("char_start", IntegerType(), False),
                StructField("char_end", IntegerType(), False)
            ])
        )
        
        # Create UDF with closure over chunk_size and overlap
        def chunking_func(text: str, doc_id: str) -> List[Dict]:
            return self.chunk_fixed_size(text, doc_id, chunk_size, overlap)
        
        return udf(chunking_func, chunk_schema)
    
    def load_documents(self, input_path: str) -> DataFrame:
        """
        Load processed documents from Parquet.
        
        Args:
            input_path: Path to Parquet files with processed documents
            
        Returns:
            DataFrame with documents
        """
        logger.info(f"Loading documents from {input_path}")
        
        try:
            df = self.spark.read.parquet(input_path)
            doc_count = df.count()
            logger.info(f"Loaded {doc_count} documents")
            return df
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            raise
    
    def chunk_documents(self, df: DataFrame, chunk_size: int = 512, 
                       overlap: int = 50, strategy: str = "fixed") -> DataFrame:
        """
        Chunk documents using specified strategy.
        
        Args:
            df: DataFrame with documents
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            strategy: Chunking strategy ("fixed" or "semantic")
            
        Returns:
            DataFrame with chunks
        """
        logger.info(f"Starting chunking with strategy: {strategy}, size: {chunk_size}, overlap: {overlap}")
        
        if strategy == "fixed":
            # Create chunking UDF
            chunking_udf = self.create_fixed_size_chunking_udf(chunk_size, overlap)
            
            # Apply chunking UDF
            chunked_df = df.withColumn(
                "chunks",
                chunking_udf(col("text"), col("doc_id"))
            )
            
            # Explode chunks array into separate rows
            exploded_df = chunked_df.select(
                col("doc_id"),
                explode(col("chunks")).alias("chunk")
            )
            
            # Extract chunk fields
            result_df = exploded_df.select(
                col("chunk.chunk_id").alias("chunk_id"),
                col("chunk.doc_id").alias("doc_id"),
                col("chunk.chunk_text").alias("chunk_text"),
                col("chunk.chunk_index").alias("chunk_index"),
                col("chunk.char_start").alias("char_start"),
                col("chunk.char_end").alias("char_end")
            )
            
            # Add chunking strategy
            result_df = result_df.withColumn("chunking_strategy", lit(strategy))
            
            logger.info(f"Chunking complete. Generated {result_df.count()} chunks")
            
            return result_df
        else:
            raise NotImplementedError(f"Chunking strategy '{strategy}' not implemented")

    
    def calculate_token_count(self, text: str) -> int:
        """
        Approximate token count for a text chunk.
        
        Uses simple whitespace-based approximation: ~1.3 tokens per word.
        
        Args:
            text: Text content
            
        Returns:
            Approximate token count
        """
        if not text:
            return 0
        
        # Simple approximation: count words and multiply by 1.3
        word_count = len(text.split())
        return int(word_count * 1.3)
    
    def add_chunk_metadata(self, chunks_df: DataFrame, docs_df: DataFrame) -> DataFrame:
        """
        Join chunk data with document metadata and add token counts.
        
        Args:
            chunks_df: DataFrame with chunks
            docs_df: DataFrame with document metadata
            
        Returns:
            DataFrame with enriched chunk metadata
        """
        logger.info("Adding chunk metadata")
        
        # Create UDF for token counting
        token_count_udf = udf(self.calculate_token_count, IntegerType())
        
        # Add token count to chunks
        chunks_with_tokens = chunks_df.withColumn(
            "token_count",
            token_count_udf(col("chunk_text"))
        )
        
        # Join with document metadata (select relevant fields)
        docs_metadata = docs_df.select(
            col("doc_id"),
            col("source_path"),
            col("file_type"),
            col("ingestion_timestamp")
        )
        
        # Perform join
        enriched_df = chunks_with_tokens.join(
            docs_metadata,
            on="doc_id",
            how="left"
        )
        
        logger.info("Chunk metadata added successfully")
        
        return enriched_df
    
    def save_chunks(self, df: DataFrame, output_path: str):
        """
        Save chunks to Parquet with partitioning by chunking_strategy.
        
        Args:
            df: DataFrame with chunks
            output_path: Output directory path
        """
        logger.info(f"Saving chunks to Parquet at {output_path}")
        
        # Select final schema columns in order
        final_df = df.select(
            "chunk_id",
            "doc_id",
            "chunk_text",
            "chunk_index",
            "char_start",
            "char_end",
            "token_count",
            "chunking_strategy",
            "source_path",
            "file_type",
            "ingestion_timestamp"
        )
        
        # Write to Parquet with partitioning by chunking_strategy
        (final_df
         .write
         .mode("overwrite")
         .partitionBy("chunking_strategy")
         .parquet(output_path))
        
        logger.info(f"Successfully saved chunks to Parquet")
    
    def calculate_metrics(self, chunks_df: DataFrame, docs_df: DataFrame, 
                         processing_time: float):
        """
        Calculate and store chunking metrics.
        
        Args:
            chunks_df: DataFrame with chunks
            docs_df: DataFrame with documents
            processing_time: Processing time in seconds
        """
        logger.info("Calculating chunking metrics")
        
        # Count chunks and documents
        total_chunks = chunks_df.count()
        total_documents = docs_df.count()
        
        # Calculate average chunk size
        avg_chunk_size = chunks_df.agg({"token_count": "avg"}).collect()[0][0]
        
        # Calculate chunks per second
        chunks_per_second = total_chunks / processing_time if processing_time > 0 else 0
        
        # Update metrics
        self.metrics["total_chunks"] = total_chunks
        self.metrics["total_documents"] = total_documents
        self.metrics["average_chunk_size"] = round(avg_chunk_size, 2) if avg_chunk_size else 0.0
        self.metrics["chunks_per_second"] = round(chunks_per_second, 2)
        self.metrics["processing_time_seconds"] = round(processing_time, 2)
        
        logger.info(f"Metrics calculated: {total_chunks} chunks from {total_documents} documents")
        logger.info(f"Average chunk size: {self.metrics['average_chunk_size']} tokens")
        logger.info(f"Throughput: {self.metrics['chunks_per_second']} chunks/sec")
    
    def save_metrics(self, output_path: str):
        """
        Save chunking metrics to JSON file.
        
        Args:
            output_path: Directory to save metrics.json
        """
        metrics_path = os.path.join(output_path, "metrics.json")
        os.makedirs(output_path, exist_ok=True)
        
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Saved metrics to {metrics_path}")
    
    def run(self, input_path: str, output_path: str, chunk_size: int = 512, 
            overlap: int = 50, strategy: str = "fixed"):
        """
        Run the complete chunking pipeline.
        
        Args:
            input_path: Directory containing processed documents (Parquet)
            output_path: Directory to save chunks (Parquet)
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks
            strategy: Chunking strategy ("fixed" or "semantic")
        """
        self.metrics["start_time"] = datetime.now().isoformat()
        start_time = datetime.now()
        
        try:
            # Load documents
            docs_df = self.load_documents(input_path)
            
            # Chunk documents
            chunks_df = self.chunk_documents(docs_df, chunk_size, overlap, strategy)
            
            # Add metadata
            enriched_chunks_df = self.add_chunk_metadata(chunks_df, docs_df)
            
            # Save chunks
            self.save_chunks(enriched_chunks_df, output_path)
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            self.metrics["end_time"] = end_time.isoformat()
            
            # Calculate and save metrics
            self.calculate_metrics(chunks_df, docs_df, processing_time)
            self.save_metrics(output_path)
            
            logger.info("Chunking pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Chunking pipeline failed: {e}")
            raise


def main():
    """Main entry point for chunking engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed Chunking Engine")
    parser.add_argument("--input", required=True, help="Input directory with processed documents (Parquet)")
    parser.add_argument("--output", required=True, help="Output directory for chunks (Parquet)")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in characters (default: 512)")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap between chunks in characters (default: 50)")
    parser.add_argument("--strategy", choices=["fixed", "semantic"], default="fixed", 
                       help="Chunking strategy (default: fixed)")
    
    args = parser.parse_args()
    
    # Run chunking
    engine = ChunkingEngine()
    engine.run(
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        strategy=args.strategy
    )
    
    # Stop Spark session
    engine.spark.stop()


if __name__ == "__main__":
    main()
