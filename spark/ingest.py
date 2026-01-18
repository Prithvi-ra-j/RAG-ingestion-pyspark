"""
Document Ingestion Pipeline (PySpark)

This module implements the document parsing, normalization, and deduplication
pipeline using PySpark for distributed processing.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
"""

import os
import json
import hashlib
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf, col, lit, current_timestamp
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, 
    TimestampType, BooleanType
)

import PyPDF2
import chardet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Parser version for reproducibility
PARSER_VERSION = "1.0.0"


class DocumentIngestion:
    """
    Document ingestion pipeline using PySpark.
    
    Handles parsing, normalization, deduplication, and persistence
    of documents to Parquet format.
    """
    
    def __init__(self, spark: Optional[SparkSession] = None):
        """
        Initialize the ingestion pipeline.
        
        Args:
            spark: Optional SparkSession. If None, creates a new session.
        """
        self.spark = spark or self._create_spark_session()
        self.failed_docs: List[Dict] = []
        self.metrics = {
            "documents_processed": 0,
            "parse_failures": 0,
            "duplicates_removed": 0,
            "processing_time_seconds": 0.0,
            "start_time": None,
            "end_time": None
        }
    
    def _create_spark_session(self) -> SparkSession:
        """Create and configure SparkSession for document ingestion."""
        return (SparkSession.builder
                .appName("DocumentIngestion")
                .config("spark.executor.memory", "4g")
                .config("spark.executor.cores", "2")
                .config("spark.sql.shuffle.partitions", "200")
                .config("spark.sql.adaptive.enabled", "true")
                .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .getOrCreate())
    
    def parse_pdf(self, file_path: str) -> Tuple[Optional[str], Optional[int], Optional[str]]:
        """
        Parse PDF file and extract text content.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (text_content, page_count, error_message)
            Returns (None, None, error_msg) if parsing fails
        """
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                page_count = len(pdf_reader.pages)
                
                text_parts = []
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num} from {file_path}: {e}")
                
                full_text = "\n".join(text_parts)
                return full_text, page_count, None
                
        except PyPDF2.errors.PdfReadError as e:
            error_msg = f"PDF read error: {str(e)}"
            logger.error(f"Failed to parse PDF {file_path}: {error_msg}")
            return None, None, error_msg
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"Failed to parse PDF {file_path}: {error_msg}")
            return None, None, error_msg
    
    def parse_txt(self, file_path: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse text file with encoding detection.
        
        Tries UTF-8 first, then Latin-1 as fallback.
        
        Args:
            file_path: Path to text file
            
        Returns:
            Tuple of (text_content, error_message)
            Returns (None, error_msg) if parsing fails
        """
        try:
            # Try UTF-8 first
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                return text, None
            except UnicodeDecodeError:
                # Fallback to Latin-1
                logger.info(f"UTF-8 failed for {file_path}, trying Latin-1")
                with open(file_path, 'r', encoding='latin-1') as file:
                    text = file.read()
                return text, None
                
        except Exception as e:
            error_msg = f"Failed to read file: {str(e)}"
            logger.error(f"Failed to parse TXT {file_path}: {error_msg}")
            return None, error_msg
    
    def parse_document(self, file_path: str) -> Dict:
        """
        Parse a document based on its file type.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Dictionary with parsed document data and metadata
        """
        file_type = os.path.splitext(file_path)[1].lower().lstrip('.')
        file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
        
        result = {
            "source_path": file_path,
            "file_type": file_type,
            "file_size_bytes": file_size,
            "text": None,
            "page_count": None,
            "parse_success": False,
            "error_message": None
        }
        
        if file_type == 'pdf':
            text, page_count, error = self.parse_pdf(file_path)
            result["text"] = text
            result["page_count"] = page_count
            result["error_message"] = error
            result["parse_success"] = text is not None
            
        elif file_type in ['txt', 'log', 'md']:
            text, error = self.parse_txt(file_path)
            result["text"] = text
            result["page_count"] = 1  # Text files are single "page"
            result["error_message"] = error
            result["parse_success"] = text is not None
            
        else:
            result["error_message"] = f"Unsupported file type: {file_type}"
            logger.warning(f"Unsupported file type for {file_path}")
        
        # Log failed documents
        if not result["parse_success"]:
            self.failed_docs.append({
                "source_path": file_path,
                "file_type": file_type,
                "error": result["error_message"],
                "timestamp": datetime.now().isoformat()
            })
            self.metrics["parse_failures"] += 1
        
        return result
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text by removing non-printable characters and normalizing whitespace.
        
        Args:
            text: Raw text content
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Remove non-printable characters (keep newlines, tabs, and printable chars)
        printable_text = ''.join(
            char for char in text 
            if char.isprintable() or char in ['\n', '\t']
        )
        
        # Normalize whitespace: collapse multiple spaces, normalize line breaks
        lines = printable_text.split('\n')
        normalized_lines = [' '.join(line.split()) for line in lines]
        normalized_text = '\n'.join(line for line in normalized_lines if line)
        
        return normalized_text
    
    def compute_hash(self, text: str) -> str:
        """
        Compute SHA256 hash of text content for deduplication.
        
        Args:
            text: Text content
            
        Returns:
            SHA256 hash as hexadecimal string
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def generate_doc_id(self) -> str:
        """
        Generate unique document ID using UUID.
        
        Returns:
            UUID string
        """
        return str(uuid.uuid4())
    
    def parse_documents(self, input_path: str) -> DataFrame:
        """
        Parse all documents in the input directory.
        
        Args:
            input_path: Directory containing documents to parse
            
        Returns:
            Spark DataFrame with parsed documents
        """
        logger.info(f"Starting document parsing from {input_path}")
        
        # Find all supported files
        supported_extensions = ['.pdf', '.txt', '.log', '.md']
        file_paths = []
        
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in supported_extensions):
                    file_paths.append(os.path.join(root, file))
        
        logger.info(f"Found {len(file_paths)} documents to process")
        
        # Create DataFrame from file paths
        files_df = self.spark.createDataFrame(
            [(path,) for path in file_paths],
            ["file_path"]
        )
        
        # Define UDF for parsing
        parse_udf = udf(self.parse_document, 
                       StructType([
                           StructField("source_path", StringType(), False),
                           StructField("file_type", StringType(), False),
                           StructField("file_size_bytes", IntegerType(), False),
                           StructField("text", StringType(), True),
                           StructField("page_count", IntegerType(), True),
                           StructField("parse_success", BooleanType(), False),
                           StructField("error_message", StringType(), True)
                       ]))
        
        # Parse documents
        parsed_df = files_df.withColumn("parsed", parse_udf(col("file_path")))
        
        # Expand struct columns
        result_df = parsed_df.select(
            col("parsed.source_path").alias("source_path"),
            col("parsed.file_type").alias("file_type"),
            col("parsed.file_size_bytes").alias("file_size_bytes"),
            col("parsed.text").alias("text"),
            col("parsed.page_count").alias("page_count"),
            col("parsed.parse_success").alias("parse_success")
        )
        
        # Filter to successfully parsed documents
        success_df = result_df.filter(col("parse_success") == True).drop("parse_success")
        
        logger.info(f"Successfully parsed {success_df.count()} documents")
        
        return success_df
    
    def normalize_and_deduplicate(self, df: DataFrame) -> DataFrame:
        """
        Normalize text and remove duplicates based on content hash.
        
        Args:
            df: DataFrame with parsed documents
            
        Returns:
            DataFrame with normalized, deduplicated documents
        """
        logger.info("Starting text normalization and deduplication")
        
        # Define UDFs
        normalize_udf = udf(self.normalize_text, StringType())
        hash_udf = udf(self.compute_hash, StringType())
        id_udf = udf(self.generate_doc_id, StringType())
        
        # Normalize text
        normalized_df = df.withColumn("text", normalize_udf(col("text")))
        
        # Compute hash
        hashed_df = normalized_df.withColumn("hash", hash_udf(col("text")))
        
        # Count before deduplication
        count_before = hashed_df.count()
        
        # Remove duplicates based on hash
        deduplicated_df = hashed_df.dropDuplicates(["hash"])
        
        # Count after deduplication
        count_after = deduplicated_df.count()
        duplicates_removed = count_before - count_after
        
        self.metrics["duplicates_removed"] = duplicates_removed
        logger.info(f"Removed {duplicates_removed} duplicate documents")
        
        # Add document IDs
        result_df = deduplicated_df.withColumn("doc_id", id_udf())
        
        return result_df
    
    def load_existing_hashes(self, output_path: str) -> set:
        """
        Load hashes of already-processed documents for resumption.
        
        Args:
            output_path: Output directory path
            
        Returns:
            Set of document hashes already processed
        """
        try:
            if os.path.exists(output_path):
                existing_df = self.spark.read.parquet(output_path)
                hashes = existing_df.select("hash").rdd.flatMap(lambda x: x).collect()
                logger.info(f"Found {len(hashes)} already-processed documents")
                return set(hashes)
        except Exception as e:
            logger.warning(f"Could not load existing documents: {e}")
        
        return set()
    
    def save_parquet(self, df: DataFrame, output_path: str, checkpoint_interval: int = 10000, resume: bool = True):
        """
        Save DataFrame to Parquet with date-based partitioning and checkpointing.
        
        Args:
            df: DataFrame to save
            output_path: Output directory path
            checkpoint_interval: Save checkpoint every N documents
            resume: If True, skip already-processed documents
        """
        logger.info(f"Saving documents to Parquet at {output_path}")
        
        # Load existing hashes for resumption
        if resume:
            existing_hashes = self.load_existing_hashes(output_path)
            if existing_hashes:
                # Filter out already-processed documents
                df = df.filter(~col("hash").isin(existing_hashes))
                logger.info(f"Resuming: {df.count()} new documents to process")
        
        # Add metadata columns
        final_df = (df
                   .withColumn("ingestion_timestamp", current_timestamp())
                   .withColumn("parser_version", lit(PARSER_VERSION)))
        
        # Select and order columns according to schema
        final_df = final_df.select(
            "doc_id",
            "source_path",
            "text",
            "hash",
            "file_type",
            "file_size_bytes",
            "page_count",
            "ingestion_timestamp",
            "parser_version"
        )
        
        # Determine write mode
        write_mode = "append" if resume and os.path.exists(output_path) else "overwrite"
        
        # Write to Parquet with date-based partitioning
        # Note: For large datasets, consider using coalesce to control file size
        doc_count = final_df.count()
        
        if doc_count > checkpoint_interval:
            # For large datasets, write in batches with checkpointing
            logger.info(f"Writing {doc_count} documents with checkpointing every {checkpoint_interval} docs")
            
            # Cache the dataframe for multiple passes
            final_df.cache()
            
            # Write in chunks
            total_written = 0
            batch_num = 0
            
            while total_written < doc_count:
                batch_df = final_df.limit(checkpoint_interval).offset(total_written)
                batch_count = batch_df.count()
                
                if batch_count == 0:
                    break
                
                batch_df.write.mode("append" if batch_num > 0 or write_mode == "append" else "overwrite").parquet(output_path)
                
                total_written += batch_count
                batch_num += 1
                logger.info(f"Checkpoint {batch_num}: Written {total_written}/{doc_count} documents")
            
            final_df.unpersist()
        else:
            # For smaller datasets, write all at once
            (final_df
             .write
             .mode(write_mode)
             .parquet(output_path))
        
        logger.info(f"Successfully saved {doc_count} documents to Parquet")
    
    def save_failed_docs(self, output_path: str):
        """
        Save failed document logs to JSON file.
        
        Args:
            output_path: Directory to save failed_docs.json
        """
        if self.failed_docs:
            failed_docs_path = os.path.join(output_path, "failed_docs.json")
            os.makedirs(output_path, exist_ok=True)
            
            with open(failed_docs_path, 'w') as f:
                json.dump(self.failed_docs, f, indent=2)
            
            logger.info(f"Saved {len(self.failed_docs)} failed document logs to {failed_docs_path}")
    
    def save_metrics(self, output_path: str):
        """
        Save ingestion metrics to JSON file.
        
        Args:
            output_path: Directory to save metrics.json
        """
        metrics_path = os.path.join(output_path, "metrics.json")
        os.makedirs(output_path, exist_ok=True)
        
        # Calculate throughput
        if self.metrics["processing_time_seconds"] > 0:
            docs_per_minute = (self.metrics["documents_processed"] / 
                             self.metrics["processing_time_seconds"]) * 60
            self.metrics["documents_per_minute"] = round(docs_per_minute, 2)
        
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        logger.info(f"Saved metrics to {metrics_path}")
        logger.info(f"Throughput: {self.metrics.get('documents_per_minute', 0)} docs/min")
    
    def run(self, input_path: str, output_path: str):
        """
        Run the complete ingestion pipeline.
        
        Args:
            input_path: Directory containing raw documents
            output_path: Directory to save processed documents
        """
        self.metrics["start_time"] = datetime.now().isoformat()
        start_time = datetime.now()
        
        try:
            # Parse documents
            parsed_df = self.parse_documents(input_path)
            
            # Normalize and deduplicate
            processed_df = self.normalize_and_deduplicate(parsed_df)
            
            # Update metrics
            self.metrics["documents_processed"] = processed_df.count()
            
            # Save to Parquet
            self.save_parquet(processed_df, output_path)
            
            # Save failed documents log
            self.save_failed_docs(output_path)
            
        finally:
            # Calculate processing time
            end_time = datetime.now()
            self.metrics["end_time"] = end_time.isoformat()
            self.metrics["processing_time_seconds"] = (end_time - start_time).total_seconds()
            
            # Save metrics
            self.save_metrics(output_path)
            
            logger.info("Ingestion pipeline completed")


def main():
    """Main entry point for document ingestion."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Document Ingestion Pipeline")
    parser.add_argument("--input", required=True, help="Input directory with raw documents")
    parser.add_argument("--output", required=True, help="Output directory for processed documents")
    
    args = parser.parse_args()
    
    # Run ingestion
    ingestion = DocumentIngestion()
    ingestion.run(args.input, args.output)
    
    # Stop Spark session
    ingestion.spark.stop()


if __name__ == "__main__":
    main()
