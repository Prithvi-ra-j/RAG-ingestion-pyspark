# Document Ingestion Pipeline - Usage Guide

## Overview

The document ingestion pipeline (`ingest.py`) processes raw documents (PDFs, TXT, logs) and prepares them for the RAG system.

## Features

- **Multi-format support**: PDF, TXT, LOG, MD files
- **Robust parsing**: Error handling for corrupted files
- **Encoding detection**: UTF-8 with Latin-1 fallback
- **Text normalization**: Removes non-printable chars, normalizes whitespace
- **Deduplication**: SHA256 hash-based duplicate detection
- **Checkpointing**: Progress saved every 10k documents
- **Resumption**: Skip already-processed documents
- **Metrics tracking**: Throughput, failures, processing time

## Requirements

Ensure dependencies are installed:

```bash
pip install -r requirements.txt
```

## Basic Usage

### Command Line

```bash
python spark/ingest.py --input data/raw --output data/processed/docs
```

### Arguments

- `--input`: Directory containing raw documents
- `--output`: Directory to save processed Parquet files

## Programmatic Usage

```python
from ingest import DocumentIngestion

# Initialize pipeline
ingestion = DocumentIngestion()

# Run ingestion
ingestion.run(
    input_path="data/raw",
    output_path="data/processed/docs"
)

# Stop Spark session
ingestion.spark.stop()
```

## Output Files

### Parquet Files

Processed documents are saved in Parquet format with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| doc_id | string | Unique document identifier (UUID) |
| source_path | string | Original file path |
| text | string | Extracted and normalized text |
| hash | string | SHA256 hash for deduplication |
| file_type | string | File extension (pdf, txt, log, md) |
| file_size_bytes | int | Original file size |
| page_count | int | Number of pages (PDFs) or 1 (text files) |
| ingestion_timestamp | timestamp | Processing timestamp |
| parser_version | string | Parser version for reproducibility |

### Metrics File

`metrics.json` contains:

```json
{
  "documents_processed": 1000,
  "parse_failures": 5,
  "duplicates_removed": 50,
  "processing_time_seconds": 120.5,
  "documents_per_minute": 497.93,
  "start_time": "2026-01-15T10:00:00",
  "end_time": "2026-01-15T10:02:00"
}
```

### Failed Documents Log

`failed_docs.json` logs unparseable documents:

```json
[
  {
    "source_path": "/path/to/corrupted.pdf",
    "file_type": "pdf",
    "error": "PDF read error: Invalid PDF structure",
    "timestamp": "2026-01-15T10:00:30"
  }
]
```

## Performance

### Target Metrics

- **Throughput**: 1000+ documents/minute on 4-node Spark cluster
- **Checkpointing**: Every 10k documents
- **Memory**: 4GB per executor

### Optimization Tips

1. **Adjust Spark configuration** for your cluster:
   ```python
   spark = SparkSession.builder \
       .config("spark.executor.memory", "8g") \
       .config("spark.executor.cores", "4") \
       .getOrCreate()
   ```

2. **Tune checkpoint interval** for large datasets:
   ```python
   ingestion.save_parquet(df, output_path, checkpoint_interval=5000)
   ```

3. **Enable resumption** to skip already-processed documents:
   ```python
   ingestion.save_parquet(df, output_path, resume=True)
   ```

## Testing

Run the validation script to test basic functionality:

```bash
python spark/test_ingest.py
```

This will:
- Test text normalization
- Test hash generation
- Test document ID generation
- Test file parsing
- Run a full pipeline with sample data
- Verify metrics output

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'pyspark'`
- **Solution**: Install dependencies: `pip install -r requirements.txt`

**Issue**: PDF parsing fails with "Invalid PDF structure"
- **Solution**: Check `failed_docs.json` for details. Corrupted PDFs are logged and skipped.

**Issue**: Out of memory errors
- **Solution**: Reduce executor memory or increase cluster resources

**Issue**: Slow processing
- **Solution**: 
  - Increase Spark parallelism: `.config("spark.sql.shuffle.partitions", "400")`
  - Add more executor cores
  - Check for disk I/O bottlenecks

## Next Steps

After ingestion, proceed to:

1. **Chunking**: `python spark/chunk.py --input data/processed/docs --output data/processed/chunks`
2. **Embedding**: `python spark/embed.py --input data/processed/chunks --output data/embeddings`
3. **RAG API**: Start the API server to query documents

## Requirements Mapping

This implementation satisfies:

- **Requirement 1.1**: Document parsing with error handling
- **Requirement 1.2**: Text normalization
- **Requirement 1.3**: Deduplication using content hash
- **Requirement 1.4**: Unique document identifiers
- **Requirement 1.5**: Parquet persistence with metadata
- **Requirement 6.1**: Metrics tracking and reporting
