# Data Directory

This directory contains all data files for the RAG pipeline.

## Structure

- `raw/` - Raw input documents (PDFs, TXT files)
- `processed/` - Processed documents in Parquet format
  - `docs/` - Parsed and normalized documents
  - `chunks/` - Chunked documents
- `embeddings/` - Generated embeddings and vector indexes

## Usage

### Ingestion

Place raw documents in `data/raw/` and run the ingestion pipeline:

```bash
python spark/ingest.py --input data/raw --output data/processed/docs
```

### Output Files

After ingestion, you'll find:
- `data/processed/docs/*.parquet` - Processed documents with schema:
  - doc_id, source_path, text, hash, file_type, file_size_bytes, page_count, ingestion_timestamp, parser_version
- `data/processed/docs/metrics.json` - Ingestion metrics (throughput, failures, etc.)
- `data/processed/docs/failed_docs.json` - Failed document logs (if any)

## Sample Data

A sample text file is provided in `data/raw/sample.txt` for testing the ingestion pipeline.

## Note

Data files are excluded from version control via `.gitignore`.
