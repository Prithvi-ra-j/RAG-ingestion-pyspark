# Ingestion Pipeline Integration Tests

## Overview

The `test_ingest.py` file contains comprehensive integration tests for the document ingestion pipeline, covering all requirements from the specification.

## Test Coverage

### Requirements Coverage

- **Requirement 1.1**: Document parsing (PDF and TXT files)
- **Requirement 1.2**: Text normalization
- **Requirement 1.3**: Deduplication based on content hash
- **Requirement 1.4**: Unique document ID generation
- **Requirement 1.5**: Parquet persistence with metadata

### Test Cases

1. **Text Normalization** - Validates whitespace and newline normalization
2. **Hash Consistency** - Ensures SHA256 hashing is deterministic
3. **Document ID Uniqueness** - Verifies UUID generation produces unique IDs
4. **Parsing with Corrupted Files** - Tests error handling and logging for corrupted PDFs
5. **Deduplication** - Validates duplicate document removal
6. **Parquet Schema and Content** - Verifies output schema and data integrity
7. **Full Corpus Ingestion** - End-to-end test with 100 documents including:
   - 52 text files (50 unique + 2 special)
   - 40 PDF files
   - 3 corrupted files
   - 5 duplicate files

## Prerequisites

Install required dependencies:

```bash
pip install -r requirements.txt
```

Key dependencies:
- `pyspark==3.5.0`
- `PyPDF2==3.0.1`
- `chardet` (for encoding detection)

## Running the Tests

### Run All Tests

```bash
python spark/test_ingest.py
```

### Expected Output

```
======================================================================
INGESTION PIPELINE INTEGRATION TESTS
======================================================================

=== Test: Text Normalization ===
✓ Multiple spaces normalized
✓ Multiple newlines normalized
✓ Mixed whitespace normalized
✓ Empty string handled
✓ None handled

=== Test: Hash Consistency ===
✓ Same content produces same hash
✓ Different content produces different hash
✓ Hash is valid SHA256 format

=== Test: Document ID Uniqueness ===
✓ 100 generated IDs are all unique
✓ All IDs are valid UUID format

=== Test: Parsing with Corrupted Files ===
✓ Corrupted files handled correctly: 1 failures logged
✓ Valid documents processed: 2

=== Test: Deduplication ===
✓ Deduplication works: 3 duplicates removed
✓ Unique documents processed: 2

=== Test: Parquet Schema and Content ===
✓ Schema verified: all 9 expected columns present
✓ Content verified: doc_id, hash, text, and metadata are correct

=== Test: Full Corpus Ingestion (100 documents) ===
Creating test corpus...
✓ Created test corpus: 95 files (52 TXT, 40 PDF, 3 corrupted)
  Including 5 duplicate files for deduplication testing

📊 Ingestion Metrics:
  Documents processed: 87
  Parse failures: 3
  Duplicates removed: 5
  Processing time: X.XX seconds
  Throughput: XXX.XX docs/min

✓ Parquet output contains 87 documents
✓ Failed documents logged: 3 entries

✅ Full corpus ingestion test passed!

======================================================================
TEST SUMMARY: 7 passed, 0 failed
======================================================================

✅ ALL INTEGRATION TESTS PASSED!
```

## Test Details

### Test Corpus Creation

The `create_test_corpus()` function generates:
- 50 unique text files with varied content
- 5 duplicate text files (same content as document_000.txt)
- 40 blank PDF files
- 3 corrupted PDF files (invalid PDF format)
- 2 special text files (special characters, whitespace edge cases)

Total: 100 files

### Validation Checks

Each test validates specific aspects:

1. **Parsing Tests**
   - Valid files are parsed successfully
   - Corrupted files are logged to `failed_docs.json`
   - Error messages are descriptive

2. **Deduplication Tests**
   - Duplicate documents are identified by content hash
   - Only unique documents are persisted
   - Metrics accurately report duplicates removed

3. **Schema Tests**
   - All required columns are present in Parquet output
   - Data types match specification
   - Metadata fields are populated correctly

4. **Content Tests**
   - Document IDs are valid UUIDs
   - Hashes are valid SHA256 (64 hex characters)
   - Text content is preserved
   - File metadata is accurate

## Troubleshooting

### PySpark Not Found

If you see `ModuleNotFoundError: No module named 'pyspark'`:

```bash
pip install pyspark==3.5.0
```

### PyPDF2 Not Found

If you see `ModuleNotFoundError: No module named 'PyPDF2'`:

```bash
pip install PyPDF2==3.0.1
```

### Java Not Found

PySpark requires Java. If you see Java-related errors:

1. Install Java 8 or 11
2. Set `JAVA_HOME` environment variable
3. Add Java to your PATH

### Memory Issues

If tests fail with out-of-memory errors:

1. Reduce the test corpus size in `create_test_corpus()`
2. Adjust Spark memory settings in `DocumentIngestion._create_spark_session()`

## Performance Expectations

On a typical development machine (8GB RAM, 4 cores):

- **Text Normalization**: < 1 second
- **Hash Consistency**: < 1 second
- **Document ID Uniqueness**: < 1 second
- **Parsing with Corrupted Files**: 5-10 seconds
- **Deduplication**: 5-10 seconds
- **Parquet Schema and Content**: 5-10 seconds
- **Full Corpus Ingestion**: 30-60 seconds

Total test suite runtime: ~1-2 minutes

## Integration with CI/CD

To integrate these tests into a CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
- name: Run Integration Tests
  run: |
    pip install -r requirements.txt
    python spark/test_ingest.py
```

## Next Steps

After running these tests successfully:

1. Proceed to task 3.1: Implement chunking pipeline
2. Use the Parquet output from ingestion as input for chunking
3. Verify end-to-end pipeline with `experiments/` notebooks
