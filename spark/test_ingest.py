"""
Integration tests for document ingestion pipeline.

Tests cover:
- Parsing of PDF and TXT files (Requirement 1.1)
- Text normalization (Requirement 1.2)
- Deduplication (Requirement 1.3)
- Document ID generation (Requirement 1.4)
- Parquet persistence with metadata (Requirement 1.5)
- Error handling for corrupted files
"""

import os
import sys
import tempfile
import shutil
import json
from ingest import DocumentIngestion
from PyPDF2 import PdfWriter


def create_test_corpus(input_dir: str) -> dict:
    """
    Create a test corpus with 100 sample PDFs and TXT files.
    
    Returns:
        Dictionary with corpus statistics
    """
    print("Creating test corpus...")
    
    stats = {
        "txt_files": 0,
        "pdf_files": 0,
        "corrupted_files": 0,
        "duplicate_files": 0
    }
    
    # Create 50 text files with varied content
    for i in range(50):
        file_path = os.path.join(input_dir, f"document_{i:03d}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"Document {i}\n")
            f.write(f"This is test document number {i}.\n")
            f.write(f"It contains sample text for testing the ingestion pipeline.\n")
            f.write(f"Line 4 of document {i}.\n")
        stats["txt_files"] += 1
    
    # Create 5 duplicate text files (same content as document_000.txt)
    duplicate_content = "Document 0\nThis is test document number 0.\nIt contains sample text for testing the ingestion pipeline.\nLine 4 of document 0.\n"
    for i in range(5):
        file_path = os.path.join(input_dir, f"duplicate_{i}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(duplicate_content)
        stats["duplicate_files"] += 1
    
    # Create 40 simple PDF files
    for i in range(40):
        file_path = os.path.join(input_dir, f"document_{i:03d}.pdf")
        pdf_writer = PdfWriter()
        pdf_writer.add_blank_page(width=612, height=792)
        
        # Note: PyPDF2 doesn't easily support adding text to pages
        # For testing purposes, we create blank PDFs
        with open(file_path, 'wb') as f:
            pdf_writer.write(f)
        stats["pdf_files"] += 1
    
    # Create 3 corrupted PDF files
    for i in range(3):
        file_path = os.path.join(input_dir, f"corrupted_{i}.pdf")
        with open(file_path, 'wb') as f:
            f.write(b"This is not a valid PDF file content")
        stats["corrupted_files"] += 1
    
    # Create 2 text files with special characters and encoding issues
    special_file = os.path.join(input_dir, "special_chars.txt")
    with open(special_file, 'w', encoding='utf-8') as f:
        f.write("Special characters: é, ñ, ü, 中文\n")
        f.write("Symbols: @#$%^&*()\n")
    stats["txt_files"] += 1
    
    # Create a file with multiple spaces and newlines
    whitespace_file = os.path.join(input_dir, "whitespace.txt")
    with open(whitespace_file, 'w', encoding='utf-8') as f:
        f.write("  Multiple   spaces   here  \n\n\n\nMultiple newlines above\n")
    stats["txt_files"] += 1
    
    total_files = stats["txt_files"] + stats["pdf_files"] + stats["corrupted_files"]
    print(f"✓ Created test corpus: {total_files} files ({stats['txt_files']} TXT, {stats['pdf_files']} PDF, {stats['corrupted_files']} corrupted)")
    print(f"  Including {stats['duplicate_files']} duplicate files for deduplication testing")
    
    return stats


def test_parsing_with_corrupted_files():
    """Test parsing with corrupted files and verify error logging."""
    print("\n=== Test: Parsing with Corrupted Files ===")
    
    test_dir = tempfile.mkdtemp()
    input_dir = os.path.join(test_dir, "input")
    output_dir = os.path.join(test_dir, "output")
    os.makedirs(input_dir)
    
    try:
        # Create test files including corrupted ones
        # Valid text file
        with open(os.path.join(input_dir, "valid.txt"), 'w') as f:
            f.write("Valid document content")
        
        # Corrupted PDF
        with open(os.path.join(input_dir, "corrupted.pdf"), 'wb') as f:
            f.write(b"Not a valid PDF")
        
        # Valid PDF
        pdf_writer = PdfWriter()
        pdf_writer.add_blank_page(width=612, height=792)
        with open(os.path.join(input_dir, "valid.pdf"), 'wb') as f:
            pdf_writer.write(f)
        
        # Run ingestion
        ingestion = DocumentIngestion()
        ingestion.run(input_dir, output_dir)
        
        # Verify failed_docs.json exists and contains error
        failed_docs_path = os.path.join(output_dir, "failed_docs.json")
        assert os.path.exists(failed_docs_path), "failed_docs.json should exist"
        
        with open(failed_docs_path, 'r') as f:
            failed_docs = json.load(f)
        
        assert len(failed_docs) > 0, "Should have at least one failed document"
        assert any("corrupted.pdf" in doc["source_path"] for doc in failed_docs), "Corrupted PDF should be logged"
        
        # Verify metrics show parse failures
        with open(os.path.join(output_dir, "metrics.json"), 'r') as f:
            metrics = json.load(f)
        
        assert metrics["parse_failures"] > 0, "Should have parse failures"
        assert metrics["documents_processed"] >= 1, "Should have processed valid documents"
        
        print(f"✓ Corrupted files handled correctly: {metrics['parse_failures']} failures logged")
        print(f"✓ Valid documents processed: {metrics['documents_processed']}")
        
        ingestion.spark.stop()
        
    finally:
        shutil.rmtree(test_dir)


def test_deduplication():
    """Test deduplication with duplicate documents."""
    print("\n=== Test: Deduplication ===")
    
    test_dir = tempfile.mkdtemp()
    input_dir = os.path.join(test_dir, "input")
    output_dir = os.path.join(test_dir, "output")
    os.makedirs(input_dir)
    
    try:
        # Create original document
        original_content = "This is the original document content.\nIt has multiple lines.\n"
        with open(os.path.join(input_dir, "original.txt"), 'w') as f:
            f.write(original_content)
        
        # Create 3 exact duplicates
        for i in range(3):
            with open(os.path.join(input_dir, f"duplicate_{i}.txt"), 'w') as f:
                f.write(original_content)
        
        # Create a different document
        with open(os.path.join(input_dir, "different.txt"), 'w') as f:
            f.write("This is a different document.\n")
        
        # Run ingestion
        ingestion = DocumentIngestion()
        ingestion.run(input_dir, output_dir)
        
        # Verify metrics
        with open(os.path.join(output_dir, "metrics.json"), 'r') as f:
            metrics = json.load(f)
        
        # Should process 2 unique documents (original + different)
        # 3 duplicates should be removed
        assert metrics["documents_processed"] == 2, f"Expected 2 unique documents, got {metrics['documents_processed']}"
        assert metrics["duplicates_removed"] == 3, f"Expected 3 duplicates removed, got {metrics['duplicates_removed']}"
        
        print(f"✓ Deduplication works: {metrics['duplicates_removed']} duplicates removed")
        print(f"✓ Unique documents processed: {metrics['documents_processed']}")
        
        ingestion.spark.stop()
        
    finally:
        shutil.rmtree(test_dir)


def test_parquet_schema_and_content():
    """Verify Parquet output schema and content."""
    print("\n=== Test: Parquet Schema and Content ===")
    
    test_dir = tempfile.mkdtemp()
    input_dir = os.path.join(test_dir, "input")
    output_dir = os.path.join(test_dir, "output")
    os.makedirs(input_dir)
    
    try:
        # Create test documents
        with open(os.path.join(input_dir, "test1.txt"), 'w') as f:
            f.write("Test document 1 content")
        
        with open(os.path.join(input_dir, "test2.txt"), 'w') as f:
            f.write("Test document 2 content")
        
        # Run ingestion
        ingestion = DocumentIngestion()
        ingestion.run(input_dir, output_dir)
        
        # Read Parquet output
        df = ingestion.spark.read.parquet(output_dir)
        
        # Verify schema
        expected_columns = [
            "doc_id", "source_path", "text", "hash", "file_type",
            "file_size_bytes", "page_count", "ingestion_timestamp", "parser_version"
        ]
        
        actual_columns = df.columns
        for col in expected_columns:
            assert col in actual_columns, f"Column '{col}' missing from schema"
        
        print(f"✓ Schema verified: all {len(expected_columns)} expected columns present")
        
        # Verify content
        rows = df.collect()
        assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"
        
        for row in rows:
            # Verify doc_id is UUID format (36 chars with hyphens)
            assert len(row.doc_id) == 36, "doc_id should be UUID format"
            assert row.doc_id.count('-') == 4, "doc_id should have 4 hyphens"
            
            # Verify hash is SHA256 (64 hex chars)
            assert len(row.hash) == 64, "hash should be SHA256 (64 chars)"
            
            # Verify text content exists
            assert row.text is not None, "text should not be None"
            assert len(row.text) > 0, "text should not be empty"
            
            # Verify file_type
            assert row.file_type == "txt", "file_type should be 'txt'"
            
            # Verify metadata
            assert row.file_size_bytes > 0, "file_size_bytes should be positive"
            assert row.page_count == 1, "page_count should be 1 for text files"
            assert row.parser_version is not None, "parser_version should be set"
            assert row.ingestion_timestamp is not None, "ingestion_timestamp should be set"
        
        print("✓ Content verified: doc_id, hash, text, and metadata are correct")
        
        ingestion.spark.stop()
        
    finally:
        shutil.rmtree(test_dir)


def test_full_corpus_ingestion():
    """Test full ingestion pipeline with 100-document corpus."""
    print("\n=== Test: Full Corpus Ingestion (100 documents) ===")
    
    test_dir = tempfile.mkdtemp()
    input_dir = os.path.join(test_dir, "input")
    output_dir = os.path.join(test_dir, "output")
    os.makedirs(input_dir)
    
    try:
        # Create test corpus
        corpus_stats = create_test_corpus(input_dir)
        
        # Run ingestion
        ingestion = DocumentIngestion()
        ingestion.run(input_dir, output_dir)
        
        # Verify output files exist
        assert os.path.exists(output_dir), "Output directory should exist"
        assert os.path.exists(os.path.join(output_dir, "metrics.json")), "metrics.json should exist"
        
        # Verify metrics
        with open(os.path.join(output_dir, "metrics.json"), 'r') as f:
            metrics = json.load(f)
        
        print(f"\n📊 Ingestion Metrics:")
        print(f"  Documents processed: {metrics['documents_processed']}")
        print(f"  Parse failures: {metrics['parse_failures']}")
        print(f"  Duplicates removed: {metrics['duplicates_removed']}")
        print(f"  Processing time: {metrics['processing_time_seconds']:.2f} seconds")
        print(f"  Throughput: {metrics.get('documents_per_minute', 0):.2f} docs/min")
        
        # Verify expected results
        # Total files: 52 TXT + 40 PDF + 3 corrupted = 95 files
        # Duplicates: 5 (should be removed)
        # Corrupted: 3 (should fail)
        # Expected processed: 95 - 5 duplicates - 3 corrupted = 87
        # Note: PDFs are blank so may not have text, but should still be processed
        
        assert metrics["parse_failures"] == corpus_stats["corrupted_files"], \
            f"Expected {corpus_stats['corrupted_files']} parse failures"
        
        assert metrics["duplicates_removed"] == corpus_stats["duplicate_files"], \
            f"Expected {corpus_stats['duplicate_files']} duplicates removed"
        
        # Verify Parquet output
        df = ingestion.spark.read.parquet(output_dir)
        row_count = df.count()
        
        print(f"\n✓ Parquet output contains {row_count} documents")
        assert row_count == metrics["documents_processed"], "Parquet row count should match metrics"
        
        # Verify failed docs log
        if metrics["parse_failures"] > 0:
            failed_docs_path = os.path.join(output_dir, "failed_docs.json")
            assert os.path.exists(failed_docs_path), "failed_docs.json should exist"
            
            with open(failed_docs_path, 'r') as f:
                failed_docs = json.load(f)
            
            assert len(failed_docs) == metrics["parse_failures"], \
                "failed_docs.json should contain all parse failures"
            print(f"✓ Failed documents logged: {len(failed_docs)} entries")
        
        print("\n✅ Full corpus ingestion test passed!")
        
        ingestion.spark.stop()
        
    finally:
        shutil.rmtree(test_dir)


def test_text_normalization():
    """Test text normalization functionality."""
    print("\n=== Test: Text Normalization ===")
    
    ingestion = DocumentIngestion()
    
    # Test multiple spaces
    text = "Multiple   spaces   between   words"
    normalized = ingestion.normalize_text(text)
    assert "Multiple spaces between words" == normalized
    print("✓ Multiple spaces normalized")
    
    # Test multiple newlines
    text = "Line 1\n\n\n\nLine 2"
    normalized = ingestion.normalize_text(text)
    assert "Line 1\nLine 2" == normalized
    print("✓ Multiple newlines normalized")
    
    # Test mixed whitespace
    text = "  Leading and trailing spaces  \n\n  Another line  "
    normalized = ingestion.normalize_text(text)
    assert "Leading and trailing spaces\nAnother line" == normalized
    print("✓ Mixed whitespace normalized")
    
    # Test empty string
    text = ""
    normalized = ingestion.normalize_text(text)
    assert "" == normalized
    print("✓ Empty string handled")
    
    # Test None
    text = None
    normalized = ingestion.normalize_text(text)
    assert "" == normalized
    print("✓ None handled")
    
    ingestion.spark.stop()


def test_hash_consistency():
    """Test hash generation consistency."""
    print("\n=== Test: Hash Consistency ===")
    
    ingestion = DocumentIngestion()
    
    # Same content should produce same hash
    content = "Test document content"
    hash1 = ingestion.compute_hash(content)
    hash2 = ingestion.compute_hash(content)
    assert hash1 == hash2
    print("✓ Same content produces same hash")
    
    # Different content should produce different hash
    hash3 = ingestion.compute_hash("Different content")
    assert hash1 != hash3
    print("✓ Different content produces different hash")
    
    # Hash should be SHA256 (64 hex characters)
    assert len(hash1) == 64
    assert all(c in '0123456789abcdef' for c in hash1)
    print("✓ Hash is valid SHA256 format")
    
    ingestion.spark.stop()


def test_document_id_uniqueness():
    """Test document ID generation uniqueness."""
    print("\n=== Test: Document ID Uniqueness ===")
    
    ingestion = DocumentIngestion()
    
    # Generate multiple IDs
    ids = [ingestion.generate_doc_id() for _ in range(100)]
    
    # All should be unique
    assert len(ids) == len(set(ids))
    print("✓ 100 generated IDs are all unique")
    
    # All should be valid UUID format
    for doc_id in ids:
        assert len(doc_id) == 36
        assert doc_id.count('-') == 4
    print("✓ All IDs are valid UUID format")
    
    ingestion.spark.stop()


def run_all_tests():
    """Run all integration tests."""
    print("=" * 70)
    print("INGESTION PIPELINE INTEGRATION TESTS")
    print("=" * 70)
    
    tests = [
        ("Text Normalization", test_text_normalization),
        ("Hash Consistency", test_hash_consistency),
        ("Document ID Uniqueness", test_document_id_uniqueness),
        ("Parsing with Corrupted Files", test_parsing_with_corrupted_files),
        ("Deduplication", test_deduplication),
        ("Parquet Schema and Content", test_parquet_schema_and_content),
        ("Full Corpus Ingestion", test_full_corpus_ingestion),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ Test '{test_name}' FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 70)
    
    if failed == 0:
        print("\n✅ ALL INTEGRATION TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
