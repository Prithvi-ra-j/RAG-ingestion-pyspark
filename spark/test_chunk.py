"""
Integration tests for chunking engine.

Tests cover:
- Fixed-size chunking with various sizes and overlaps (Requirement 2.1)
- Chunk metadata correctness (Requirement 2.2)
- Metadata join with document data (Requirement 2.3)
- Token counting (Requirement 2.4)
- Chunking strategy field (Requirement 2.5)
"""

import os
import sys
import tempfile
import shutil
import json
from chunk import ChunkingEngine, Chunk
from ingest import DocumentIngestion
from pyspark.sql import SparkSession


def create_test_documents(input_dir: str, output_dir: str):
    """Create test documents for chunking tests."""
    print("Creating test documents...")
    
    # Create input directory
    os.makedirs(input_dir, exist_ok=True)
    
    # Create sample documents
    docs = [
        ("doc1.txt", "This is a short document. " * 10),  # ~250 chars
        ("doc2.txt", "This is a longer document with more content. " * 20),  # ~900 chars
        ("doc3.txt", "A" * 1000),  # Exactly 1000 chars
        ("doc4.txt", "Short."),  # Very short
        ("doc5.txt", "This document has multiple sentences. Each sentence adds content. " * 15),  # ~960 chars
    ]
    
    for filename, content in docs:
        with open(os.path.join(input_dir, filename), 'w') as f:
            f.write(content)
    
    # Run ingestion to create processed documents
    ingestion = DocumentIngestion()
    ingestion.run(input_dir, output_dir)
    ingestion.spark.stop()
    
    print(f"✓ Created {len(docs)} test documents")


def test_fixed_size_chunking_basic():
    """Test basic fixed-size chunking functionality."""
    print("\n=== Test: Fixed-Size Chunking (Basic) ===")
    
    engine = ChunkingEngine()
    
    # Test with simple text
    text = "A" * 100
    doc_id = "test-doc-1"
    chunks = engine.chunk_fixed_size(text, doc_id, chunk_size=30, overlap=5)
    
    # Verify chunk count
    # 100 chars, 30 char chunks with 5 char overlap
    # Chunk 1: 0-30, Chunk 2: 25-55, Chunk 3: 50-80, Chunk 4: 75-100
    assert len(chunks) >= 3, f"Expected at least 3 chunks, got {len(chunks)}"
    print(f"✓ Generated {len(chunks)} chunks from 100-char text")
    
    # Verify chunk structure
    for i, chunk in enumerate(chunks):
        assert "chunk_id" in chunk
        assert "doc_id" in chunk
        assert "chunk_text" in chunk
        assert "chunk_index" in chunk
        assert "char_start" in chunk
        assert "char_end" in chunk
        
        assert chunk["doc_id"] == doc_id
        assert chunk["chunk_index"] == i
        assert len(chunk["chunk_text"]) <= 30
    
    print("✓ Chunk structure verified")
    
    # Verify overlap
    if len(chunks) > 1:
        # Check that consecutive chunks overlap
        chunk1_end = chunks[0]["char_end"]
        chunk2_start = chunks[1]["char_start"]
        overlap_size = chunk1_end - chunk2_start
        assert overlap_size == 5, f"Expected 5 char overlap, got {overlap_size}"
        print(f"✓ Overlap verified: {overlap_size} chars")
    
    engine.spark.stop()


def test_fixed_size_chunking_various_sizes():
    """Test fixed-size chunking with various chunk sizes and overlaps."""
    print("\n=== Test: Fixed-Size Chunking (Various Sizes) ===")
    
    engine = ChunkingEngine()
    
    text = "This is a test document. " * 50  # ~1250 chars
    doc_id = "test-doc-2"
    
    test_cases = [
        (512, 50),   # Default
        (256, 25),   # Smaller chunks
        (1024, 100), # Larger chunks
        (100, 10),   # Small chunks
    ]
    
    for chunk_size, overlap in test_cases:
        chunks = engine.chunk_fixed_size(text, doc_id, chunk_size, overlap)
        
        # Verify all chunks are within size limit
        for chunk in chunks:
            assert len(chunk["chunk_text"]) <= chunk_size, \
                f"Chunk exceeds size limit: {len(chunk['chunk_text'])} > {chunk_size}"
        
        print(f"✓ Chunk size {chunk_size}, overlap {overlap}: {len(chunks)} chunks generated")
    
    engine.spark.stop()


def test_chunking_edge_cases():
    """Test chunking with edge cases."""
    print("\n=== Test: Chunking Edge Cases ===")
    
    engine = ChunkingEngine()
    
    # Empty text
    chunks = engine.chunk_fixed_size("", "doc-1", 512, 50)
    assert len(chunks) == 0, "Empty text should produce no chunks"
    print("✓ Empty text handled")
    
    # Text shorter than chunk size
    text = "Short text"
    chunks = engine.chunk_fixed_size(text, "doc-2", 512, 50)
    assert len(chunks) == 1, "Short text should produce 1 chunk"
    assert chunks[0]["chunk_text"] == text
    print("✓ Text shorter than chunk size handled")
    
    # Text with only whitespace
    text = "   \n\n\t  "
    chunks = engine.chunk_fixed_size(text, "doc-3", 512, 50)
    # Should produce no chunks since whitespace is stripped
    assert len(chunks) == 0, "Whitespace-only text should produce no chunks"
    print("✓ Whitespace-only text handled")
    
    # Very large overlap (larger than chunk size)
    text = "A" * 100
    chunks = engine.chunk_fixed_size(text, "doc-4", 30, 40)
    # Should still work, but overlap will be capped
    assert len(chunks) > 0, "Should handle large overlap"
    print("✓ Large overlap handled")
    
    engine.spark.stop()


def test_chunk_metadata_correctness():
    """Test chunk metadata indices and offsets."""
    print("\n=== Test: Chunk Metadata Correctness ===")
    
    engine = ChunkingEngine()
    
    text = "0123456789" * 10  # 100 chars, easy to verify offsets
    doc_id = "test-doc-meta"
    chunks = engine.chunk_fixed_size(text, doc_id, chunk_size=30, overlap=5)
    
    for i, chunk in enumerate(chunks):
        # Verify chunk_index is sequential
        assert chunk["chunk_index"] == i, f"Chunk index mismatch: expected {i}, got {chunk['chunk_index']}"
        
        # Verify char_start and char_end match chunk_text
        expected_text = text[chunk["char_start"]:chunk["char_end"]]
        assert chunk["chunk_text"] == expected_text, \
            f"Chunk text doesn't match offsets: '{chunk['chunk_text']}' != '{expected_text}'"
        
        # Verify char_end > char_start
        assert chunk["char_end"] > chunk["char_start"], "char_end should be greater than char_start"
    
    print(f"✓ Metadata verified for {len(chunks)} chunks")
    
    engine.spark.stop()


def test_full_chunking_pipeline():
    """Test full chunking pipeline with document ingestion."""
    print("\n=== Test: Full Chunking Pipeline ===")
    
    test_dir = tempfile.mkdtemp()
    input_dir = os.path.join(test_dir, "input")
    docs_dir = os.path.join(test_dir, "docs")
    chunks_dir = os.path.join(test_dir, "chunks")
    
    try:
        # Create test documents
        create_test_documents(input_dir, docs_dir)
        
        # Run chunking
        engine = ChunkingEngine()
        engine.run(
            input_path=docs_dir,
            output_path=chunks_dir,
            chunk_size=512,
            overlap=50,
            strategy="fixed"
        )
        
        # Verify output files exist
        assert os.path.exists(chunks_dir), "Chunks directory should exist"
        assert os.path.exists(os.path.join(chunks_dir, "metrics.json")), "metrics.json should exist"
        
        # Verify metrics
        with open(os.path.join(chunks_dir, "metrics.json"), 'r') as f:
            metrics = json.load(f)
        
        print(f"\n📊 Chunking Metrics:")
        print(f"  Total chunks: {metrics['total_chunks']}")
        print(f"  Total documents: {metrics['total_documents']}")
        print(f"  Average chunk size: {metrics['average_chunk_size']} tokens")
        print(f"  Chunks per second: {metrics['chunks_per_second']}")
        print(f"  Processing time: {metrics['processing_time_seconds']:.2f} seconds")
        
        # Verify chunks were generated
        assert metrics["total_chunks"] > 0, "Should have generated chunks"
        assert metrics["total_documents"] > 0, "Should have processed documents"
        assert metrics["average_chunk_size"] > 0, "Average chunk size should be positive"
        
        # Verify Parquet output
        chunks_df = engine.spark.read.parquet(chunks_dir)
        row_count = chunks_df.count()
        
        print(f"\n✓ Parquet output contains {row_count} chunks")
        assert row_count == metrics["total_chunks"], "Parquet row count should match metrics"
        
        # Verify schema
        expected_columns = [
            "chunk_id", "doc_id", "chunk_text", "chunk_index",
            "char_start", "char_end", "token_count", "chunking_strategy",
            "source_path", "file_type", "ingestion_timestamp"
        ]
        
        actual_columns = chunks_df.columns
        for col in expected_columns:
            assert col in actual_columns, f"Column '{col}' missing from schema"
        
        print(f"✓ Schema verified: all {len(expected_columns)} expected columns present")
        
        # Verify chunk content
        sample_chunks = chunks_df.limit(5).collect()
        for chunk in sample_chunks:
            # Verify chunk_id is UUID format
            assert len(chunk.chunk_id) == 36, "chunk_id should be UUID format"
            
            # Verify doc_id is UUID format
            assert len(chunk.doc_id) == 36, "doc_id should be UUID format"
            
            # Verify chunk_text exists
            assert chunk.chunk_text is not None, "chunk_text should not be None"
            assert len(chunk.chunk_text) > 0, "chunk_text should not be empty"
            
            # Verify token_count is positive
            assert chunk.token_count > 0, "token_count should be positive"
            
            # Verify chunking_strategy
            assert chunk.chunking_strategy == "fixed", "chunking_strategy should be 'fixed'"
            
            # Verify metadata from document join
            assert chunk.source_path is not None, "source_path should be present"
            assert chunk.file_type is not None, "file_type should be present"
        
        print("✓ Chunk content verified")
        
        print("\n✅ Full chunking pipeline test passed!")
        
        engine.spark.stop()
        
    finally:
        shutil.rmtree(test_dir)


def test_token_counting():
    """Test token count approximation."""
    print("\n=== Test: Token Counting ===")
    
    engine = ChunkingEngine()
    
    # Test simple text
    text = "This is a test"  # 4 words
    token_count = engine.calculate_token_count(text)
    # ~1.3 tokens per word = 4 * 1.3 = 5.2 ≈ 5 tokens
    assert token_count >= 4 and token_count <= 6, f"Expected ~5 tokens, got {token_count}"
    print(f"✓ Token count for 4 words: {token_count} tokens")
    
    # Test empty text
    token_count = engine.calculate_token_count("")
    assert token_count == 0, "Empty text should have 0 tokens"
    print("✓ Empty text: 0 tokens")
    
    # Test None
    token_count = engine.calculate_token_count(None)
    assert token_count == 0, "None should have 0 tokens"
    print("✓ None: 0 tokens")
    
    # Test longer text
    text = "This is a longer text with many words to test the token counting functionality"  # 15 words
    token_count = engine.calculate_token_count(text)
    # ~1.3 tokens per word = 15 * 1.3 = 19.5 ≈ 19 tokens
    assert token_count >= 15 and token_count <= 25, f"Expected ~19 tokens, got {token_count}"
    print(f"✓ Token count for 15 words: {token_count} tokens")
    
    engine.spark.stop()


def test_metadata_join():
    """Test metadata join with document data."""
    print("\n=== Test: Metadata Join ===")
    
    test_dir = tempfile.mkdtemp()
    input_dir = os.path.join(test_dir, "input")
    docs_dir = os.path.join(test_dir, "docs")
    chunks_dir = os.path.join(test_dir, "chunks")
    
    try:
        # Create test documents
        create_test_documents(input_dir, docs_dir)
        
        # Run chunking
        engine = ChunkingEngine()
        engine.run(
            input_path=docs_dir,
            output_path=chunks_dir,
            chunk_size=256,
            overlap=25,
            strategy="fixed"
        )
        
        # Read chunks
        chunks_df = engine.spark.read.parquet(chunks_dir)
        
        # Verify all chunks have document metadata
        chunks_with_metadata = chunks_df.filter(
            (chunks_df.source_path.isNotNull()) &
            (chunks_df.file_type.isNotNull()) &
            (chunks_df.ingestion_timestamp.isNotNull())
        ).count()
        
        total_chunks = chunks_df.count()
        
        assert chunks_with_metadata == total_chunks, \
            f"All chunks should have metadata: {chunks_with_metadata}/{total_chunks}"
        
        print(f"✓ All {total_chunks} chunks have document metadata")
        
        # Verify metadata values are correct
        sample = chunks_df.limit(1).collect()[0]
        assert sample.file_type == "txt", "file_type should be 'txt'"
        assert "doc" in sample.source_path.lower(), "source_path should contain 'doc'"
        
        print("✓ Metadata values verified")
        
        engine.spark.stop()
        
    finally:
        shutil.rmtree(test_dir)


def test_chunking_strategy_field():
    """Test chunking strategy field in output."""
    print("\n=== Test: Chunking Strategy Field ===")
    
    test_dir = tempfile.mkdtemp()
    input_dir = os.path.join(test_dir, "input")
    docs_dir = os.path.join(test_dir, "docs")
    chunks_dir = os.path.join(test_dir, "chunks")
    
    try:
        # Create test documents
        create_test_documents(input_dir, docs_dir)
        
        # Run chunking with fixed strategy
        engine = ChunkingEngine()
        engine.run(
            input_path=docs_dir,
            output_path=chunks_dir,
            chunk_size=512,
            overlap=50,
            strategy="fixed"
        )
        
        # Read chunks
        chunks_df = engine.spark.read.parquet(chunks_dir)
        
        # Verify all chunks have strategy field
        strategies = chunks_df.select("chunking_strategy").distinct().collect()
        assert len(strategies) == 1, "Should have exactly one strategy"
        assert strategies[0].chunking_strategy == "fixed", "Strategy should be 'fixed'"
        
        print("✓ Chunking strategy field verified: 'fixed'")
        
        # Verify partitioning by strategy
        # Check if partition directory exists
        partition_dir = os.path.join(chunks_dir, "chunking_strategy=fixed")
        assert os.path.exists(partition_dir), "Should be partitioned by chunking_strategy"
        
        print("✓ Parquet partitioned by chunking_strategy")
        
        engine.spark.stop()
        
    finally:
        shutil.rmtree(test_dir)


def run_all_tests():
    """Run all chunking integration tests."""
    print("=" * 70)
    print("CHUNKING ENGINE INTEGRATION TESTS")
    print("=" * 70)
    
    tests = [
        ("Fixed-Size Chunking (Basic)", test_fixed_size_chunking_basic),
        ("Fixed-Size Chunking (Various Sizes)", test_fixed_size_chunking_various_sizes),
        ("Chunking Edge Cases", test_chunking_edge_cases),
        ("Chunk Metadata Correctness", test_chunk_metadata_correctness),
        ("Token Counting", test_token_counting),
        ("Metadata Join", test_metadata_join),
        ("Chunking Strategy Field", test_chunking_strategy_field),
        ("Full Chunking Pipeline", test_full_chunking_pipeline),
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
        print("\n✅ ALL CHUNKING TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
