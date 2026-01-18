"""
Performance Metrics Analysis

This script analyzes metrics from all pipeline stages and generates
a performance report.

Usage:
    python experiments/performance_metrics.py
"""

import json
import os
from typing import Dict
import pandas as pd


def load_metrics(metrics_path: str) -> Dict:
    """Load metrics from JSON file."""
    if not os.path.exists(metrics_path):
        return {}
    
    with open(metrics_path, 'r') as f:
        return json.load(f)


def analyze_ingestion_metrics(base_path: str = "data/processed/docs"):
    """Analyze ingestion pipeline metrics."""
    metrics = load_metrics(os.path.join(base_path, "metrics.json"))
    
    if not metrics:
        print("⚠️  Ingestion metrics not found")
        return None
    
    print("=" * 60)
    print("INGESTION METRICS")
    print("=" * 60)
    print(f"Documents processed: {metrics.get('documents_processed', 0)}")
    print(f"Parse failures: {metrics.get('parse_failures', 0)}")
    print(f"Duplicates removed: {metrics.get('duplicates_removed', 0)}")
    print(f"Processing time: {metrics.get('processing_time_seconds', 0):.2f}s")
    print(f"Throughput: {metrics.get('documents_per_minute', 0):.2f} docs/min")
    print()
    
    return metrics


def analyze_chunking_metrics(base_path: str = "data/processed/chunks"):
    """Analyze chunking pipeline metrics."""
    metrics = load_metrics(os.path.join(base_path, "metrics.json"))
    
    if not metrics:
        print("⚠️  Chunking metrics not found")
        return None
    
    print("=" * 60)
    print("CHUNKING METRICS")
    print("=" * 60)
    print(f"Total chunks: {metrics.get('total_chunks', 0)}")
    print(f"Total documents: {metrics.get('total_documents', 0)}")
    print(f"Average chunk size: {metrics.get('average_chunk_size', 0):.2f} tokens")
    print(f"Chunks per second: {metrics.get('chunks_per_second', 0):.2f}")
    print(f"Processing time: {metrics.get('processing_time_seconds', 0):.2f}s")
    
    if metrics.get('total_documents', 0) > 0:
        chunks_per_doc = metrics['total_chunks'] / metrics['total_documents']
        print(f"Chunks per document: {chunks_per_doc:.2f}")
    
    print()
    
    return metrics


def analyze_embedding_metrics(base_path: str = "data/embeddings"):
    """Analyze embedding pipeline metrics."""
    metrics = load_metrics(os.path.join(base_path, "metrics.json"))
    
    if not metrics:
        print("⚠️  Embedding metrics not found")
        return None
    
    print("=" * 60)
    print("EMBEDDING METRICS")
    print("=" * 60)
    print(f"Total embeddings: {metrics.get('total_embeddings', 0)}")
    print(f"Model: {metrics.get('model_name', 'Unknown')}")
    print(f"Device: {metrics.get('device', 'Unknown')}")
    print(f"Chunks per second: {metrics.get('chunks_per_second', 0):.2f}")
    print(f"Processing time: {metrics.get('processing_time_seconds', 0):.2f}s")
    
    # Calculate estimated time for different scales
    if metrics.get('chunks_per_second', 0) > 0:
        cps = metrics['chunks_per_second']
        print(f"\nEstimated processing time:")
        print(f"  10k chunks: {(10000 / cps / 60):.1f} minutes")
        print(f"  100k chunks: {(100000 / cps / 60):.1f} minutes")
        print(f"  1M chunks: {(1000000 / cps / 3600):.1f} hours")
    
    print()
    
    return metrics


def analyze_vector_store_metrics(base_path: str = "data/vector_store"):
    """Analyze vector store metrics."""
    metrics = load_metrics(os.path.join(base_path, "metrics.json"))
    
    if not metrics:
        print("⚠️  Vector store metrics not found")
        return None
    
    print("=" * 60)
    print("VECTOR STORE METRICS")
    print("=" * 60)
    print(f"Total vectors: {metrics.get('total_vectors', 0)}")
    print(f"Index type: {metrics.get('index_type', 'Unknown')}")
    print(f"GPU enabled: {metrics.get('use_gpu', False)}")
    print(f"Build time: {metrics.get('build_time_seconds', 0):.2f}s")
    print()
    
    return metrics


def generate_summary_report():
    """Generate comprehensive summary report."""
    print("\n")
    print("=" * 60)
    print("PIPELINE SUMMARY REPORT")
    print("=" * 60)
    print()
    
    # Load all metrics
    ingestion = analyze_ingestion_metrics()
    chunking = analyze_chunking_metrics()
    embedding = analyze_embedding_metrics()
    vector_store = analyze_vector_store_metrics()
    
    # Calculate end-to-end metrics
    print("=" * 60)
    print("END-TO-END METRICS")
    print("=" * 60)
    
    total_time = 0
    if ingestion:
        total_time += ingestion.get('processing_time_seconds', 0)
    if chunking:
        total_time += chunking.get('processing_time_seconds', 0)
    if embedding:
        total_time += embedding.get('processing_time_seconds', 0)
    if vector_store:
        total_time += vector_store.get('build_time_seconds', 0)
    
    print(f"Total pipeline time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    
    if ingestion and embedding:
        docs = ingestion.get('documents_processed', 0)
        embeddings = embedding.get('total_embeddings', 0)
        if docs > 0:
            print(f"Documents processed: {docs}")
            print(f"Embeddings generated: {embeddings}")
            print(f"Embeddings per document: {embeddings/docs:.2f}")
    
    print()
    
    # Performance assessment
    print("=" * 60)
    print("PERFORMANCE ASSESSMENT")
    print("=" * 60)
    
    targets = {
        "ingestion": {"target": 1000, "actual": ingestion.get('documents_per_minute', 0) if ingestion else 0, "unit": "docs/min"},
        "chunking": {"target": 5000, "actual": chunking.get('chunks_per_second', 0) if chunking else 0, "unit": "chunks/sec"},
        "embedding": {"target": 200, "actual": embedding.get('chunks_per_second', 0) if embedding else 0, "unit": "chunks/sec"},
    }
    
    for stage, data in targets.items():
        actual = data["actual"]
        target = data["target"]
        unit = data["unit"]
        
        if actual > 0:
            percentage = (actual / target) * 100
            status = "✅" if percentage >= 80 else "⚠️" if percentage >= 50 else "❌"
            print(f"{status} {stage.capitalize()}: {actual:.2f} {unit} ({percentage:.1f}% of target)")
        else:
            print(f"⚠️  {stage.capitalize()}: No data")
    
    print()


def export_metrics_csv():
    """Export all metrics to CSV for further analysis."""
    data = []
    
    # Load all metrics
    ingestion = load_metrics("data/processed/docs/metrics.json")
    chunking = load_metrics("data/processed/chunks/metrics.json")
    embedding = load_metrics("data/embeddings/metrics.json")
    vector_store = load_metrics("data/vector_store/metrics.json")
    
    # Combine into single row
    row = {
        "ingestion_docs": ingestion.get('documents_processed', 0),
        "ingestion_failures": ingestion.get('parse_failures', 0),
        "ingestion_duplicates": ingestion.get('duplicates_removed', 0),
        "ingestion_time_s": ingestion.get('processing_time_seconds', 0),
        "ingestion_throughput": ingestion.get('documents_per_minute', 0),
        
        "chunking_chunks": chunking.get('total_chunks', 0),
        "chunking_docs": chunking.get('total_documents', 0),
        "chunking_avg_size": chunking.get('average_chunk_size', 0),
        "chunking_time_s": chunking.get('processing_time_seconds', 0),
        "chunking_throughput": chunking.get('chunks_per_second', 0),
        
        "embedding_count": embedding.get('total_embeddings', 0),
        "embedding_model": embedding.get('model_name', ''),
        "embedding_device": embedding.get('device', ''),
        "embedding_time_s": embedding.get('processing_time_seconds', 0),
        "embedding_throughput": embedding.get('chunks_per_second', 0),
        
        "vector_store_vectors": vector_store.get('total_vectors', 0),
        "vector_store_type": vector_store.get('index_type', ''),
        "vector_store_build_time_s": vector_store.get('build_time_seconds', 0),
    }
    
    df = pd.DataFrame([row])
    output_path = "experiments/pipeline_metrics.csv"
    df.to_csv(output_path, index=False)
    
    print(f"📊 Metrics exported to {output_path}")
    print()


def main():
    """Main entry point."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "ENTERPRISE RAG PERFORMANCE ANALYSIS" + " " * 12 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    generate_summary_report()
    export_metrics_csv()
    
    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
