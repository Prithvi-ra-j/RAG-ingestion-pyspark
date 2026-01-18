# Enterprise RAG Pipeline Runner for PowerShell
# This script runs the complete RAG pipeline using Docker

param(
    [string]$Stage = "all",
    [string]$OpenRouterKey = $env:OPENROUTER_API_KEY
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Step {
    param([string]$Message)
    Write-Host "`n=== $Message ===" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

# Check prerequisites
Write-Step "Checking Prerequisites"

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error-Custom "Docker is not installed or not in PATH"
    exit 1
}
Write-Success "Docker found"

if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
    Write-Error-Custom "docker-compose is not installed or not in PATH"
    exit 1
}
Write-Success "docker-compose found"

if (-not (Test-Path "docker/docker-compose.yml")) {
    Write-Error-Custom "docker-compose.yml not found in docker/ directory"
    exit 1
}
Write-Success "docker-compose.yml found"

# Create data directories
Write-Step "Creating Data Directories"
$directories = @("data/raw", "data/processed", "data/embeddings", "data/vector_store")
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Success "Created $dir"
    } else {
        Write-Success "$dir exists"
    }
}

# Check for documents
$docCount = (Get-ChildItem -Path "data/raw" -File -ErrorAction SilentlyContinue).Count
if ($docCount -eq 0) {
    Write-Host "`nWARNING: No documents found in data/raw/" -ForegroundColor Yellow
    Write-Host "Please add PDF or TXT files to data/raw/ before running the pipeline" -ForegroundColor Yellow
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") {
        exit 0
    }
} else {
    Write-Success "Found $docCount files in data/raw/"
}

# Set environment variable
if ($OpenRouterKey) {
    $env:OPENROUTER_API_KEY = $OpenRouterKey
    Write-Success "OpenRouter API key set"
} else {
    Write-Host "`nWARNING: OPENROUTER_API_KEY not set" -ForegroundColor Yellow
    Write-Host "LLM generation will not work without it" -ForegroundColor Yellow
}

# Run pipeline stages
$composeFile = "docker/docker-compose.yml"

if ($Stage -eq "all" -or $Stage -eq "ingest") {
    Write-Step "Stage 1/5: Document Ingestion"
    docker-compose -f $composeFile run --rm spark-ingestion python spark/ingest.py --input /data/raw --output /data/processed/docs
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Ingestion failed"
        exit 1
    }
    Write-Success "Ingestion complete"
    
    if (Test-Path "data/processed/docs/metrics.json") {
        $metrics = Get-Content "data/processed/docs/metrics.json" | ConvertFrom-Json
        Write-Host "  Documents processed: $($metrics.documents_processed)" -ForegroundColor Gray
        Write-Host "  Parse failures: $($metrics.parse_failures)" -ForegroundColor Gray
    }
}

if ($Stage -eq "all" -or $Stage -eq "chunk") {
    Write-Step "Stage 2/5: Document Chunking"
    docker-compose -f $composeFile run --rm spark-chunking python spark/chunk.py --input /data/processed/docs --output /data/processed/chunks --chunk-size 512 --overlap 50 --strategy fixed
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Chunking failed"
        exit 1
    }
    Write-Success "Chunking complete"
    
    if (Test-Path "data/processed/chunks/metrics.json") {
        $metrics = Get-Content "data/processed/chunks/metrics.json" | ConvertFrom-Json
        Write-Host "  Total chunks: $($metrics.total_chunks)" -ForegroundColor Gray
        Write-Host "  Average chunk size: $($metrics.average_chunk_size) tokens" -ForegroundColor Gray
    }
}

if ($Stage -eq "all" -or $Stage -eq "embed") {
    Write-Step "Stage 3/5: Embedding Generation"
    Write-Host "This may take 30-60 minutes for 10k documents..." -ForegroundColor Yellow
    docker-compose -f $composeFile run --rm embedding-worker python spark/embed.py --input /data/processed/chunks --output /data/embeddings --model all-MiniLM-L6-v2 --batch-size 16 --checkpoint-interval 5000
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Embedding failed"
        exit 1
    }
    Write-Success "Embedding complete"
    
    if (Test-Path "data/embeddings/metrics.json") {
        $metrics = Get-Content "data/embeddings/metrics.json" | ConvertFrom-Json
        Write-Host "  Total embeddings: $($metrics.total_embeddings)" -ForegroundColor Gray
        Write-Host "  Throughput: $($metrics.chunks_per_second) chunks/sec" -ForegroundColor Gray
    }
}

if ($Stage -eq "all" -or $Stage -eq "vector") {
    Write-Step "Stage 4/5: Vector Store Building"
    python rag/vector_store.py --embeddings data/embeddings --chunks data/processed/chunks --output data/vector_store --index-type Flat
    if ($LASTEXITCODE -ne 0) {
        Write-Error-Custom "Vector store building failed"
        exit 1
    }
    Write-Success "Vector store built"
    
    if (Test-Path "data/vector_store/metrics.json") {
        $metrics = Get-Content "data/vector_store/metrics.json" | ConvertFrom-Json
        Write-Host "  Total vectors: $($metrics.total_vectors)" -ForegroundColor Gray
        Write-Host "  Index type: $($metrics.index_type)" -ForegroundColor Gray
    }
}

if ($Stage -eq "all" -or $Stage -eq "api") {
    Write-Step "Stage 5/5: Starting API"
    Write-Host "API will start on http://localhost:8000" -ForegroundColor Yellow
    Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
    Write-Host ""
    docker-compose -f $composeFile up rag-api
}

if ($Stage -eq "all") {
    Write-Step "Pipeline Complete!"
    Write-Success "All stages completed successfully"
    Write-Host "`nNext steps:" -ForegroundColor Cyan
    Write-Host "  1. Test API: Invoke-WebRequest http://localhost:8000/health"
    Write-Host "  2. Query: See USAGE_EXAMPLES.md for examples"
    Write-Host "  3. Metrics: python experiments/performance_metrics.py"
}
