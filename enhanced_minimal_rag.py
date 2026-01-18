"""
Enhanced Minimal RAG system that handles both TXT and PDF files
Uses only built-in Python libraries + PyPDF2 for PDF reading
"""

import json
import math
import re
from pathlib import Path
from typing import List, Dict
import http.server
import socketserver
import urllib.parse
import threading
import time

# Try to import PyPDF2 for PDF support
try:
    import PyPDF2
    PDF_SUPPORT = True
    print("✅ PDF support available")
except ImportError:
    PDF_SUPPORT = False
    print("⚠️  PDF support not available. Install PyPDF2 for PDF reading: pip install PyPDF2")

class SimpleEmbedding:
    """Simple TF-IDF based embedding (no external dependencies)"""
    
    def __init__(self):
        self.vocabulary = {}
        self.idf_scores = {}
        self.documents = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return re.findall(r'\b\w+\b', text.lower())
    
    def fit(self, documents: List[str]):
        """Build vocabulary and IDF scores"""
        self.documents = documents
        doc_count = len(documents)
        
        # Build vocabulary
        all_tokens = set()
        for doc in documents:
            tokens = self._tokenize(doc)
            all_tokens.update(tokens)
        
        self.vocabulary = {token: i for i, token in enumerate(sorted(all_tokens))}
        
        # Calculate IDF scores
        for token in self.vocabulary:
            doc_freq = sum(1 for doc in documents if token in self._tokenize(doc))
            self.idf_scores[token] = math.log(doc_count / (doc_freq + 1))
    
    def transform(self, text: str) -> List[float]:
        """Convert text to TF-IDF vector"""
        tokens = self._tokenize(text)
        token_counts = {}
        
        # Count tokens
        for token in tokens:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        # Create TF-IDF vector
        vector = [0.0] * len(self.vocabulary)
        total_tokens = len(tokens)
        
        for token, count in token_counts.items():
            if token in self.vocabulary:
                tf = count / total_tokens
                idf = self.idf_scores.get(token, 0)
                vector[self.vocabulary[token]] = tf * idf
        
        return vector
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(a * a for a in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

class DocumentLoader:
    """Load documents from various file formats"""
    
    @staticmethod
    def read_txt_file(file_path: Path) -> str:
        """Read text file"""
        try:
            return file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                return file_path.read_text(encoding='latin-1')
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                return ""
    
    @staticmethod
    def read_pdf_file(file_path: Path) -> str:
        """Read PDF file using PyPDF2"""
        if not PDF_SUPPORT:
            return f"[PDF file: {file_path.name} - PDF support not available]"
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                return text.strip()
        except Exception as e:
            print(f"Error reading PDF {file_path}: {e}")
            return f"[PDF file: {file_path.name} - Error reading: {str(e)}]"
    
    @staticmethod
    def load_documents_from_directory(data_dir: str = "data/raw") -> List[Dict]:
        """Load all supported documents from directory"""
        data_path = Path(data_dir)
        documents = []
        
        if not data_path.exists():
            print(f"Directory {data_dir} not found")
            return documents
        
        # Supported file extensions
        supported_extensions = {'.txt', '.pdf'}
        
        for file_path in data_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                print(f"Loading: {file_path.name}")
                
                if file_path.suffix.lower() == '.txt':
                    content = DocumentLoader.read_txt_file(file_path)
                elif file_path.suffix.lower() == '.pdf':
                    content = DocumentLoader.read_pdf_file(file_path)
                else:
                    continue
                
                if content.strip():
                    # Split long documents into chunks for better retrieval
                    chunks = DocumentLoader.chunk_document(content, file_path.name)
                    documents.extend(chunks)
        
        return documents
    
    @staticmethod
    def chunk_document(content: str, filename: str, chunk_size: int = 1000, overlap: int = 100) -> List[Dict]:
        """Split document into overlapping chunks"""
        if len(content) <= chunk_size:
            return [{
                'content': content,
                'source': filename,
                'chunk_id': 0,
                'total_chunks': 1
            }]
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(content):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(content):
                # Look for sentence endings near the chunk boundary
                for i in range(end, max(start + chunk_size - 200, start), -1):
                    if content[i] in '.!?':
                        end = i + 1
                        break
            
            chunk_content = content[start:end].strip()
            
            if chunk_content:
                chunks.append({
                    'content': chunk_content,
                    'source': filename,
                    'chunk_id': chunk_id,
                    'total_chunks': 0  # Will be updated later
                })
                chunk_id += 1
            
            start = end - overlap
        
        # Update total chunks count
        for chunk in chunks:
            chunk['total_chunks'] = len(chunks)
        
        return chunks

class EnhancedMinimalRAG:
    """Enhanced Minimal RAG system with PDF support"""
    
    def __init__(self):
        self.embedder = SimpleEmbedding()
        self.documents = []
        self.doc_vectors = []
    
    def load_documents(self, data_dir: str = "data/raw"):
        """Load documents from directory"""
        print(f"Loading documents from {data_dir}...")
        
        # Load documents
        doc_chunks = DocumentLoader.load_documents_from_directory(data_dir)
        
        if not doc_chunks:
            print("No documents found!")
            return
        
        # Extract content for embedding
        self.documents = doc_chunks
        contents = [doc['content'] for doc in doc_chunks]
        
        print(f"Loaded {len(doc_chunks)} document chunks from {len(set(doc['source'] for doc in doc_chunks))} files")
        
        # Build embeddings
        if contents:
            self.embedder.fit(contents)
            self.doc_vectors = [self.embedder.transform(content) for content in contents]
            print("Built TF-IDF embeddings")
    
    def query(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Query the RAG system"""
        if not self.documents:
            return []
        
        # Get query vector
        query_vector = self.embedder.transform(query_text)
        
        # Calculate similarities
        similarities = []
        for i, doc_vector in enumerate(self.doc_vectors):
            similarity = self.embedder.cosine_similarity(query_vector, doc_vector)
            similarities.append((similarity, i))
        
        # Sort by similarity
        similarities.sort(reverse=True)
        
        # Return top results
        results = []
        for similarity, doc_idx in similarities[:top_k]:
            if similarity > 0:  # Only return relevant results
                doc = self.documents[doc_idx]
                results.append({
                    "content": doc['content'],
                    "source": doc['source'],
                    "chunk_id": doc['chunk_id'],
                    "total_chunks": doc['total_chunks'],
                    "similarity": similarity,
                    "rank": len(results) + 1
                })
        
        return results

class EnhancedHTTPHandler(http.server.BaseHTTPRequestHandler):
    """Enhanced HTTP handler for RAG API"""
    
    def __init__(self, *args, rag_system=None, **kwargs):
        self.rag_system = rag_system
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == "/health":
            self._send_json({
                "status": "healthy", 
                "documents": len(self.rag_system.documents),
                "sources": len(set(doc['source'] for doc in self.rag_system.documents)),
                "pdf_support": PDF_SUPPORT
            })
        elif self.path == "/":
            self._send_html()
        else:
            self._send_error(404, "Not found")
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == "/query":
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                query = data.get('query', '')
                top_k = data.get('top_k', 5)
                
                results = self.rag_system.query(query, top_k)
                
                response = {
                    "query": query,
                    "results": results,
                    "total_results": len(results)
                }
                
                self._send_json(response)
                
            except Exception as e:
                self._send_error(500, f"Query failed: {str(e)}")
        else:
            self._send_error(404, "Not found")
    
    def _send_json(self, data):
        """Send JSON response"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def _send_html(self):
        """Send enhanced HTML interface"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Enhanced Minimal RAG System</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .search-box { margin: 20px 0; }
                .search-box input { width: 400px; padding: 10px; font-size: 16px; }
                .search-box button { padding: 10px 20px; font-size: 16px; }
                .result { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }
                .result-header { font-weight: bold; color: #333; margin-bottom: 10px; }
                .result-content { line-height: 1.6; }
                .result-meta { color: #666; font-size: 12px; margin-top: 10px; }
                .stats { background: #f5f5f5; padding: 10px; border-radius: 5px; margin: 20px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🔍 Enhanced Minimal RAG System</h1>
                
                <div class="stats" id="stats">Loading system stats...</div>
                
                <div class="search-box">
                    <input type="text" id="query" placeholder="Ask about your documents..." onkeypress="if(event.key==='Enter') search()">
                    <button onclick="search()">Search</button>
                    <label>
                        Results: <select id="topk">
                            <option value="3">3</option>
                            <option value="5" selected>5</option>
                            <option value="10">10</option>
                        </select>
                    </label>
                </div>
                
                <div id="results"></div>
            </div>
            
            <script>
            // Load stats on page load
            fetch('/health')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('stats').innerHTML = `
                        📊 <strong>System Status:</strong> ${data.status} | 
                        📄 <strong>Document Chunks:</strong> ${data.documents} | 
                        📁 <strong>Source Files:</strong> ${data.sources} | 
                        📋 <strong>PDF Support:</strong> ${data.pdf_support ? '✅ Available' : '❌ Not Available'}
                    `;
                });
            
            function search() {
                const query = document.getElementById('query').value;
                const topk = document.getElementById('topk').value;
                
                if (!query.trim()) return;
                
                document.getElementById('results').innerHTML = '<p>Searching...</p>';
                
                fetch('/query', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({query: query, top_k: parseInt(topk)})
                })
                .then(response => response.json())
                .then(data => {
                    const resultsDiv = document.getElementById('results');
                    
                    if (data.results.length === 0) {
                        resultsDiv.innerHTML = '<p>No relevant results found.</p>';
                        return;
                    }
                    
                    resultsDiv.innerHTML = `<h3>Found ${data.total_results} results for: "${data.query}"</h3>`;
                    
                    data.results.forEach(result => {
                        const preview = result.content.length > 300 ? 
                            result.content.substring(0, 300) + '...' : 
                            result.content;
                            
                        resultsDiv.innerHTML += `
                            <div class="result">
                                <div class="result-header">
                                    Rank ${result.rank} - ${result.source} 
                                    (Chunk ${result.chunk_id + 1}/${result.total_chunks})
                                </div>
                                <div class="result-content">${preview}</div>
                                <div class="result-meta">
                                    Similarity: ${result.similarity.toFixed(3)}
                                </div>
                            </div>
                        `;
                    });
                })
                .catch(error => {
                    document.getElementById('results').innerHTML = `<p>Error: ${error}</p>`;
                });
            }
            </script>
        </body>
        </html>
        """
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def _send_error(self, code, message):
        """Send error response"""
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode())
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass

def main():
    """Run the enhanced minimal RAG system"""
    print("🚀 Starting Enhanced Minimal RAG System")
    print("=" * 60)
    
    # Initialize RAG system
    rag = EnhancedMinimalRAG()
    rag.load_documents()
    
    if not rag.documents:
        print("❌ No documents loaded!")
        return
    
    # Test query
    print("\n🧪 Testing with sample query...")
    results = rag.query("control system", top_k=3)
    for result in results:
        preview = result['content'][:80] + "..." if len(result['content']) > 80 else result['content']
        print(f"  • {result['source']}: {preview} (similarity: {result['similarity']:.3f})")
    
    # Start HTTP server
    port = 8001
    handler = lambda *args, **kwargs: EnhancedHTTPHandler(*args, rag_system=rag, **kwargs)
    
    try:
        with socketserver.TCPServer(("", port), handler) as httpd:
            print(f"\n🌐 Server running at http://localhost:{port}")
            print("📖 Open http://localhost:8000 in your browser")
            print("🔍 Try queries like: 'control system', 'safety procedures', 'DCS configuration'")
            print("\nPress Ctrl+C to stop")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")

if __name__ == "__main__":
    main()