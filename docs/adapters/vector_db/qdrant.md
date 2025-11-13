# Qdrant Vector Database Adapter

> **High-performance vector database backend for Parlant** - Store, search, and retrieve document embeddings with enterprise-grade performance and scalability.

## 🚀 Overview

Qdrant is an open-source vector similarity search engine written in Rust, providing exceptional performance and scalability for semantic search applications. The Qdrant adapter seamlessly integrates with Parlant, enabling you to build powerful AI agents with efficient document storage and retrieval.

### Key Features

- ⚡ **High Performance**: Optimized Rust implementation for blazing-fast searches
- 📈 **Scalability**: Handles millions of vectors efficiently
- 🔄 **Flexibility**: In-memory, local, or cloud deployments
- 🏭 **Production Ready**: Battle-tested in production environments
- ☁️ **Cloud Support**: Native integration with Qdrant Cloud

---

## 📋 Prerequisites

Before getting started, ensure you have:

1. **Python 3.8+** installed
2. **Qdrant Client** library (`qdrant-client`)
3. **Storage Option**: Choose one:
   - Local file system (for development)
   - Qdrant Cloud account (for production)
   - Docker (for local server)

---

## 🔧 Installation

### Step 1: Install the Qdrant Client

```bash
pip install qdrant-client
```

### Step 2: Choose Your Deployment Mode

#### Option A: Local Development (Docker)

```bash
# Pull and run Qdrant locally
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
```

#### Option B: Qdrant Cloud (Recommended for Production)

1. Sign up at [Qdrant Cloud](https://cloud.qdrant.io/)
2. Create a cluster
3. Get your API key and endpoint URL

---

## ⚙️ Configuration

The Qdrant adapter supports three deployment modes:

### 1. In-Memory Mode (Testing)

Perfect for unit tests and development. Data is stored in memory and lost when the process ends.

```python
from parlant.adapters.vector_db.qdrant import QdrantDatabase
from parlant.core.nlp.embedding import EmbedderFactory, NullEmbeddingCache
from parlant.core.loggers import Logger

async with QdrantDatabase(
    logger=logger,
    path=None,  # None triggers in-memory mode
    embedder_factory=embedder_factory,
    embedding_cache_provider=NullEmbeddingCache,
) as qdrant_db:
    # Your code here
    pass
```

### 2. Local File System (Persistent Storage)

Store data on your local file system for persistence across sessions.

```python
from pathlib import Path

async with QdrantDatabase(
    logger=logger,
    path=Path("./qdrant_data"),  # Local directory path
    embedder_factory=embedder_factory,
    embedding_cache_provider=NullEmbeddingCache,
) as qdrant_db:
    # Your code here
    pass
```

### 3. Qdrant Cloud (Production)

Connect to Qdrant Cloud for production deployments with managed infrastructure.

```python
async with QdrantDatabase(
    logger=logger,
    url="https://your-cluster-id.us-east4-0.gcp.cloud.qdrant.io",
    api_key="your-api-key-here",
    embedder_factory=embedder_factory,
    embedding_cache_provider=NullEmbeddingCache,
) as qdrant_db:
    # Your code here
    pass
```

---

## 📚 Adding Documents

### Basic Text Documents

The simplest way to add documents is with plain text content:

```python
from parlant.adapters.vector_db.qdrant import QdrantDatabase
from parlant.core.persistence.vector_database import BaseDocument, identity_loader
from parlant.core.nlp.embedding import OpenAITextEmbedding3Large
from parlant.core.common import md5_checksum, ObjectId
from pathlib import Path

async def add_text_documents():
    async with QdrantDatabase(
        logger=logger,
        path=Path("./qdrant_data"),
        embedder_factory=embedder_factory,
        embedding_cache_provider=embedding_cache_provider,
    ) as qdrant_db:
        # Get or create collection
        collection = await qdrant_db.get_or_create_collection(
            name="documents",
            schema=BaseDocument,
            embedder_type=OpenAITextEmbedding3Large,
            document_loader=identity_loader,
        )
        
        # Add a simple text document
        document = {
            "id": ObjectId("doc_001"),
            "content": "Python is a high-level programming language known for its simplicity and readability.",
            "checksum": md5_checksum("Python is a high-level programming language..."),
        }
        
        result = await collection.insert_one(document)
        print(f"Document inserted: {result.acknowledged}")
```

### Adding Multiple Documents

For batch operations, iterate through your documents:

```python
async def add_multiple_documents():
    async with QdrantDatabase(
        logger=logger,
        path=Path("./qdrant_data"),
        embedder_factory=embedder_factory,
        embedding_cache_provider=embedding_cache_provider,
    ) as qdrant_db:
        collection = await qdrant_db.get_or_create_collection(
            name="documents",
            schema=BaseDocument,
            embedder_type=OpenAITextEmbedding3Large,
            document_loader=identity_loader,
        )
        
        documents = [
            {
                "id": ObjectId("doc_001"),
                "content": "Introduction to machine learning and artificial intelligence.",
                "checksum": md5_checksum("Introduction to machine learning..."),
            },
            {
                "id": ObjectId("doc_002"),
                "content": "Deep learning neural networks for image recognition.",
                "checksum": md5_checksum("Deep learning neural networks..."),
            },
            {
                "id": ObjectId("doc_003"),
                "content": "Natural language processing with transformers.",
                "checksum": md5_checksum("Natural language processing..."),
            },
        ]
        
        for doc in documents:
            await collection.insert_one(doc)
        
        print(f"Successfully added {len(documents)} documents")
```

### Adding Documents from Files (TXT, Markdown, etc.)

Extract text from text-based files and add them to your collection:

```python
from pathlib import Path
import asyncio

async def add_documents_from_files():
    async with QdrantDatabase(
        logger=logger,
        path=Path("./qdrant_data"),
        embedder_factory=embedder_factory,
        embedding_cache_provider=embedding_cache_provider,
    ) as qdrant_db:
        collection = await qdrant_db.get_or_create_collection(
            name="documents",
            schema=BaseDocument,
            embedder_type=OpenAITextEmbedding3Large,
            document_loader=identity_loader,
        )
        
        # List of file paths to process
        file_paths = [
            Path("./documents/article1.txt"),
            Path("./documents/article2.md"),
            Path("./documents/notes.txt"),
        ]
        
        for file_path in file_paths:
            if file_path.exists():
                # Read file content
                content = file_path.read_text(encoding="utf-8")
                
                # Create document
                document = {
                    "id": ObjectId(f"doc_{file_path.stem}"),
                    "content": content,
                    "checksum": md5_checksum(content),
                    "file_path": str(file_path),  # Optional metadata
                    "file_name": file_path.name,
                }
                
                await collection.insert_one(document)
                print(f"Added document from: {file_path.name}")
```

### Adding Documents from PDF Files

Extract text from PDF files using a library like `PyPDF2` or `pdfplumber`:

```python
import PyPDF2
from pathlib import Path

async def add_documents_from_pdf():
    async with QdrantDatabase(
        logger=logger,
        path=Path("./qdrant_data"),
        embedder_factory=embedder_factory,
        embedding_cache_provider=embedding_cache_provider,
    ) as qdrant_db:
        collection = await qdrant_db.get_or_create_collection(
            name="documents",
            schema=BaseDocument,
            embedder_type=OpenAITextEmbedding3Large,
            document_loader=identity_loader,
        )
        
        pdf_path = Path("./documents/report.pdf")
        
        # Extract text from PDF
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text_content = ""
            
            # Extract text from all pages
            for page_num, page in enumerate(pdf_reader.pages):
                text_content += f"\n--- Page {page_num + 1} ---\n"
                text_content += page.extract_text()
        
        # Create document
        document = {
            "id": ObjectId(f"pdf_{pdf_path.stem}"),
            "content": text_content,
            "checksum": md5_checksum(text_content),
            "source_type": "pdf",
            "file_name": pdf_path.name,
            "page_count": len(pdf_reader.pages),
        }
        
        await collection.insert_one(document)
        print(f"Added PDF document: {pdf_path.name} ({len(pdf_reader.pages)} pages)")
```

**Alternative: Using `pdfplumber` (Better for complex PDFs)**

```python
import pdfplumber

async def add_documents_from_pdf_advanced():
    async with QdrantDatabase(
        logger=logger,
        path=Path("./qdrant_data"),
        embedder_factory=embedder_factory,
        embedding_cache_provider=embedding_cache_provider,
    ) as qdrant_db:
        collection = await qdrant_db.get_or_create_collection(
            name="documents",
            schema=BaseDocument,
            embedder_type=OpenAITextEmbedding3Large,
            document_loader=identity_loader,
        )
        
        pdf_path = Path("./documents/report.pdf")
        
        with pdfplumber.open(pdf_path) as pdf:
            text_content = ""
            for page_num, page in enumerate(pdf.pages):
                text_content += f"\n--- Page {page_num + 1} ---\n"
                text_content += page.extract_text()
        
        document = {
            "id": ObjectId(f"pdf_{pdf_path.stem}"),
            "content": text_content,
            "checksum": md5_checksum(text_content),
            "source_type": "pdf",
            "file_name": pdf_path.name,
        }
        
        await collection.insert_one(document)
```

### Adding Documents from Other File Types

#### Word Documents (.docx)

```python
from docx import Document

async def add_documents_from_docx():
    async with QdrantDatabase(
        logger=logger,
        path=Path("./qdrant_data"),
        embedder_factory=embedder_factory,
        embedding_cache_provider=embedding_cache_provider,
    ) as qdrant_db:
        collection = await qdrant_db.get_or_create_collection(
            name="documents",
            schema=BaseDocument,
            embedder_type=OpenAITextEmbedding3Large,
            document_loader=identity_loader,
        )
        
        docx_path = Path("./documents/document.docx")
        doc = Document(docx_path)
        
        # Extract text from all paragraphs
        text_content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        
        document = {
            "id": ObjectId(f"docx_{docx_path.stem}"),
            "content": text_content,
            "checksum": md5_checksum(text_content),
            "source_type": "docx",
            "file_name": docx_path.name,
        }
        
        await collection.insert_one(document)
```

#### CSV Files

```python
import csv

async def add_documents_from_csv():
    async with QdrantDatabase(
        logger=logger,
        path=Path("./qdrant_data"),
        embedder_factory=embedder_factory,
        embedding_cache_provider=embedding_cache_provider,
    ) as qdrant_db:
        collection = await qdrant_db.get_or_create_collection(
            name="documents",
            schema=BaseDocument,
            embedder_type=OpenAITextEmbedding3Large,
            document_loader=identity_loader,
        )
        
        csv_path = Path("./documents/data.csv")
        
        with open(csv_path, "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            
            for row_num, row in enumerate(reader):
                # Convert row to text representation
                text_content = "\n".join([f"{key}: {value}" for key, value in row.items()])
                
                document = {
                    "id": ObjectId(f"csv_{csv_path.stem}_{row_num}"),
                    "content": text_content,
                    "checksum": md5_checksum(text_content),
                    "source_type": "csv",
                    "row_number": row_num,
                }
                
                await collection.insert_one(document)
```

### Chunking Large Documents

For very large documents, split them into smaller chunks:

```python
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

async def add_large_document_in_chunks():
    async with QdrantDatabase(
        logger=logger,
        path=Path("./qdrant_data"),
        embedder_factory=embedder_factory,
        embedding_cache_provider=embedding_cache_provider,
    ) as qdrant_db:
        collection = await qdrant_db.get_or_create_collection(
            name="documents",
            schema=BaseDocument,
            embedder_type=OpenAITextEmbedding3Large,
            document_loader=identity_loader,
        )
        
        # Read large document
        large_file = Path("./documents/large_book.txt")
        full_text = large_file.read_text(encoding="utf-8")
        
        # Split into chunks
        chunks = chunk_text(full_text, chunk_size=1000, overlap=200)
        
        # Add each chunk as a separate document
        for chunk_num, chunk in enumerate(chunks):
            document = {
                "id": ObjectId(f"chunk_{large_file.stem}_{chunk_num}"),
                "content": chunk,
                "checksum": md5_checksum(chunk),
                "source_file": large_file.name,
                "chunk_number": chunk_num,
                "total_chunks": len(chunks),
            }
            
            await collection.insert_one(document)
        
        print(f"Added {len(chunks)} chunks from {large_file.name}")
```

---

## 🔍 Searching Documents

### Similarity Search

Find documents similar to a query:

```python
async def search_documents():
    async with QdrantDatabase(
        logger=logger,
        path=Path("./qdrant_data"),
        embedder_factory=embedder_factory,
        embedding_cache_provider=embedding_cache_provider,
    ) as qdrant_db:
        collection = await qdrant_db.get_or_create_collection(
            name="documents",
            schema=BaseDocument,
            embedder_type=OpenAITextEmbedding3Large,
            document_loader=identity_loader,
        )
        
        # Search for similar documents
        results = await collection.find_similar_documents(
            filters={},
            query="machine learning algorithms",
            k=5,  # Return top 5 results
        )
        
        for result in results:
            print(f"Document ID: {result.document['id']}")
            print(f"Similarity Distance: {result.distance:.4f}")
            print(f"Content Preview: {result.document['content'][:100]}...")
            print("-" * 50)
```

### Filtered Search

Combine similarity search with filters:

```python
# Search with filters
results = await collection.find_similar_documents(
    filters={
        "source_type": {"$eq": "pdf"},
        "page_count": {"$gte": 10},
    },
    query="data analysis",
    k=3,
)
```

### Finding Documents by Metadata

```python
# Find documents by exact match
results = await collection.find({"file_name": {"$eq": "report.pdf"}})

# Find documents with range filters
results = await collection.find({
    "page_count": {"$gte": 5, "$lte": 20}
})

# Find documents with multiple conditions
results = await collection.find({
    "$and": [
        {"source_type": {"$eq": "pdf"}},
        {"page_count": {"$gte": 10}},
    ]
})
```

---

## 🎯 Complete Example: Building a Document Search Agent

Here's a complete example of building a document search system with Parlant and Qdrant:

```python
import asyncio
from pathlib import Path
from parlant.adapters.vector_db.qdrant import QdrantDatabase
from parlant.core.persistence.vector_database import BaseDocument, identity_loader
from parlant.core.nlp.embedding import OpenAITextEmbedding3Large, EmbedderFactory
from parlant.core.loggers import Logger
from parlant.core.common import md5_checksum, ObjectId
import PyPDF2

async def build_document_search_system():
    """Complete example: Build a document search system with Qdrant."""
    
    # Initialize Qdrant database (using Qdrant Cloud)
    async with QdrantDatabase(
        logger=Logger(),
        url="https://your-cluster-id.us-east4-0.gcp.cloud.qdrant.io",
        api_key="your-api-key",
        embedder_factory=EmbedderFactory(...),
        embedding_cache_provider=...,
    ) as qdrant_db:
        
        # Create collection
        collection = await qdrant_db.get_or_create_collection(
            name="knowledge_base",
            schema=BaseDocument,
            embedder_type=OpenAITextEmbedding3Large,
            document_loader=identity_loader,
        )
        
        # Add documents from various sources
        documents_dir = Path("./documents")
        
        # Process PDF files
        for pdf_file in documents_dir.glob("*.pdf"):
            with open(pdf_file, "rb") as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = "\n".join([page.extract_text() for page in pdf_reader.pages])
                
                doc = {
                    "id": ObjectId(f"pdf_{pdf_file.stem}"),
                    "content": text,
                    "checksum": md5_checksum(text),
                    "source_type": "pdf",
                    "file_name": pdf_file.name,
                }
                await collection.insert_one(doc)
        
        # Process text files
        for txt_file in documents_dir.glob("*.txt"):
            text = txt_file.read_text(encoding="utf-8")
            doc = {
                "id": ObjectId(f"txt_{txt_file.stem}"),
                "content": text,
                "checksum": md5_checksum(text),
                "source_type": "txt",
                "file_name": txt_file.name,
            }
            await collection.insert_one(doc)
        
        # Search functionality
        async def search_knowledge_base(query: str, k: int = 5):
            results = await collection.find_similar_documents(
                filters={},
                query=query,
                k=k,
            )
            return results
        
        # Example search
        search_results = await search_knowledge_base("machine learning", k=3)
        
        for result in search_results:
            print(f"Found: {result.document.get('file_name', 'Unknown')}")
            print(f"Relevance: {1 - result.distance:.2%}")
            print(f"Preview: {result.document['content'][:150]}...\n")

# Run the example
if __name__ == "__main__":
    asyncio.run(build_document_search_system())
```

---

## 🏗️ Architecture

### Dual Collection System

The Qdrant adapter uses a dual collection architecture:

- **Unembedded Collection**: Stores raw documents (source of truth)
- **Embedded Collection**: Stores documents with embeddings for search

**Benefits:**
- ✅ Efficient re-indexing when changing embedders
- ✅ Document migration and versioning
- ✅ Independent management of embeddings and metadata

### Version Management

Collections maintain version numbers for:
- Migration detection
- Incremental updates
- Consistency checks

---

## ⚡ Performance Tips

### 1. Use Embedding Cache

```python
from parlant.core.nlp.embedding import BasicEmbeddingCache

embedding_cache = BasicEmbeddingCache(...)

async with QdrantDatabase(
    logger=logger,
    path=Path("./qdrant_data"),
    embedder_factory=embedder_factory,
    embedding_cache_provider=lambda: embedding_cache,
) as qdrant_db:
    # Embeddings will be cached
    pass
```

### 2. Batch Operations

```python
# Process documents in batches
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    for doc in batch:
        await collection.insert_one(doc)
```

### 3. Optimize Collection Size

- Split large collections into smaller, topic-specific ones
- Use filters to narrow search scope
- Monitor collection sizes

---

## 🐛 Troubleshooting

### Connection Issues

**Problem**: Cannot connect to Qdrant server

**Solutions**:
1. Verify server is running: `curl http://localhost:6333/health`
2. Check URL and port are correct
3. For Qdrant Cloud, verify API key is valid
4. Check firewall/network settings

### Performance Issues

**Problem**: Slow search performance

**Solutions**:
1. Use embedding cache to avoid re-computation
2. Optimize collection size (split if too large)
3. Use appropriate distance metric (COSINE is default)
4. Monitor memory usage

### Data Persistence

- **In-memory**: Data lost on process exit
- **File system**: Data persists in specified directory
- **Remote server**: Data persists on Qdrant server

---

## 📊 Comparison with Other Adapters

| Feature | Qdrant | Chroma | Transient |
|---------|--------|--------|-----------|
| **Persistence** | ✅ Yes | ✅ Yes | ❌ No |
| **Remote Server** | ✅ Yes | ❌ No | ❌ No |
| **Cloud Support** | ✅ Yes | ❌ No | ❌ No |
| **Production Ready** | ✅ Yes | ✅ Yes | ❌ No |
| **In-Memory Mode** | ✅ Yes | ❌ No | ✅ Yes |
| **Performance** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## 📚 Additional Resources

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Qdrant Python Client](https://github.com/qdrant/qdrant-client)
- [Qdrant Cloud](https://cloud.qdrant.io/)
- [Qdrant Examples](https://github.com/qdrant/examples)

---

## 💡 Best Practices

1. ✅ **Use persistent storage** for production
2. ✅ **Implement error handling** for all operations
3. ✅ **Monitor collection versions** for migrations
4. ✅ **Use embedding caches** for performance
5. ✅ **Clean up unused collections**
6. ✅ **Test migrations** in development first
7. ✅ **Chunk large documents** for better search results
8. ✅ **Add metadata** to documents for filtering

---

**Ready to build powerful semantic search with Qdrant and Parlant!** 🚀
