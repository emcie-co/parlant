# Qdrant Vector Database

> **Production-ready vector database backend for Parlant** — Integration guide, usage patterns, and common friction points.

## Quick Start

```python
from pathlib import Path
from parlant.adapters.vector_db.qdrant import QdrantDatabase
from parlant.core.nlp.embedding import EmbedderFactory, NullEmbeddingCache
from parlant.core.loggers import Logger

async with QdrantDatabase(
    logger=logger,
    path=Path("./qdrant_data"),  # Local storage
    embedder_factory=embedder_factory,
    embedding_cache_provider=NullEmbeddingCache,
) as qdrant_db:
    collection = await qdrant_db.get_or_create_collection(
        name="documents",
        schema=BaseDocument,
        embedder_type=OpenAITextEmbedding3Large,
        document_loader=identity_loader,
    )
```

---

## Configuration

The adapter supports three deployment modes, each suited for different use cases:

### In-Memory Mode (Testing)

Perfect for unit tests and development. Data is ephemeral and lost when the process ends.

```python
async with QdrantDatabase(
    logger=logger,
    path=None,  # None triggers in-memory mode
    embedder_factory=embedder_factory,
    embedding_cache_provider=NullEmbeddingCache,
) as qdrant_db:
    # Your code here
    pass
```

### Local File System (Development/Production)

Persistent storage on your local file system. Ideal for single-instance deployments.

```python
async with QdrantDatabase(
    logger=logger,
    path=Path("./qdrant_data"),
    embedder_factory=embedder_factory,
    embedding_cache_provider=NullEmbeddingCache,
) as qdrant_db:
    # Your code here
    pass
```

**⚠️ Windows Note**: File locks may persist after closing. The adapter handles this automatically with retries and cleanup, but ensure you're using the `async with` context manager.

### Qdrant Cloud (Production)

Connect to Qdrant Cloud for managed infrastructure with high availability.

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

## Architecture: Dual Collection System

The Qdrant adapter uses a **dual collection architecture** that's fundamental to understanding how it works:

- **Unembedded Collection** (`{name}_unembedded`): Stores raw documents (source of truth)
- **Embedded Collection** (`{name}_{EmbedderType}`): Stores documents with embeddings for search

### Why Two Collections?

1. **Efficient Re-indexing**: When changing embedders, only the embedded collection needs regeneration
2. **Document Migration**: The unembedded collection acts as SSOT during schema migrations
3. **Version Management**: Both collections maintain version numbers for consistency checks

### Collection Naming

- Unembedded: `{name}_unembedded`
- Embedded: `{name}_{EmbedderType.__name__}` (e.g., `documents_OpenAITextEmbedding3Large`)

**Note**: Collections are automatically created and synchronized. You don't manage them directly.

---

## Usage Patterns

### Basic Document Operations

```python
from parlant.core.persistence.vector_database import BaseDocument, identity_loader
from parlant.core.common import ObjectId, md5_checksum

# Create or get collection
collection = await qdrant_db.get_or_create_collection(
    name="knowledge_base",
    schema=BaseDocument,
    embedder_type=OpenAITextEmbedding3Large,
    document_loader=identity_loader,
)

# Insert a document
document = {
    "id": ObjectId("doc_001"),
    "content": "Python is a high-level programming language known for its simplicity.",
    "checksum": md5_checksum("Python is a high-level programming language..."),
}

result = await collection.insert_one(document)

# Search for similar documents
results = await collection.find_similar_documents(
    filters={},
    query="programming languages",
    k=5,
)

for result in results:
    print(f"Document: {result.document['id']}")
    print(f"Similarity: {1 - result.distance:.2%}")
```

### Filtered Search

```python
# Search with metadata filters
results = await collection.find_similar_documents(
    filters={
        "category": {"$eq": "technical"},
        "priority": {"$gte": 5},
    },
    query="machine learning",
    k=3,
)

# Find documents by exact match
documents = await collection.find({
    "status": {"$eq": "published"}
})
```

### Logical Operators

```python
# AND operator
results = await collection.find({
    "$and": [
        {"category": {"$eq": "technical"}},
        {"status": {"$eq": "published"}},
    ]
})

# OR operator
results = await collection.find({
    "$or": [
        {"category": {"$eq": "technical"}},
        {"category": {"$eq": "scientific"}},
    ]
})

# Nested logical operators
results = await collection.find({
    "$and": [
        {
            "$or": [
                {"author": {"$eq": "Alice"}},
                {"author": {"$eq": "Bob"}},
            ]
        },
        {"status": {"$eq": "published"}},
    ]
})
```

### Supported Filter Operators

- `$eq`: Equal
- `$ne`: Not equal
- `$gt`, `$gte`, `$lt`, `$lte`: Range comparisons
- `$in`: Value in list
- `$nin`: Value not in list
- `$and`, `$or`: Logical operators

---

## Integration Example: Healthcare Knowledge Base

Here's a complete example showing how to integrate Qdrant with a Parlant agent, similar to the [healthcare example](https://github.com/emcie-co/parlant/blob/develop/examples/healthcare.py):

```python
import parlant.sdk as p
import asyncio
from pathlib import Path
from parlant.adapters.vector_db.qdrant import QdrantDatabase
from parlant.core.nlp.embedding import EmbedderFactory, NullEmbeddingCache
from parlant.core.persistence.vector_database import BaseDocument, identity_loader
from parlant.core.common import ObjectId, md5_checksum


async def setup_medical_knowledge_base(agent: p.Agent) -> None:
    """Set up a medical knowledge base using Qdrant."""
    
    # Initialize Qdrant database
    async with QdrantDatabase(
        logger=agent._logger,  # Use agent's logger
        path=Path("./medical_kb"),
        embedder_factory=EmbedderFactory(agent._container),
        embedding_cache_provider=NullEmbeddingCache,
    ) as qdrant_db:
        
        # Create collection for medical documents
        collection = await qdrant_db.get_or_create_collection(
            name="medical_documents",
            schema=BaseDocument,
            embedder_type=OpenAITextEmbedding3Large,
            document_loader=identity_loader,
        )
        
        # Add medical knowledge documents
        medical_docs = [
            {
                "id": ObjectId("doc_001"),
                "content": "Hypertension, also known as high blood pressure, is a condition where the force of blood against artery walls is too high.",
                "checksum": md5_checksum("Hypertension, also known as..."),
                "category": "cardiology",
                "topic": "hypertension",
            },
            {
                "id": ObjectId("doc_002"),
                "content": "Diabetes is a chronic condition that affects how your body turns food into energy.",
                "checksum": md5_checksum("Diabetes is a chronic condition..."),
                "category": "endocrinology",
                "topic": "diabetes",
            },
            {
                "id": ObjectId("doc_003"),
                "content": "The recommended daily water intake is about 8 glasses or 2 liters for adults.",
                "checksum": md5_checksum("The recommended daily water intake..."),
                "category": "general",
                "topic": "hydration",
            },
        ]
        
        for doc in medical_docs:
            await collection.insert_one(doc)


@p.tool
async def search_medical_knowledge(
    context: p.ToolContext,
    query: str,
) -> p.ToolResult:
    """Search the medical knowledge base for relevant information."""
    
    async with QdrantDatabase(
        logger=context.logger,
        path=Path("./medical_kb"),
        embedder_factory=EmbedderFactory(context.container),
        embedding_cache_provider=NullEmbeddingCache,
    ) as qdrant_db:
        
        collection = await qdrant_db.get_or_create_collection(
            name="medical_documents",
            schema=BaseDocument,
            embedder_type=OpenAITextEmbedding3Large,
            document_loader=identity_loader,
        )
        
        # Search for relevant documents
        results = await collection.find_similar_documents(
            filters={},
            query=query,
            k=3,
        )
        
        # Format results
        knowledge = [
            {
                "content": r.document["content"],
                "category": r.document.get("category", "unknown"),
                "relevance": f"{(1 - r.distance) * 100:.1f}%",
            }
            for r in results
        ]
        
        return p.ToolResult(data=knowledge)


async def create_medical_agent(server: p.Server) -> p.Agent:
    """Create a medical agent with Qdrant-powered knowledge base."""
    
    agent = await server.create_agent(
        name="Medical Assistant",
        description="Helps patients with medical questions using a knowledge base.",
    )
    
    # Set up the knowledge base
    await setup_medical_knowledge_base(agent)
    
    # Add guideline that uses the knowledge base
    await agent.create_guideline(
        condition="The patient asks a medical question",
        action="Search the medical knowledge base and provide relevant information",
        tools=[search_medical_knowledge],
    )
    
    return agent


async def main() -> None:
    async with p.Server() as server:
        agent = await create_medical_agent(server)
        # Your agent is now ready to use the Qdrant-powered knowledge base


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Friction Points & Gotchas

### Windows File Locks

**Problem**: On Windows, Qdrant file locks may persist after closing the database, preventing re-opening.

**Solution**: The adapter handles this automatically:
- Retries with exponential backoff (up to 5 attempts)
- Explicit cleanup with `gc.collect()` and brief delay
- Proper client closure in `__aexit__`

**Best Practice**: Always use `async with` context manager and wait a moment before reopening the same path.

### Collection Synchronization

**Problem**: Embedded and unembedded collections can get out of sync.

**Solution**: The adapter automatically syncs on `get_or_create_collection`:
- Checks version numbers stored in metadata collection
- Re-indexes if versions differ
- Migrates documents if schema changed

**Note**: Large collections may take time to sync on first access. Monitor logs for progress.

### Embedder Changes

**Problem**: Changing embedder types requires re-indexing.

**Solution**: The dual collection system handles this:
- Unembedded collection remains unchanged
- New embedded collection is created with new embedder
- Old embedded collection can be manually deleted if needed

**Note**: Collection names include embedder type, so old collections persist until manually deleted.

### Document Loading Failures

**Problem**: Documents that fail to load during migration are stored in `failed_migrations` collection.

**Solution**: Check the `failed_migrations` collection for debugging:

```python
failed_collection = await qdrant_db.get_or_create_collection(
    "failed_migrations",
    BaseDocument,
    NullEmbedder,
    identity_loader,
)
failed_docs = await failed_collection.find({})
```

### Filter Performance

**Problem**: Filters may be slow without proper indexes.

**Solution**: The adapter automatically creates payload indexes for common fields. If filters are still slow:
- Ensure fields are indexed (check Qdrant collection info)
- Consider splitting large collections
- Use more specific filters to narrow search scope

**Note**: Payload indexes have no effect in local Qdrant. Use server Qdrant for production performance.

### Memory Usage

**Problem**: Large collections can consume significant memory.

**Solution**:
- Use embedding cache to avoid re-computation
- Consider splitting collections by topic/domain
- Use local file system or cloud for large datasets (not in-memory)

### Version Storage

**Problem**: Version metadata was previously stored in collections, causing `find()` to return version points.

**Solution**: Versions are now stored in the metadata collection, keeping document collections clean. This is handled automatically.

---

## Troubleshooting

### Cannot Connect to Qdrant Server

1. **Local file system**: Check path exists and is writable
2. **Remote server**: Verify URL and API key are correct
3. **Docker**: Ensure container is running: `docker ps`
4. **Health check**: `curl http://localhost:6333/health` (for local server)

### Slow Search Performance

1. Use embedding cache to avoid re-computation
2. Check collection size — consider splitting if too large
3. Verify payload indexes exist for filtered fields
4. Monitor memory usage
5. Use Qdrant Cloud or server instance for better performance

### Data Not Persisting

- **In-memory mode**: Data is lost on process exit (expected)
- **File system**: Check path is correct and writable
- **Remote server**: Verify connection and API key

### Collection Not Found Errors

- Ensure you're using `get_or_create_collection` (not `get_collection`)
- Check collection names match (including embedder type suffix)
- Verify both embedded and unembedded collections exist

### Filter Errors

If filters fail due to missing indexes, the adapter falls back to in-memory filtering. This works but is slower. For production, ensure proper indexes are created.

---

## Requirements

- Python 3.8+
- `qdrant-client` package (install with `pip install qdrant-client` or via Poetry extras: `poetry install --extras qdrant`)
- For local file system: writable directory
- For cloud: Qdrant Cloud account and API key
- For Docker: Docker installed (optional, for local server)

---

## Key Differences from Other Adapters

- **Dual collection system**: Unique to Qdrant adapter for efficient re-indexing
- **Windows file lock handling**: Automatic retry and cleanup
- **Version-based synchronization**: Automatic detection and re-indexing
- **Payload indexing**: Automatic index creation for filtering (server Qdrant only)
- **Metadata collection**: Versions stored separately from document collections

---

## Best Practices

1. ✅ **Use persistent storage** for production (file system or cloud)
2. ✅ **Implement error handling** for all operations
3. ✅ **Use embedding caches** for performance
4. ✅ **Monitor collection versions** for migrations
5. ✅ **Clean up unused collections** manually
6. ✅ **Test migrations** in development first
7. ✅ **Chunk large documents** for better search results
8. ✅ **Add metadata** to documents for filtering
9. ✅ **Use `async with`** context manager for proper cleanup
10. ✅ **Use Qdrant Cloud or server** for production payload indexing

