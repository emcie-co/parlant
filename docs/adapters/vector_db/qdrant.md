# Qdrant Vector Database

The Qdrant adapter provides persistent vector storage for Parlant using Qdrant's vector database. This replaces the default in-memory storage with production-ready persistence.

For general Parlant usage, see the [official documentation](https://www.parlant.io/docs/).

## Prerequisites

1. **Install Qdrant adapter**: `pip install parlant[qdrant]`
2. **Choose storage**: Local file system or Qdrant Cloud

## Quick Start

### Simple One-Line Setup (Recommended)

The easiest way to use Qdrant with Parlant:

```python
import parlant.sdk as p
from pathlib import Path
from parlant.adapters.vector_db.qdrant import configure_qdrant_for_parlant

# Local storage
async def configure_container(container: p.Container) -> p.Container:
    return await configure_qdrant_for_parlant(
        container,
        path=Path("./qdrant_data")
    )

# OR Qdrant Cloud  
async def configure_container(container: p.Container) -> p.Container:
    return await configure_qdrant_for_parlant(
        container,
        url="https://your-cluster-id.us-east4-0.gcp.cloud.qdrant.io",
        api_key="your-api-key-here"
    )

async def main():
    async with p.Server(configure_container=configure_container) as server:
        agent = await server.create_agent(
            name="My Agent",
            description="Agent using Qdrant for persistent storage",
        )
        # All vector operations now use Qdrant automatically
```

The `configure_qdrant_for_parlant()` function automatically:
- Creates a QdrantDatabase instance  
- Overrides all 4 built-in vector stores (GlossaryStore, CannedResponseStore, CapabilityStore, JourneyStore)
- Ensures all vector data goes to Qdrant instead of transient storage

### Advanced Setup (Manual)

If you need more control over the QdrantDatabase configuration:

```python
import parlant.sdk as p
from pathlib import Path
from parlant.adapters.vector_db.qdrant import QdrantDatabase, setup_qdrant_for_parlant
from parlant.core.nlp.embedding import EmbedderFactory, EmbeddingCache
from parlant.core.loggers import Logger

async def configure_container(container: p.Container) -> p.Container:
    # Create custom QdrantDatabase instance
    qdrant_db = await QdrantDatabase(
        logger=container[Logger],
        path=Path("./qdrant_data"),  # or url/api_key for cloud
        embedder_factory=EmbedderFactory(container),
        embedding_cache_provider=lambda: container[EmbeddingCache],
        # Add any custom parameters here
    ).__aenter__()
    
    # Setup all vector stores to use this Qdrant instance
    return await setup_qdrant_for_parlant(container, qdrant_db)
```

## How It Works

Qdrant uses two collections per data type:
- **Raw documents** (`{name}_unembedded`): Source of truth
- **Embedded documents** (`{name}_{EmbedderType}`): For vector search

When you change embedders, only the embedded collection is regenerated. Collections are created and synced automatically.

## Verification

To verify Qdrant integration is working correctly:

### Check Collections
**Qdrant Cloud:** Collections appear in your Qdrant dashboard with names like:
- `glossary_OpenAITextEmbedding3Large`
- `glossary_unembedded`
- `capabilities_OpenAITextEmbedding3Large`
- `canned_responses_OpenAITextEmbedding3Large`

**Local Qdrant:** A folder is created at your specified path containing Qdrant database files.

### Confirm No Transient Storage
When Qdrant is properly configured:
- **No vector files** are created in the `parlant-data` folder
- Vector data is stored only in Qdrant (cloud or local)
- Data persists across server restarts

### Test Vector Search
Create terms and test them:
```python
term = await agent.create_term(
    name="Test Term",
    description="This should be stored in Qdrant",
)
# Then chat with agent about "test term" - it should understand via vector search
```

---

## Custom Stores

To create custom vector stores beyond Parlant's built-in stores, accept a `VectorDatabase` in your constructor:

```python
class MyCustomStore:
    def __init__(self, vector_db: VectorDatabase):
        self._vector_db = vector_db
    
    async def __aenter__(self):
        self._collection = await self._vector_db.get_or_create_collection(
            name="my_collection",
            schema=MyDocumentSchema,
            embedder_type=MyEmbedderType,
            document_loader=my_loader,
        )
        return self
```

Register in `configure_container`:

```python
async def configure_container(container: p.Container) -> p.Container:
    # Register Qdrant (as shown above)
    container[VectorDatabase] = get_qdrant_vector_db
    
    # Register custom store
    async def get_my_store() -> MyCustomStore:
        vector_db = await container[VectorDatabase]()
        return await MyCustomStore(vector_db).__aenter__()
    
    container[MyCustomStore] = get_my_store
    return container
```

**More examples**: [Engine Extensions](https://www.parlant.io/docs/advanced/engine-extensions/#registering-components) and [Quickstart Examples](https://www.parlant.io/docs/quickstart/examples).

---

## Common Issues

### Integration Not Working (Still Using Transient Storage)
**Symptoms:**
- No collections appear in Qdrant dashboard
- Vector data appears in `parlant-data` folder
- Data lost on server restart

**Solution:** Use the `configure_qdrant_for_parlant()` function instead of registering `VectorDatabase`:

```python
# ❌ This doesn't work (stores already created with transient storage)
async def configure_container(container: p.Container) -> p.Container:
    container[VectorDatabase] = my_qdrant_factory
    return container

# ✅ This works (automatically overrides all vector stores)  
async def configure_container(container: p.Container) -> p.Container:
    return await configure_qdrant_for_parlant(
        container,
        url="your-cluster-url",
        api_key="your-api-key"
    )
```

### Windows File Locks
On Windows, use `async with` context manager. The adapter automatically handles file lock retries.

### Collection Sync
Collections auto-sync when embedders or schemas change. Large collections may take time on first access.

### Embedder Changes  
When changing embedder types, old embedded collections persist until manually deleted.

### Performance
Use Qdrant Cloud or server for production - local mode doesn't support payload indexes.

---

## Troubleshooting

### Connection Issues
- **Local**: Check path exists and is writable
- **Remote**: Verify URL and API key

### Slow Performance  
- Use embedding cache
- Use Qdrant Cloud/server for payload indexes
- Consider splitting large collections

### Data Not Persisting
- Check file path is correct and writable
- Verify connection settings for remote servers

---

## Requirements

- Python 3.8+
- `pip install parlant[qdrant]`
- Writable directory (for local storage) or Qdrant Cloud account

## Key Features

- **Persistent storage**: Replaces in-memory storage with production-ready persistence
- **Dual collections**: Efficient re-indexing when changing embedders or schemas  
- **Auto-sync**: Collections automatically sync when configurations change
- **Windows support**: Automatic file lock handling

