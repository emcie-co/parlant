# Qdrant Vector Database

The Qdrant adapter provides persistent vector storage for Parlant using Qdrant's vector database. This replaces the default in-memory storage with production-ready persistence.

For general Parlant usage, see the [official documentation](https://www.parlant.io/docs/).

## Prerequisites

1. **Install Qdrant adapter**: `pip install parlant[qdrant]`
2. **Choose storage**: Local file system or Qdrant Cloud

## Quick Start

### Setup (Manual)

```python
import parlant.sdk as p
from pathlib import Path
from contextlib import AsyncExitStack
from parlant.adapters.vector_db.qdrant import QdrantDatabase
from parlant.core.nlp.embedding import EmbedderFactory, EmbeddingCache, Embedder
from parlant.core.loggers import Logger
from parlant.core.nlp.service import NLPService
from parlant.core.glossary import GlossaryVectorStore, GlossaryStore
from parlant.core.canned_responses import CannedResponseVectorStore, CannedResponseStore
from parlant.core.capabilities import CapabilityVectorStore, CapabilityStore
from parlant.core.journeys import JourneyVectorStore, JourneyStore
from parlant.adapters.db.transient import TransientDocumentDatabase

async def configure_container(container: p.Container) -> p.Container:
    embedder_factory = EmbedderFactory(container)

    async def get_embedder_type() -> type[Embedder]:
        return type(await container[NLPService].get_embedder())
    
    exit_stack = AsyncExitStack()
    qdrant_db = await exit_stack.enter_async_context(
        QdrantDatabase(
            logger=container[Logger],
            path=Path("./qdrant_data"),
            embedder_factory=EmbedderFactory(container),
            embedding_cache_provider=lambda: container[EmbeddingCache],
        )
    )
    
    # For Qdrant Cloud, replace the above with:
    # qdrant_db = await exit_stack.enter_async_context(
    #     QdrantDatabase(
    #         logger=container[Logger],
    #         url="https://your-cluster-id.us-east4-0.gcp.cloud.qdrant.io",
    #         api_key="your-api-key-here",
    #         embedder_factory=EmbedderFactory(container),
    #         embedding_cache_provider=lambda: container[EmbeddingCache],
    #     )
    # )
    
    # Configure stores using vector database
    container[GlossaryStore] = await exit_stack.enter_async_context(
        GlossaryVectorStore(
            id_generator=container[p.IdGenerator],
            vector_db=qdrant_db,
            document_db=TransientDocumentDatabase(),
            embedder_factory=embedder_factory,
            embedder_type_provider=get_embedder_type,
        )  # type: ignore
    )
    
    container[CannedResponseStore] = await exit_stack.enter_async_context(
        CannedResponseVectorStore(
            id_generator=container[p.IdGenerator],
            vector_db=qdrant_db,
            document_db=TransientDocumentDatabase(),
            embedder_factory=embedder_factory,
            embedder_type_provider=get_embedder_type,
        )  # type: ignore
    )
    
    container[CapabilityStore] = await exit_stack.enter_async_context(
        CapabilityVectorStore(
            id_generator=container[p.IdGenerator],
            vector_db=qdrant_db,
            document_db=TransientDocumentDatabase(),
            embedder_factory=embedder_factory,
            embedder_type_provider=get_embedder_type,
        )  # type: ignore
    )
    
    container[JourneyStore] = await exit_stack.enter_async_context(
        JourneyVectorStore(
            id_generator=container[p.IdGenerator],
            vector_db=qdrant_db,
            document_db=TransientDocumentDatabase(),
            embedder_factory=embedder_factory,
            embedder_type_provider=get_embedder_type,
        )  # type: ignore
    )
    
    return container

async def main():
    async with p.Server(configure_container=configure_container) as server:
        agent = await server.create_agent(
            name="My Agent",
            description="Agent using Qdrant for persistent storage",
        )
        
        # Test: Create a term to verify Qdrant is working
        term = await agent.create_term(
            name="Example Term",
            description="This is stored in Qdrant",
        )
        print(f"Created term: {term.name}")
        # All vector operations now use Qdrant
```


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
Create terms and test persistence:
```python
term = await agent.create_term(
    name="Test Term",
    description="This should be stored in Qdrant",
)
# Then chat with agent about "test term" - it should understand via vector search

# Test persistence: close the server and run again
# The term should still be available after restart
```

---

## Common Issues

### Integration Not Working (Still Using Transient Storage)
**Symptoms:**
- No collections appear in Qdrant dashboard
- Vector data appears in `parlant-data` folder
- Data lost on server restart

**Solution:** Ensure all vector stores are properly configured with Qdrant in your `configure_container` function. Make sure you're using `AsyncExitStack` to properly manage the Qdrant database and vector stores lifecycle.

### Windows File Locks
On Windows, use `async with` context manager. The adapter automatically handles file lock retries.

### Collection Sync
Collections auto-sync when embedders or schemas change. Large collections may take time on first access.

### Embedder Changes  
When changing embedder types, old embedded collections persist until manually deleted.

### Performance
Use Qdrant Cloud or server for production - local mode doesn't support payload indexes. You'll see a warning about this when using local Qdrant, which is expected and can be ignored.

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
- Test by closing the server and restarting—data should persist

---

## Requirements

- Python 3.8+
- `pip install parlant[qdrant]`
- Writable directory (for local storage) or Qdrant Cloud account

## Key Features

- **Persistent storage**: Replaces in-memory storage with production-ready persistence
- **Auto-sync**: Collections automatically sync when embedders or schemas change
- **Windows support**: Automatic file lock handling

