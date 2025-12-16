# Weaviate Vector Database

The Weaviate adapter provides persistent vector storage for Parlant using Weaviate's vector database. This replaces the default in-memory storage with production-ready persistence.

For general Parlant usage, see the [official documentation](https://www.parlant.io/docs/).

## Prerequisites

1. **Install Weaviate adapter**: `pip install parlant[weaviate]`
2. **Choose storage**: Local Weaviate instance or Weaviate Cloud

## Quick Start

### Setup (Manual)

```python
import parlant.sdk as p
from contextlib import AsyncExitStack
from parlant.adapters.vector_db.weaviate import WeaviateDatabase
from parlant.core.nlp.embedding import EmbedderFactory, EmbeddingCache, Embedder
from parlant.core.loggers import Logger
from parlant.core.nlp.service import NLPService
from parlant.core.glossary import GlossaryVectorStore, GlossaryStore
from parlant.core.canned_responses import CannedResponseVectorStore, CannedResponseStore
from parlant.core.capabilities import CapabilityVectorStore, CapabilityStore
from parlant.core.journeys import JourneyVectorStore, JourneyStore
from parlant.adapters.db.transient import TransientDocumentDatabase

# Global exit stack to manage resources created in configure_container
# This ensures proper cleanup when the server shuts down
service_exit_stack = AsyncExitStack()

async def configure_container(container: p.Container) -> p.Container:
    embedder_factory = EmbedderFactory(container)

    async def get_embedder_type() -> type[Embedder]:
        return type(await container[NLPService].get_embedder())
    
    # For local Weaviate instance:
    weaviate_db = await service_exit_stack.enter_async_context(
        WeaviateDatabase(
            logger=container[Logger],
            url="http://localhost:8080",  # Default local Weaviate URL
            embedder_factory=EmbedderFactory(container),
            embedding_cache_provider=lambda: container[EmbeddingCache],
        )
    )
    
    # For Weaviate Cloud, use:
    # weaviate_db = await service_exit_stack.enter_async_context(
    #     WeaviateDatabase(
    #         logger=container[Logger],
    #         url="https://YOUR_CLUSTER_URL.weaviate.network",
    #         api_key="YOUR_API_KEY",
    #         embedder_factory=EmbedderFactory(container),
    #         embedding_cache_provider=lambda: container[EmbeddingCache],
    #     )
    # )
    
    # Configure stores using vector database
    container[GlossaryStore] = await service_exit_stack.enter_async_context(
        GlossaryVectorStore(
            id_generator=container[p.IdGenerator],
            vector_db=weaviate_db,
            document_db=TransientDocumentDatabase(),
            embedder_factory=embedder_factory,
            embedder_type_provider=get_embedder_type,
        )  # type: ignore
    )
    
    container[CannedResponseStore] = await service_exit_stack.enter_async_context(
        CannedResponseVectorStore(
            id_generator=container[p.IdGenerator],
            vector_db=weaviate_db,
            document_db=TransientDocumentDatabase(),
            embedder_factory=embedder_factory,
            embedder_type_provider=get_embedder_type,
        )  # type: ignore
    )
    
    container[CapabilityStore] = await service_exit_stack.enter_async_context(
        CapabilityVectorStore(
            id_generator=container[p.IdGenerator],
            vector_db=weaviate_db,
            document_db=TransientDocumentDatabase(),
            embedder_factory=embedder_factory,
            embedder_type_provider=get_embedder_type,
        )  # type: ignore
    )
    
    container[JourneyStore] = await service_exit_stack.enter_async_context(
        JourneyVectorStore(
            id_generator=container[p.IdGenerator],
            vector_db=weaviate_db,
            document_db=TransientDocumentDatabase(),
            embedder_factory=embedder_factory,
            embedder_type_provider=get_embedder_type,
        )  # type: ignore
    )
    
    return container

async def main():
    try:
        async with p.Server(configure_container=configure_container) as server:
            agent = await server.create_agent(
                name="My Agent",
                description="Agent using Weaviate for persistent storage",
            )
            
            # Test: Create a term to verify Weaviate is working
            term = await agent.create_term(
                name="Example Term",
                description="This is stored in Weaviate",
            )
            print(f"Created term: {term.name}")
            # All vector operations now use Weaviate
    finally:
        # Ensure proper cleanup of resources
        await service_exit_stack.aclose()
```


## Verification

To verify Weaviate integration is working correctly:

### Check Collections
**Weaviate Cloud:** Collections appear in your Weaviate dashboard with names like:
- `glossary_OpenAITextEmbedding3Large`
- `glossary_unembedded`
- `capabilities_OpenAITextEmbedding3Large`
- `canned_responses_OpenAITextEmbedding3Large`

**Local Weaviate:** Collections are created in your local Weaviate instance and can be viewed via the Weaviate console or API.

### Confirm No Transient Storage
When Weaviate is properly configured:
- **No vector files** are created in the `parlant-data` folder
- Vector data is stored only in Weaviate (cloud or local)
- Data persists across server restarts

### Test Vector Search
Create terms and test persistence:
```python
term = await agent.create_term(
    name="Test Term",
    description="This should be stored in Weaviate",
)
# Then chat with agent about "test term" - it should understand via vector search

# Test persistence: close the server and run again
# The term should still be available after restart
```

---

## Common Issues

### Integration Not Working (Still Using Transient Storage)
**Symptoms:**
- No collections appear in Weaviate dashboard/console
- Vector data appears in `parlant-data` folder
- Data lost on server restart

**Solution:** Ensure all vector stores are properly configured with Weaviate in your `configure_container` function. **Important:** Use a global `AsyncExitStack` (not local) to properly manage the Weaviate database and vector stores lifecycle. Always call `await service_exit_stack.aclose()` in a `finally` block to ensure proper cleanup.

### Connection Issues
On Windows, the adapter automatically retries connections. If you see connection errors:
- Ensure Weaviate is running (for local instances)
- Check the URL and port (default: `http://localhost:8080`)
- Verify API key for Weaviate Cloud instances
- The adapter includes automatic retry logic with exponential backoff for connection issues

### Collection Sync
Collections auto-sync when embedders or schemas change. Large collections may take time on first access.

### Embedder Changes  
When changing embedder types, old embedded collections persist until manually deleted.

### Performance
Use Weaviate Cloud or server for production. Local mode works well for development and testing.

---

## Troubleshooting

### Connection Issues
- **Local**: Ensure Weaviate is running and accessible at the specified URL
- **Remote**: Verify URL and API key are correct
- Check network connectivity and firewall settings

### Slow Performance  
- Use embedding cache
- Use Weaviate Cloud/server for better performance
- Consider splitting large collections

### Data Not Persisting
- Check Weaviate instance is running and accessible
- Verify connection settings for remote servers
- Test by closing the server and restarting—data should persist

### Resource Warnings or Connection Leaks
If you see warnings about unclosed connections or SSL sockets:
- Ensure you're using a global `AsyncExitStack` (not local)
- Always call `await service_exit_stack.aclose()` in a `finally` block
- The adapter includes automatic cleanup with timeout protection
- On Windows, allow a brief moment for SSL connections to release (handled automatically)

### Hanging on Shutdown
If the server hangs during shutdown:
- This is typically related to server background tasks, not Weaviate
- The Weaviate adapter properly closes all connections with timeout protection
- Verify your `AsyncExitStack` cleanup is being called
- Check for other background tasks that may need cleanup

---

## Requirements

- Python 3.8+
- `pip install parlant[weaviate]`
- Running Weaviate instance (local or cloud) or Weaviate Cloud account

## Key Features

- **Persistent storage**: Replaces in-memory storage with production-ready persistence
- **Auto-sync**: Collections automatically sync when embedders or schemas change
- **Windows support**: Automatic connection retry handling and proper SSL connection cleanup
- **Cloud and local**: Supports both Weaviate Cloud and local instances
- **Proper cleanup**: Automatic connection cleanup with timeout protection to prevent resource leaks
- **Async-safe**: All operations are properly wrapped to avoid blocking the event loop

## Important Notes

### Resource Management
The Weaviate adapter uses async context managers for proper resource cleanup. Always:
1. Use a **global** `AsyncExitStack` (not local to `configure_container`)
2. Call `await service_exit_stack.aclose()` in a `finally` block
3. This ensures SSL connections and other resources are properly released

### Connection Cleanup
The adapter includes:
- Automatic timeout protection (5 seconds) for client close operations
- Proper SSL connection cleanup on Windows
- Garbage collection assistance for persistent connections
- All operations wrapped in `asyncio.to_thread()` to avoid blocking

### Performance
- Connection pooling is handled automatically by the Weaviate client
- Embedding cache integration reduces redundant API calls
- Collections are automatically indexed and synced

