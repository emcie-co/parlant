# Pinecone Vector Database

The Pinecone adapter provides persistent vector storage for Parlant using Pinecone's managed vector database. This replaces the default in-memory storage with production-ready cloud persistence.

For general Parlant usage, see the [official documentation](https://www.parlant.io/docs/).

## Prerequisites

1. **Install Pinecone client**: `pip install pinecone`
2. **Pinecone Account**: Sign up for a Pinecone account at [pinecone.io](https://www.pinecone.io)
3. **API Key**: Get your API key from the Pinecone dashboard
4. **Create Index**: Create a Pinecone index with **3072 dimensions** in your Pinecone dashboard (required, otherwise you'll get an error)

## Quick Start

### Setup (Manual)

```python
import parlant.sdk as p
from contextlib import AsyncExitStack
from parlant.adapters.vector_db.pinecone import PineconeDatabase
from parlant.core.nlp.embedding import EmbedderFactory, EmbeddingCache, Embedder
from parlant.core.loggers import Logger
from parlant.core.nlp.service import NLPService
from parlant.core.glossary import GlossaryVectorStore, GlossaryStore
from parlant.core.canned_responses import CannedResponseVectorStore, CannedResponseStore
from parlant.core.capabilities import CapabilityVectorStore, CapabilityStore
from parlant.core.journeys import JourneyVectorStore, JourneyStore
from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.core.common import IdGenerator
import os

async def configure_container(container: p.Container) -> p.Container:
    embedder_factory = EmbedderFactory(container)

    async def get_embedder_type() -> type[Embedder]:
        return type(await container[NLPService].get_embedder())
    
    exit_stack = AsyncExitStack()
    pinecone_db = await exit_stack.enter_async_context(
        PineconeDatabase(
            logger=container[Logger],
            api_key=os.getenv("PINECONE_API_KEY"),  # Or pass directly
            index_name=os.getenv("PINECONE_INDEX_NAME"),  # Optional, defaults to "parlant-pineconedb"
            embedder_factory=EmbedderFactory(container),
            embedding_cache_provider=lambda: container[EmbeddingCache],
        )
    )
    
    # Configure stores using vector database
    container[GlossaryStore] = await exit_stack.enter_async_context(
        GlossaryVectorStore(
            id_generator=container[IdGenerator],
            vector_db=pinecone_db,
            document_db=TransientDocumentDatabase(),
            embedder_factory=embedder_factory,
            embedder_type_provider=get_embedder_type,
        )  # type: ignore
    )
    
    container[CannedResponseStore] = await exit_stack.enter_async_context(
        CannedResponseVectorStore(
            id_generator=container[IdGenerator],
            vector_db=pinecone_db,
            document_db=TransientDocumentDatabase(),
            embedder_factory=embedder_factory,
            embedder_type_provider=get_embedder_type,
        )  # type: ignore
    )
    
    container[CapabilityStore] = await exit_stack.enter_async_context(
        CapabilityVectorStore(
            id_generator=container[IdGenerator],
            vector_db=pinecone_db,
            document_db=TransientDocumentDatabase(),
            embedder_factory=embedder_factory,
            embedder_type_provider=get_embedder_type,
        )  # type: ignore
    )
    
    container[JourneyStore] = await exit_stack.enter_async_context(
        JourneyVectorStore(
            id_generator=container[IdGenerator],
            vector_db=pinecone_db,
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
            description="Agent using Pinecone for persistent storage",
        )
        
        # Test: Create a term to verify Pinecone is working
        term = await agent.create_term(
            name="Example Term",
            description="This is stored in Pinecone",
        )
        print(f"Created term: {term.name}")
        # All vector operations now use Pinecone
```

## Environment Variables

You can set the Pinecone API key and index name via environment variables:

```bash
export PINECONE_API_KEY="your-api-key-here"
export PINECONE_INDEX_NAME="parlant-pineconedb"  # Optional, defaults to "parlant-pineconedb"
```

Or pass them directly when creating the database:

```python
pinecone_db = PineconeDatabase(
    logger=container[Logger],
    api_key="your-api-key-here",
    index_name="my-custom-index-name",  # Optional
    embedder_factory=EmbedderFactory(container),
    embedding_cache_provider=lambda: container[EmbeddingCache],
)
```

### Index Management

- **Single Index**: All collections are stored in a single Pinecone index
- **Auto-Creation**: The index is automatically created if it doesn't exist
- **Existing Indexes**: If the index already exists, it will be reused (no new index created)
- **Custom Names**: You can specify a custom index name via `PINECONE_INDEX_NAME` environment variable or `index_name` parameter

## Verification

To verify Pinecone integration is working correctly:

### Check Index
**Pinecone Dashboard:** A single index appears in your Pinecone dashboard:
- Default name: `parlant-pineconedb` (or custom name from `PINECONE_INDEX_NAME` environment variable)
- All collections are stored within this single index using metadata filtering
- Collections are differentiated by metadata fields: `_parlant_collection`, `_parlant_type`, `_parlant_embedder`

### Confirm No Transient Storage
When Pinecone is properly configured:
- **No vector files** are created in the `parlant-data` folder
- Vector data is stored only in Pinecone cloud
- Data persists across server restarts

### Test Vector Search
Create terms and test persistence:
```python
term = await agent.create_term(
    name="Test Term",
    description="This should be stored in Pinecone",
)
# Then chat with agent about "test term" - it should understand via vector search

# Test persistence: close the server and run again
# The term should still be available after restart
```

---

## Common Issues

### Integration Not Working (Still Using Transient Storage)
**Symptoms:**
- No indexes appear in Pinecone dashboard
- Vector data appears in `parlant-data` folder
- Data lost on server restart

**Solution:** Ensure all vector stores are properly configured with Pinecone in your `configure_container` function. Make sure you're using `AsyncExitStack` to properly manage the Pinecone database and vector stores lifecycle.

### API Key Issues
**Symptoms:**
- Authentication errors
- Connection failures

**Solution:** 
- Verify your API key is correct
- Check that the API key has proper permissions
- Ensure the API key is set in environment variable or passed directly

### Index Creation Failures
**Symptoms:**
- Errors when creating index
- Index already exists errors

**Solution:**
- The Pinecone index is created automatically if it doesn't exist
- If the index already exists, it will be reused (no new index created)
- Default index name is `parlant-pineconedb` (configurable via `PINECONE_INDEX_NAME`)
- To start fresh, delete the index from Pinecone dashboard
- All collections are stored in a single index using metadata filtering

### Performance
- Use embedding cache for better performance
- Pinecone serverless indexes have automatic scaling
- Consider index configuration for your use case

---

## Troubleshooting

### Connection Issues
- **API Key**: Verify API key is correct and has proper permissions
- **Network**: Check internet connectivity
- **Region**: Ensure you're using the correct Pinecone region

### Slow Performance  
- Use embedding cache
- Consider using Pinecone's pod-based indexes for higher performance
- Optimize batch operations

### Data Not Persisting
- Check API key is correct
- Verify connection to Pinecone is successful
- Test by closing the server and restarting—data should persist
- Check Pinecone dashboard to verify indexes exist

---

## Requirements

- Python 3.8+
- `pip install parlant[pinecone]`
- Pinecone account and API key

## Key Features

- **Cloud storage**: Fully managed vector database in the cloud
- **Auto-sync**: Collections automatically sync when embedders or schemas change
- **Scalable**: Pinecone handles scaling automatically
- **Production-ready**: Built for production workloads with high availability

