# Elasticsearch Vector Database

The Elasticsearch adapter provides persistent vector storage for Parlant using Elasticsearch's dense vector capabilities. This replaces the default in-memory storage with production-ready persistence.

For general Parlant usage, see the [official documentation](https://www.parlant.io/docs/).

## Prerequisites

1. **Install Elasticsearch package**: `uv add elasticsearch`
2. **Running Elasticsearch cluster**: Local or cloud-hosted

## Quick Start

### Basic Usage (SDK)

The simplest way to use Elasticsearch for vector storage is via the SDK's `vector_store` parameter:

```python
import parlant.sdk as p

async def main():
    async with p.Server(
        vector_store="elasticsearch",  # Enable Elasticsearch vector storage
    ) as server:
        agent = await server.create_agent(
            name="My Agent",
            description="Agent using Elasticsearch for persistent vector storage",
        )

        # All vector operations now use Elasticsearch
        term = await agent.create_term(
            name="Example Term",
            description="This is stored in Elasticsearch",
        )
        print(f"Created term: {term.name}")
```

Make sure to set the required environment variables (see [Environment Variables](#environment-variables) below).

### Advanced Setup (Using configure_container)

For more control over the Elasticsearch configuration, you can use `configure_container`:

```python
import parlant.sdk as p
from contextlib import AsyncExitStack
from parlant.adapters.vector_db.elasticsearch import (
    ElasticsearchVectorDatabase,
    create_elasticsearch_client_from_env,
    get_elasticsearch_index_prefix_from_env,
)
from parlant.core.nlp.embedding import EmbedderFactory, EmbeddingCache, Embedder
from parlant.core.loggers import Logger
from parlant.core.nlp.service import NLPService
from parlant.core.glossary import GlossaryVectorStore, GlossaryStore
from parlant.core.canned_responses import CannedResponseVectorStore, CannedResponseStore
from parlant.core.capabilities import CapabilityVectorStore, CapabilityStore
from parlant.core.journeys import JourneyVectorStore, JourneyStore
from parlant.adapters.db.transient import TransientDocumentDatabase

EXIT_STACK = AsyncExitStack()


async def configure_container(container: p.Container) -> p.Container:
    embedder_factory = EmbedderFactory(container)

    async def get_embedder_type() -> type[Embedder]:
        return type(await container[NLPService].get_embedder())

    # Create Elasticsearch client from environment variables
    es_client = create_elasticsearch_client_from_env()
    index_prefix = get_elasticsearch_index_prefix_from_env()

    es_vector_db = await EXIT_STACK.enter_async_context(
        ElasticsearchVectorDatabase(
            elasticsearch_client=es_client,
            index_prefix=index_prefix,
            logger=container[Logger],
            embedder_factory=embedder_factory,
            embedding_cache_provider=lambda: container[EmbeddingCache],
        )
    )

    # Configure stores using vector database
    container[GlossaryStore] = await EXIT_STACK.enter_async_context(
        GlossaryVectorStore(
            id_generator=container[p.IdGenerator],
            vector_db=es_vector_db,
            document_db=TransientDocumentDatabase(),
            embedder_factory=embedder_factory,
            embedder_type_provider=get_embedder_type,
        )  # type: ignore
    )

    container[CannedResponseStore] = await EXIT_STACK.enter_async_context(
        CannedResponseVectorStore(
            id_generator=container[p.IdGenerator],
            vector_db=es_vector_db,
            document_db=TransientDocumentDatabase(),
            embedder_factory=embedder_factory,
            embedder_type_provider=get_embedder_type,
        )  # type: ignore
    )

    container[CapabilityStore] = await EXIT_STACK.enter_async_context(
        CapabilityVectorStore(
            id_generator=container[p.IdGenerator],
            vector_db=es_vector_db,
            document_db=TransientDocumentDatabase(),
            embedder_factory=embedder_factory,
            embedder_type_provider=get_embedder_type,
        )  # type: ignore
    )

    container[JourneyStore] = await EXIT_STACK.enter_async_context(
        JourneyVectorStore(
            id_generator=container[p.IdGenerator],
            vector_db=es_vector_db,
            document_db=TransientDocumentDatabase(),
            embedder_factory=embedder_factory,
            embedder_type_provider=get_embedder_type,
        )  # type: ignore
    )

    return container


async def shutdown_elasticsearch() -> None:
    await EXIT_STACK.aclose()


async def main():
    try:
        async with p.Server(configure_container=configure_container) as server:
            agent = await server.create_agent(
                name="My Agent",
                description="Agent using Elasticsearch for persistent vector storage",
            )

            # Test: Create a term to verify Elasticsearch is working
            term = await agent.create_term(
                name="Example Term",
                description="This is stored in Elasticsearch",
            )
            print(f"Created term: {term.name}")
            # All vector operations now use Elasticsearch
    finally:
        await shutdown_elasticsearch()
```

## Environment Variables

Parlant supports the following environment variables for Elasticsearch configuration:

### Connection Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ELASTICSEARCH__HOST` | Elasticsearch server hostname | `localhost` | No |
| `ELASTICSEARCH__PORT` | Elasticsearch server port | `9200` | No |
| `ELASTICSEARCH__USERNAME` | Username for authentication | None | No |
| `ELASTICSEARCH__PASSWORD` | Password for authentication | None | No |

### SSL/TLS Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ELASTICSEARCH__USE_SSL` | Use SSL/TLS connection | `false` | No |
| `ELASTICSEARCH__VERIFY_CERTS` | Verify SSL certificates | `false` | No |

### Connection and Retry Settings

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ELASTICSEARCH__TIMEOUT` | Request timeout in seconds | `30` | No |
| `ELASTICSEARCH__MAX_RETRIES` | Maximum number of retries | `3` | No |
| `ELASTICSEARCH__RETRY_ON_TIMEOUT` | Retry on timeout | `true` | No |

### Index Configuration

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ELASTICSEARCH__INDEX_PREFIX` | Prefix for all Elasticsearch indices | `parlant` | No |
| `ELASTICSEARCH__NUMBER_OF_SHARDS` | Number of primary shards per index | `1` | No |
| `ELASTICSEARCH__NUMBER_OF_REPLICAS` | Number of replica shards per index | `0` | No |
| `ELASTICSEARCH__REFRESH_INTERVAL` | How often to refresh the index | `5s` | No |
| `ELASTICSEARCH__CODEC` | Compression codec for storage | `best_compression` | No |
| `ELASTICSEARCH__ENABLE_QUERY_CACHE` | Enable query result caching | `true` | No |

## Verification

To verify Elasticsearch vector integration is working correctly:

### Check Indices

Indices appear in your Elasticsearch cluster with names like:
- `{prefix}_glossary_OpenAITextEmbedding3Large`
- `{prefix}_glossary_unembedded`
- `{prefix}_capabilities_OpenAITextEmbedding3Large`
- `{prefix}_canned_responses_OpenAITextEmbedding3Large`

### Confirm No Transient Storage

When Elasticsearch is properly configured:
- **No vector files** are created in the `parlant-data` folder
- Vector data is stored only in Elasticsearch
- Data persists across server restarts

### Test Vector Search

Create terms and test persistence:
```python
term = await agent.create_term(
    name="Test Term",
    description="This should be stored in Elasticsearch",
)
# Then chat with agent about "test term" - it should understand via vector search

# Test persistence: close the server and run again
# The term should still be available after restart
```

## Advanced Usage

### Direct Database Access

For advanced use cases, you can create the vector database directly:

```python
from parlant.adapters.vector_db.elasticsearch import (
    create_elasticsearch_client_from_env,
    get_elasticsearch_index_prefix_from_env,
    ElasticsearchVectorDatabase,
)
from parlant.core.nlp.embedding import EmbedderFactory, NullEmbeddingCache
from parlant.core.loggers import Logger

# Create client from environment variables
es_client = create_elasticsearch_client_from_env()
index_prefix = get_elasticsearch_index_prefix_from_env()

# Create the vector database
vector_db = ElasticsearchVectorDatabase(
    elasticsearch_client=es_client,
    index_prefix=index_prefix,
    logger=logger,
    embedder_factory=embedder_factory,
    embedding_cache_provider=NullEmbeddingCache,
)
```

### Manual Client Configuration

If you prefer to configure the client manually:

```python
from elasticsearch import AsyncElasticsearch
from parlant.adapters.vector_db.elasticsearch import ElasticsearchVectorDatabase

# Create client with custom configuration
es_client = AsyncElasticsearch(
    hosts=["https://user:password@es.example.com:9200"],
    request_timeout=60,
    max_retries=5,
    retry_on_timeout=True,
    verify_certs=True,
)

vector_db = ElasticsearchVectorDatabase(
    elasticsearch_client=es_client,
    index_prefix="my_custom_prefix",
    logger=logger,
    embedder_factory=embedder_factory,
    embedding_cache_provider=NullEmbeddingCache,
)
```

## Example Configurations

### Local Development (No Authentication)

```bash
export ELASTICSEARCH__HOST=localhost
export ELASTICSEARCH__PORT=9200
export ELASTICSEARCH__USE_SSL=false
export ELASTICSEARCH__INDEX_PREFIX=parlant_dev
```

### Production (With Authentication and SSL)

```bash
export ELASTICSEARCH__HOST=es.example.com
export ELASTICSEARCH__PORT=9200
export ELASTICSEARCH__USERNAME=parlant_user
export ELASTICSEARCH__PASSWORD=SecurePassword123!
export ELASTICSEARCH__USE_SSL=true
export ELASTICSEARCH__VERIFY_CERTS=true
export ELASTICSEARCH__TIMEOUT=60
export ELASTICSEARCH__INDEX_PREFIX=parlant_prod
# Production index settings for high availability
export ELASTICSEARCH__NUMBER_OF_SHARDS=3
export ELASTICSEARCH__NUMBER_OF_REPLICAS=2
export ELASTICSEARCH__REFRESH_INTERVAL=30s
export ELASTICSEARCH__CODEC=best_compression
export ELASTICSEARCH__ENABLE_QUERY_CACHE=true
```

### Docker/Podman Container

```bash
export ELASTICSEARCH__HOST=elasticsearch
export ELASTICSEARCH__PORT=9200
export ELASTICSEARCH__USERNAME=elastic
export ELASTICSEARCH__PASSWORD=ChangeThisPassword123!
export ELASTICSEARCH__USE_SSL=false
export ELASTICSEARCH__VERIFY_CERTS=false
export ELASTICSEARCH__INDEX_PREFIX=parlant
```

### Elasticsearch 8 with Security Enabled

```bash
export ELASTICSEARCH__HOST=localhost
export ELASTICSEARCH__PORT=9200
export ELASTICSEARCH__USERNAME=elastic
export ELASTICSEARCH__PASSWORD=YourElasticPassword
export ELASTICSEARCH__USE_SSL=false
export ELASTICSEARCH__VERIFY_CERTS=false
export ELASTICSEARCH__TIMEOUT=30
export ELASTICSEARCH__MAX_RETRIES=3
export ELASTICSEARCH__RETRY_ON_TIMEOUT=true
```

## Testing

When running tests, the environment variables will be automatically used. If no environment variables are set, sensible defaults will be used:

```bash
# Run tests with default configuration (localhost:9200, no auth)
pytest tests/adapters/db/test_elasticsearch.py

# Run tests with custom Elasticsearch instance
export ELASTICSEARCH__HOST=custom-host
export ELASTICSEARCH__PORT=9200
export ELASTICSEARCH__USERNAME=testuser
export ELASTICSEARCH__PASSWORD=testpass
pytest tests/adapters/db/test_elasticsearch.py
```

## Compatibility

- **Elasticsearch Version**: 8.x (recommended)
- **Python Client**: `elasticsearch` package version 8.x
- The adapter uses Elasticsearch's `dense_vector` field type for vector storage
- Requires Elasticsearch to be running with appropriate permissions for creating indices

## Index Settings Tuning

### Number of Shards

- **Development**: Use `1` shard for simplicity
- **Production**: Use 3-5 shards for larger datasets (depends on data size and cluster capacity)
- More shards = better parallelization but higher overhead
- Rule of thumb: Keep shard size between 10GB-50GB

### Number of Replicas

- **Development**: Use `0` replicas (single-node)
- **Production**: Use `1-2` replicas for high availability
- Each replica increases storage requirements and indexing overhead
- More replicas = better read performance and fault tolerance

### Refresh Interval

- **Development**: `5s` (default Elasticsearch behavior)
- **Production (write-heavy)**: `30s` or higher to reduce indexing overhead
- **Production (search-heavy)**: `5s` or `1s` for near real-time search
- Higher values = better indexing performance, lower search recency

### Codec

- **`best_compression`**: Better for storage efficiency (default in this adapter)
- **`default`**: Faster indexing and search, but larger storage
- Use `best_compression` for vector databases as they can be storage-intensive

### Query Cache

- **Enable (`true`)**: Improves performance for repeated queries (recommended)
- **Disable (`false`)**: Reduces memory usage if queries are always unique
- Most beneficial for filters and aggregations that are frequently reused

## Security Notes

1. **Never commit credentials**: Always use environment variables for credentials
2. **SSL/TLS in production**: Always enable SSL/TLS (`ELASTICSEARCH__USE_SSL=true`) in production
3. **Certificate verification**: Enable certificate verification (`ELASTICSEARCH__VERIFY_CERTS=true`) in production
4. **Least privilege**: Use a dedicated Elasticsearch user with minimal required permissions

## Troubleshooting

### Connection Issues

If you're having trouble connecting to Elasticsearch:

1. Verify Elasticsearch is running: `curl http://localhost:9200`
2. Check credentials are correct
3. Verify network connectivity and firewall rules
4. Check SSL/TLS settings match your Elasticsearch configuration

### Index Issues

If indices aren't being created or accessed:

1. Verify the user has permissions to create indices
2. Check the `ELASTICSEARCH__INDEX_PREFIX` doesn't conflict with existing indices
3. Review Elasticsearch logs for permission or mapping errors

### Performance Issues

If you're experiencing slow performance:

1. Adjust `ELASTICSEARCH__TIMEOUT` if requests are timing out
2. Increase `ELASTICSEARCH__MAX_RETRIES` for unreliable networks
3. Consider tuning Elasticsearch cluster settings (shards, replicas)
4. Review index settings in the code (codec, refresh_interval, etc.)

