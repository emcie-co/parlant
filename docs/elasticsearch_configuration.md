# Elasticsearch Configuration

This document describes how to configure Parlant to use Elasticsearch as a vector database.

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

## Usage

### Python Code

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

### Manual Configuration

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

