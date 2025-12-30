# Elasticsearch Document Database

The Elasticsearch Document Database adapter provides a production-ready implementation of the `DocumentDatabase` interface using Elasticsearch 8 as the backend storage. This adapter is designed for document storage (non-vector) with full CRUD operations, complex filtering, and schema migration support.

For general Parlant usage, see the [official documentation](https://www.parlant.io/docs/).

## Features

### Core Functionality

- **Full CRUD Operations**: Create, read, update, and delete documents with atomic operations
- **Complex Query Filtering**: Support for comparison operators (`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`) and logical operators (`$and`, `$or`)
- **Schema Migration**: Automatic document transformation during collection loading with document loaders
- **Failed Migration Tracking**: Documents that fail migration are stored in separate indices for inspection
- **Environment-Based Configuration**: All connection and index settings configurable via environment variables

### Performance Optimizations

- **Best Compression**: Uses `best_compression` codec by default
- **HTTP Compression**: Enabled for reduced network overhead
- **Configurable Refresh Policy**: Default `wait_for` ensures data consistency
- **Connection Pooling**: 10 connections per node for optimal throughput
- **Translog Durability**: Set to `request` for safer document storage

## Architecture

### Document Flow

```
Application → ElasticsearchDocumentDatabase → ElasticsearchDocumentCollection → Elasticsearch Index
                     ↓
            Migration & Versioning
                     ↓
          Failed Migration Storage
```

### Key Components

#### `ElasticsearchDocumentDatabase`

Main database class that manages:

- Elasticsearch client lifecycle
- Collection (index) creation and management
- Document migration orchestration
- Failed migration tracking

#### `ElasticsearchDocumentCollection`

Collection class that provides:

- Document CRUD operations
- Query translation (Where filters → Elasticsearch DSL)
- ID extraction and validation
- Error handling and logging

## Configuration

### Environment Variables

#### Connection Settings

```bash
ELASTICSEARCH_DOC__HOST=localhost          # Elasticsearch host
ELASTICSEARCH_DOC__PORT=9200               # Elasticsearch port
ELASTICSEARCH_DOC__USERNAME=elastic        # Username (optional)
ELASTICSEARCH_DOC__PASSWORD=password       # Password (optional)
ELASTICSEARCH_DOC__USE_SSL=false           # Use SSL connection
ELASTICSEARCH_DOC__VERIFY_CERTS=false      # Verify SSL certificates
ELASTICSEARCH_DOC__TIMEOUT=30              # Request timeout (seconds)
ELASTICSEARCH_DOC__MAX_RETRIES=3           # Maximum retry attempts
ELASTICSEARCH_DOC__RETRY_ON_TIMEOUT=true   # Retry on timeout
```

#### Index Settings

```bash
ELASTICSEARCH_DOC__NUMBER_OF_SHARDS=1      # Primary shards
ELASTICSEARCH_DOC__NUMBER_OF_REPLICAS=0    # Replica shards
ELASTICSEARCH_DOC__REFRESH_INTERVAL=1s     # Index refresh interval
ELASTICSEARCH_DOC__CODEC=best_compression  # Compression codec
```

## Usage Examples

### Basic Usage

```python
from parlant.adapters.db.elasticsearch import (
    ElasticsearchDocumentDatabase,
    create_elasticsearch_document_client_from_env,
)

# Create client from environment variables
es_client = create_elasticsearch_document_client_from_env()

# Create database
async with ElasticsearchDocumentDatabase(
    elasticsearch_client=es_client,
    index_prefix="myapp",
    logger=logger,
) as db:
    # Create or get collection
    collection = await db.get_or_create_collection(
        name="users",
        schema=UserDocument,
        document_loader=identity_loader,
    )

    # Insert document
    await collection.insert_one({
        "id": "user-123",
        "version": "1.0.0",
        "name": "John Doe",
        "email": "john@example.com",
    })

    # Find documents
    users = await collection.find({"name": {"$eq": "John Doe"}})

    # Update document
    await collection.update_one(
        filters={"id": {"$eq": "user-123"}},
        params={"email": "newemail@example.com"},
    )

    # Delete document
    await collection.delete_one({"id": {"$eq": "user-123"}})
```

### Schema Migration

```python
async def document_loader(doc: BaseDocument) -> Optional[UserDocumentV2]:
    """Migrate documents from v1.0.0 to v2.0.0"""
    if doc["version"] == "1.0.0":
        # Transform old schema to new schema
        return UserDocumentV2(
            id=doc["id"],
            version="2.0.0",
            full_name=doc["name"],  # renamed field
            email=doc["email"],
            created_at=datetime.now(timezone.utc),  # new field
        )
    elif doc["version"] == "2.0.0":
        return cast(UserDocumentV2, doc)
    return None  # Migration failed

# Load collection with migration
collection = await db.get_collection(
    name="users",
    schema=UserDocumentV2,
    document_loader=document_loader,
)
```

### Complex Queries

```python
# Compound filters with logical operators
users = await collection.find({
    "$and": [
        {"status": {"$eq": "active"}},
        {"age": {"$gte": 18}},
        {
            "$or": [
                {"role": {"$in": ["admin", "moderator"]}},
                {"premium": {"$eq": True}},
            ]
        }
    ]
})

# Range queries
recent_users = await collection.find({
    "created_at": {
        "$gte": "2024-01-01",
        "$lt": "2024-12-31",
    }
})

# Exclusion filters
non_admin_users = await collection.find({
    "role": {"$ne": "admin"}
})
```

## Testing

The implementation includes comprehensive tests covering:

- **Agent CRUD**: Creating and retrieving agents
- **Session Management**: Sessions and events
- **Guidelines**: CRUD operations for guidelines
- **Customers**: Customer management
- **Context Variables**: Variable creation and value management
- **Evaluations**: Creating and updating evaluations
- **Migrations**: Document schema upgrades and failed migrations
- **Delete Operations**: Document and collection deletion

### Running Tests

```bash
# Run all Elasticsearch document database tests
pytest tests/adapters/db/test_elasticsearch_db.py -v

# Run specific test
pytest tests/adapters/db/test_elasticsearch_db.py::test_agent_creation -v

# Run with coverage
pytest tests/adapters/db/test_elasticsearch_db.py --cov=parlant.adapters.db.elasticsearch
```

## Comparison with Other Adapters

| Feature          | Elasticsearch | MongoDB    | JSON File  | Transient |
| ---------------- | ------------- | ---------- | ---------- | --------- |
| Persistence      | ✅ Yes        | ✅ Yes     | ✅ Yes     | ❌ No     |
| Distributed      | ✅ Yes        | ✅ Yes     | ❌ No      | ❌ No     |
| Schema Migration | ✅ Yes        | ✅ Yes     | ✅ Yes     | ❌ No     |
| Full-text Search | ⚡ Native     | ⚠️ Limited | ❌ No      | ❌ No     |
| Complex Queries  | ✅ Yes        | ✅ Yes     | ⚠️ Limited | ✅ Yes    |
| Production Ready | ✅ Yes        | ✅ Yes     | ⚠️ Limited | ❌ No     |

## Best Practices

1. **Index Naming**: Use meaningful index prefixes to namespace your data
2. **Refresh Policy**: Use `wait_for` in production for data consistency
3. **Error Handling**: Always check operation results and handle failures
4. **Migration Strategy**: Test document loaders thoroughly before deployment
5. **Failed Migrations**: Monitor and review failed migration indices regularly
6. **Connection Pooling**: Reuse the Elasticsearch client across database instances
7. **Index Settings**: Tune shard count based on your data volume and query patterns

## Troubleshooting

### Connection Issues

- Verify Elasticsearch is running: `curl http://localhost:9200`
- Check credentials and SSL settings
- Ensure firewall allows connections on port 9200

### Migration Failures

- Check `{index_prefix}_{collection}_failed_migrations` indices
- Review document loader logic and error messages
- Validate document schemas match expected types

### Performance Issues

- Increase `NUMBER_OF_SHARDS` for large datasets
- Adjust `REFRESH_INTERVAL` for write-heavy workloads
- Enable replicas for read-heavy workloads
- Monitor Elasticsearch cluster health and resource usage

## References

- [Elasticsearch Python Client Documentation](https://elasticsearch-py.readthedocs.io/)
- [Elasticsearch 8.x Reference](https://www.elastic.co/guide/en/elasticsearch/reference/8.x/index.html)
