# pgvector Vector Database

The pgvector adapter provides persistent vector storage for Parlant using
PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension.
This replaces the default in-memory vector storage with production-ready
persistence, keeping everything inside a single PostgreSQL instance.

If you're already using the [PostgreSQL persistence adapter](../persistence/postgresql.md)
for sessions and customers, pgvector lets you consolidate vector storage into the
same database — no separate vector database service required.

## Prerequisites

1. **Install the optional dependency:**

   ```bash
   pip install "parlant[postgres]"
   ```

   This pulls in `asyncpg` and `pgvector`.

2. **PostgreSQL 14+ with the pgvector extension.**

   The easiest way is Docker with the official pgvector image:

   ```bash
   docker run -d --name parlant-postgres \
     -e POSTGRES_USER=parlant \
     -e POSTGRES_PASSWORD=parlant \
     -e POSTGRES_DB=parlant \
     -p 5432:5432 \
     pgvector/pgvector:pg16
   ```

   If you installed PostgreSQL another way, enable the extension manually:

   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

3. **Set the connection string** as an environment variable or pass it directly
   in code:

   | Variable                    | Required | Description                        |
   |-----------------------------|:--------:|------------------------------------|
   | `POSTGRES_CONNECTION_STRING`|    ✅     | Standard PostgreSQL connection URI |

## SDK Setup

Use the `configure_container` callback to replace the default vector stores with
pgvector-backed implementations:

```python
import parlant.sdk as p
from contextlib import AsyncExitStack
from parlant.adapters.vector_db.pgvector import PostgresVectorDatabase
from parlant.core.nlp.embedding import EmbedderFactory, EmbeddingCache
from parlant.core.nlp.service import NLPService
from parlant.core.glossary import GlossaryVectorStore, GlossaryStore
from parlant.core.canned_responses import CannedResponseVectorStore, CannedResponseStore
from parlant.core.capabilities import CapabilityVectorStore, CapabilityStore
from parlant.core.journeys import JourneyVectorStore, JourneyStore
from parlant.adapters.db.transient import TransientDocumentDatabase

exit_stack = AsyncExitStack()


async def configure_container(container: p.Container) -> p.Container:
    embedder_factory = EmbedderFactory(container)

    async def get_embedder_type() -> type:
        return type(await container[NLPService].get_embedder())

    vector_db = await exit_stack.enter_async_context(
        PostgresVectorDatabase(
            connection_string="postgresql://parlant:parlant@localhost:5432/parlant",
            logger=container[p.Logger],
            tracer=container[p.Tracer],
            embedder_factory=embedder_factory,
            embedding_cache_provider=lambda: container[EmbeddingCache],
        )
    )

    for store_interface, store_type in [
        (GlossaryStore, GlossaryVectorStore),
        (CannedResponseStore, CannedResponseVectorStore),
        (JourneyStore, JourneyVectorStore),
        (CapabilityStore, CapabilityVectorStore),
    ]:
        container[store_interface] = await exit_stack.enter_async_context(
            store_type(
                id_generator=container[p.IdGenerator],
                vector_db=vector_db,
                document_db=TransientDocumentDatabase(),
                embedder_factory=embedder_factory,
                embedder_type_provider=get_embedder_type,
            )
        )

    return container


async def main() -> None:
    try:
        async with p.Server(
            nlp_service=p.NLPServices.openai,
            configure_container=configure_container,
        ):
            pass
    finally:
        await exit_stack.aclose()
```

> **Tip:** For a complete example that combines PostgreSQL persistence *and*
> pgvector in a single setup, see
> [`examples/run_postgres_server.py`](../../../examples/run_postgres_server.py).

## How It Works

### Dual-table strategy

For each vector collection (e.g. `glossary`), pgvector maintains two tables:

| Table                            | Purpose                                  |
|----------------------------------|------------------------------------------|
| `{collection}_unembedded`        | Source of truth — stores content and metadata |
| `{collection}_{embedder_type}`   | Embedded copy — adds a `vector` column with the embedding |

When documents are inserted or updated, they go into the unembedded table first.
On collection load, checksums are compared and only changed documents are
re-embedded. This avoids unnecessary embedding API calls on restart.

### Vector search

Similarity queries use cosine distance via pgvector's `<=>` operator with an
HNSW index for fast approximate nearest-neighbor search:

```sql
SELECT data, (embedding <=> $1) AS distance
FROM {collection}_{embedder}
WHERE ...
ORDER BY distance ASC
LIMIT $2
```

### Dimension limit

pgvector's HNSW index supports up to 2000 dimensions. If your embedder produces
larger vectors (e.g. OpenAI `text-embedding-3-large` at 3072 dimensions), the
adapter skips index creation and falls back to sequential scans. Vector search
still works, but won't benefit from approximate nearest-neighbor acceleration.
For better performance, consider using an embedding model with 2000 or fewer
dimensions (e.g. `text-embedding-3-small` at 1536).

## What Gets Stored?

Once pgvector is configured, it becomes the source of truth for:

- **Glossary terms** — semantic search during conversations
- **Canned responses** — pre-written response matching
- **Journeys** — multi-step conversation flow matching
- **Capabilities** — agent capability matching

## Verification

After starting the server, connect to your PostgreSQL database and check that
tables were created:

```sql
\dt
```

You should see tables like:
- `glossary_unembedded`
- `glossary_openaiopenaitextembedding3large` (name varies by embedder)
- `canned_responses_unembedded`
- `_parlant_vector_metadata`

To confirm vector search is working, create a glossary term via the API and then
chat with the agent about that topic — it should be retrieved via semantic
similarity.

## Common Issues

### `extension "vector" does not exist`
The pgvector extension isn't installed. If using Docker, make sure you're using
the `pgvector/pgvector` image. Otherwise, install the extension from
[pgvector's install guide](https://github.com/pgvector/pgvector#installation)
and run `CREATE EXTENSION vector;`.

### Data not persisting across restarts
Make sure you're passing a real connection string to `PostgresVectorDatabase`,
not using `TransientDocumentDatabase` for the vector database itself.

### Slow first load after restart
On startup, the adapter compares checksums and re-embeds any documents that
changed. For large collections this involves embedding API calls and may take
a moment. Subsequent operations use the cached embeddings.

### Embedder changes
When you switch to a different embedding model, the adapter creates a new
embedded table for the new embedder type. Old embedded tables persist in the
database and can be dropped manually if no longer needed.
