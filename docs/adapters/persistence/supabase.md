# Supabase Persistence Adapter

The Supabase document adapter lets Parlant persist the long–lived parts of a
deployment—sessions, customers, and context variables—inside your Supabase
PostgreSQL database. That means you can run the server, stop it, and later
resume the exact same conversation state with full persistence.

This page walks through the required environment variables and shows how to
wire the stores into Supabase when booting Parlant via the SDK.

## Requirements

1. Install the optional dependency (or otherwise provide `supabase-py` and `asyncpg`):

   ```bash
   pip install "parlant[supabase]"
   ```

   Or install manually:
   ```bash
   pip install supabase asyncpg
   ```

2. Set the credentials that `SupabaseDocumentDatabase` consumes:

   | Variable                     | Required | Description                                                                         |
   |-----------------------------|:--------:|-------------------------------------------------------------------------------------|
   | `SUPABASE_URL`              |    ✅     | Your Supabase project URL (e.g. `https://your-project.supabase.co`).              |
   | `SUPABASE_KEY`              |    ✅     | Your Supabase anon/service role key.                                               |
   | `SUPABASE_DB_PASSWORD`      |    ✅     | Database password for direct PostgreSQL connection (for table creation).            |
   | `SUPABASE_DB_USER`          |    ➖     | Database user (defaults to `postgres`).                                            |
   | `SUPABASE_DB_NAME`          |    ➖     | Database name (defaults to `postgres`).                                            |
   | `SUPABASE_DB_PORT`          |    ➖     | Database port (defaults to `5432`).                                                |
   | `SUPABASE_SCHEMA`           |    ➖     | PostgreSQL schema name (defaults to `public`).                                      |

   > **Note**: `SUPABASE_DB_PASSWORD` is required for automatic table creation. You can find it in your Supabase project settings under Database → Connection string.

## Automatic Table Creation

The Supabase adapter automatically creates tables when collections are first accessed. Tables are created with:

- **Primary key**: `id` (TEXT)
- **Indexed fields**: `id`, `version`, `creation_utc`, `session_id`, `customer_id`, `agent_id`
- **JSONB storage**: Full document data stored in `data` column
- **GIN index**: On the `data` JSONB column for efficient JSON queries
- **Failed migrations table**: Separate table for documents that fail migration

Tables are prefixed with `parlant_` by default (configurable via `table_prefix` parameter).

## SDK / Module Setup

Parlant's SDK exposes a `configure_container` hook that lets you replace the
default persistence layer. The pattern below shows how to register
Supabase-backed implementations of the three configurable stores:

- `SessionStore` → `SessionDocumentStore`
- `CustomerStore` → `CustomerDocumentStore`
- `ContextVariableStore` → `ContextVariableDocumentStore`

Each store receives its own table prefix (`parlant_sessions_`,
`parlant_customers_`, `parlant_context_variables_`) so their metadata never
collides. We also rebind `EventEmitterFactory`, so system events get written into
the same store.

```python
from contextlib import AsyncExitStack

import parlant.sdk as p
from parlant.adapters.db.supabase_db import SupabaseDocumentDatabase
from parlant.core.emission.event_publisher import EventPublisherFactory

EXIT_STACK = AsyncExitStack()


async def _make_session_store(container: p.Container) -> p.SessionStore:
    database = await EXIT_STACK.enter_async_context(
        SupabaseDocumentDatabase(
            logger=container[p.Logger],
            table_prefix="parlant_sessions_",
        )
    )
    store = p.SessionDocumentStore(database=database, allow_migration=True)
    return await EXIT_STACK.enter_async_context(store)


async def _make_customer_store(container: p.Container) -> p.CustomerStore:
    database = await EXIT_STACK.enter_async_context(
        SupabaseDocumentDatabase(
            logger=container[p.Logger],
            table_prefix="parlant_customers_",
        )
    )
    store = p.CustomerDocumentStore(
        id_generator=container[p.IdGenerator],
        database=database,
        allow_migration=True,
    )
    return await EXIT_STACK.enter_async_context(store)


async def _make_variable_store(container: p.Container) -> p.ContextVariableStore:
    database = await EXIT_STACK.enter_async_context(
        SupabaseDocumentDatabase(
            logger=container[p.Logger],
            table_prefix="parlant_context_variables_",
        )
    )
    store = p.ContextVariableDocumentStore(
        id_generator=container[p.IdGenerator],
        database=database,
        allow_migration=True,
    )
    return await EXIT_STACK.enter_async_context(store)


async def configure_container(container: p.Container) -> p.Container:
    container = container.clone()

    session_store = await _make_session_store(container)
    container[p.SessionDocumentStore] = session_store
    container[p.SessionStore] = session_store

    customer_store = await _make_customer_store(container)
    container[p.CustomerDocumentStore] = customer_store
    container[p.CustomerStore] = customer_store

    variable_store = await _make_variable_store(container)
    container[p.ContextVariableDocumentStore] = variable_store
    container[p.ContextVariableStore] = variable_store

    container[p.EventEmitterFactory] = EventPublisherFactory(
        container[p.AgentStore],
        session_store,
    )

    return container


async def shutdown_supabase() -> None:
    await EXIT_STACK.aclose()
```

### Using the SDK

```python
async def main() -> None:
    try:
        async with p.Server(
            nlp_service=p.NLPServices.snowflake,
            configure_container=configure_container,
        ) as server:
            ...
    finally:
        await shutdown_supabase()
```

## Manual Table Creation (Optional)

If you prefer to create tables manually or need custom configurations, you can
create them directly in Supabase:

```sql
CREATE TABLE IF NOT EXISTS parlant_sessions (
    id TEXT NOT NULL PRIMARY KEY,
    version TEXT,
    creation_utc TEXT,
    session_id TEXT,
    customer_id TEXT,
    agent_id TEXT,
    data JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_parlant_sessions_creation_utc 
ON parlant_sessions(creation_utc);

CREATE INDEX IF NOT EXISTS idx_parlant_sessions_session_id 
ON parlant_sessions(session_id) WHERE session_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_parlant_sessions_customer_id 
ON parlant_sessions(customer_id) WHERE customer_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_parlant_sessions_agent_id 
ON parlant_sessions(agent_id) WHERE agent_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_parlant_sessions_data_gin 
ON parlant_sessions USING GIN (data);
```

## Features

### Automatic Table Management

- Tables are created automatically on first access
- Indexes are created for optimal query performance
- Failed migration documents are stored separately

### JSONB Storage

- Full document data stored in PostgreSQL JSONB columns
- Efficient JSON queries using GIN indexes
- Support for nested document fields

### Query Support

The adapter supports the following query operations:

- **Equality**: `{"field": {"$eq": "value"}}`
- **Comparison**: `{"field": {"$gt": 10, "$lt": 20}}`
- **Membership**: `{"field": {"$in": ["a", "b", "c"]}}`
- **Logical operators**: `{"$and": [...], "$or": [...]}`
- **Cursor-based pagination**: Automatic support for `limit` and `cursor`
- **Sorting**: Ascending and descending by `creation_utc` and `id`

### Migration Support

- Automatic document migration on collection load
- Failed migrations stored in separate `_failed_migrations` tables
- Version tracking for schema evolution

## What Gets Persisted?

Once the Supabase stores are registered, Supabase becomes the source of truth for:

- Sessions + events + inspections
- Customers + their tag associations
- Context variables + their values

Other stores (agents, guidelines, journeys, etc.) continue to use their default
backends. If you define them in code at startup, they will automatically be
recreated each time the server runs. For dynamic authoring flows you can follow
the same module approach to route additional stores into Supabase.

## Performance Considerations

- **Indexes**: The adapter creates indexes on commonly queried fields automatically
- **JSONB GIN Index**: Enables fast JSON queries on document data
- **Connection Pooling**: Supabase handles connection pooling automatically
- **Query Optimization**: Uses Supabase PostgREST API for efficient queries

## Troubleshooting

### Table Creation Fails

If automatic table creation fails, ensure:

1. `SUPABASE_DB_PASSWORD` is set correctly
2. Your database user has CREATE TABLE permissions
3. The schema exists and is accessible

### Connection Issues

If you encounter connection issues:

1. Verify `SUPABASE_URL` and `SUPABASE_KEY` are correct
2. Check that your Supabase project is active
3. Ensure network access allows connections to Supabase

### Query Performance

For large datasets:

1. Ensure indexes are created (check with `\d+ table_name` in psql)
2. Use appropriate filters to limit result sets
3. Use cursor-based pagination for large collections

## Security Best Practices

1. **Use Service Role Key Carefully**: The service role key bypasses Row Level Security (RLS)
2. **Enable RLS**: Consider enabling Row Level Security on tables for multi-tenant scenarios
3. **Connection String**: Store `SUPABASE_DB_PASSWORD` securely (use environment variables or secrets management)
4. **Network Security**: Use Supabase's IP allowlist for production deployments

