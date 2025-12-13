# Supabase Document Database

The Supabase adapter provides persistent document storage for Parlant using Supabase's PostgreSQL database. This replaces the default in-memory storage with production-ready persistence for sessions, customers, and context variables.

For general Parlant usage, see the [official documentation](https://www.parlant.io/docs/).

## Prerequisites

1. **Install Supabase adapter**: `pip install parlant[supabase]`
2. **Supabase project**: Create a project at [supabase.com](https://supabase.com) or use an existing one

## Quick Start

### Setup (Manual)

```python
import parlant.sdk as p
from contextlib import AsyncExitStack
from parlant.adapters.db.supabase_db import SupabaseDocumentDatabase
from parlant.core.emission.event_publisher import EventPublisherFactory

EXIT_STACK = AsyncExitStack()


async def configure_container(container: p.Container) -> p.Container:
    container = container.clone()
    
    # Create Supabase database instances for each store
    session_db = await EXIT_STACK.enter_async_context(
        SupabaseDocumentDatabase(
            logger=container[p.Logger],
            table_prefix="parlant_sessions_",
        )
    )
    
    customer_db = await EXIT_STACK.enter_async_context(
        SupabaseDocumentDatabase(
            logger=container[p.Logger],
            table_prefix="parlant_customers_",
        )
    )
    
    variable_db = await EXIT_STACK.enter_async_context(
        SupabaseDocumentDatabase(
            logger=container[p.Logger],
            table_prefix="parlant_context_variables_",
        )
    )
    
    # Configure stores using Supabase databases
    session_store = await EXIT_STACK.enter_async_context(
        p.SessionDocumentStore(database=session_db, allow_migration=True)
    )
    container[p.SessionDocumentStore] = session_store
    container[p.SessionStore] = session_store
    
    customer_store = await EXIT_STACK.enter_async_context(
        p.CustomerDocumentStore(
            id_generator=container[p.IdGenerator],
            database=customer_db,
            allow_migration=True,
        )
    )
    container[p.CustomerDocumentStore] = customer_store
    container[p.CustomerStore] = customer_store
    
    variable_store = await EXIT_STACK.enter_async_context(
        p.ContextVariableDocumentStore(
            id_generator=container[p.IdGenerator],
            database=variable_db,
            allow_migration=True,
        )
    )
    container[p.ContextVariableDocumentStore] = variable_store
    container[p.ContextVariableStore] = variable_store
    
    # Configure event emitter to use session store
    container[p.EventEmitterFactory] = EventPublisherFactory(
        container[p.AgentStore],
        session_store,
    )
    
    return container


async def main():
    async with p.Server(configure_container=configure_container) as server:
        agent = await server.create_agent(
            name="My Agent",
            description="Agent using Supabase for persistent storage",
        )
        
        # Test: Create a session to verify Supabase is working
        session = await agent.create_session(
            customer_id=p.CustomerId("test-customer"),
        )
        print(f"Created session: {session.id}")
        # All document operations now use Supabase
```

### Environment Variables

Set the following environment variables before running:

| Variable                | Required | Description                                                                         |
|------------------------|:--------:|-------------------------------------------------------------------------------------|
| `SUPABASE_URL`         |    ✅     | Your Supabase project URL (e.g. `https://your-project.supabase.co`).              |
| `SUPABASE_KEY`         |    ✅     | Your Supabase anon/service role key.                                               |
| `SUPABASE_DB_PASSWORD` |    ✅     | Database password for direct PostgreSQL connection (for table creation).            |
| `SUPABASE_DB_USER`     |    ➖     | Database user (defaults to `postgres`).                                            |
| `SUPABASE_DB_NAME`     |    ➖     | Database name (defaults to `postgres`).                                            |
| `SUPABASE_DB_PORT`     |    ➖     | Database port (defaults to `5432`).                                                |
| `SUPABASE_SCHEMA`      |    ➖     | PostgreSQL schema name (defaults to `public`).                                      |

> **Note**: `SUPABASE_DB_PASSWORD` is required for automatic table creation. You can find it in your Supabase project settings under Database → Connection string.

## Verification

To verify Supabase integration is working correctly:

### Check Tables

**Supabase Dashboard:** Tables appear in your Supabase dashboard with names like:
- `parlant_sessions_*`
- `parlant_customers_*`
- `parlant_context_variables_*`
- `parlant_*_failed_migrations` (for failed document migrations)

### Confirm No Transient Storage

When Supabase is properly configured:
- **No transient files** are created in the `parlant-data` folder for sessions, customers, or context variables
- Document data is stored only in Supabase PostgreSQL
- Data persists across server restarts

### Test Persistence

Create a session and test persistence:
```python
session = await agent.create_session(
    customer_id=p.CustomerId("test-customer"),
)
# Then close the server and run again
# The session should still be available after restart
```

---

## Common Issues

### Integration Not Working (Still Using Transient Storage)

**Symptoms:**
- No tables appear in Supabase dashboard
- Document data appears in `parlant-data` folder
- Data lost on server restart

**Solution:** Ensure all document stores are properly configured with Supabase in your `configure_container` function. Make sure you're using `AsyncExitStack` to properly manage the Supabase database and store lifecycle.

### Table Creation Fails

**Symptoms:**
- Error messages about table creation
- Tables don't appear in Supabase dashboard

**Solution:** 
1. Ensure `SUPABASE_DB_PASSWORD` is set correctly
2. Verify your database user has CREATE TABLE permissions
3. Check that the schema exists and is accessible
4. Verify network access allows connections to Supabase

### Connection Issues

**Symptoms:**
- Connection timeout errors
- Authentication failures

**Solution:**
1. Verify `SUPABASE_URL` and `SUPABASE_KEY` are correct
2. Check that your Supabase project is active
3. Ensure network access allows connections to Supabase
4. For production, use Supabase's IP allowlist

---

## Troubleshooting

### Query Performance

For large datasets:
1. Ensure indexes are created (check with `\d+ table_name` in psql)
2. Use appropriate filters to limit result sets
3. Use cursor-based pagination for large collections
4. The adapter automatically creates GIN indexes on JSONB columns for efficient JSON queries

### Migration Issues

If document migrations fail:
1. Failed documents are stored in separate `_failed_migrations` tables
2. Check the Supabase logs for migration errors
3. Review document versions and schema compatibility

### Data Not Persisting

If data doesn't persist:
1. Check that tables are created in Supabase dashboard
2. Verify connection settings are correct
3. Test by closing the server and restarting—data should persist
4. Check Supabase project status and quotas

---

## Requirements

- Python 3.10+
- `pip install parlant[supabase]`
- Supabase project with PostgreSQL database access
- `SUPABASE_URL`, `SUPABASE_KEY`, and `SUPABASE_DB_PASSWORD` environment variables

## Key Features

- **Persistent storage**: Replaces in-memory storage with production-ready PostgreSQL persistence
- **Automatic table creation**: Tables and indexes are created automatically on first access
- **JSONB storage**: Full document data stored in PostgreSQL JSONB columns with GIN indexes
- **Migration support**: Automatic document migration with failed migration tracking
- **Query support**: Full support for equality, comparison, membership, and logical operators
- **Cursor-based pagination**: Efficient pagination for large collections

## Advanced Configuration

### Custom Table Prefix

You can customize the table prefix when creating the database:

```python
database = SupabaseDocumentDatabase(
    logger=container[p.Logger],
    table_prefix="custom_prefix_",
)
```

### Manual Table Creation

If you prefer to create tables manually, you can use the following SQL:

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

### Query Operations

The adapter supports the following query operations:

- **Equality**: `{"field": {"$eq": "value"}}`
- **Comparison**: `{"field": {"$gt": 10, "$lt": 20}}`
- **Membership**: `{"field": {"$in": ["a", "b", "c"]}}`
- **Logical operators**: `{"$and": [...], "$or": [...]}`
- **Cursor-based pagination**: Automatic support for `limit` and `cursor`
- **Sorting**: Ascending and descending by `creation_utc` and `id`

## Security Best Practices

1. **Use Service Role Key Carefully**: The service role key bypasses Row Level Security (RLS)
2. **Enable RLS**: Consider enabling Row Level Security on tables for multi-tenant scenarios
3. **Connection String**: Store `SUPABASE_DB_PASSWORD` securely (use environment variables or secrets management)
4. **Network Security**: Use Supabase's IP allowlist for production deployments
5. **Key Rotation**: Regularly rotate your Supabase keys and update environment variables
