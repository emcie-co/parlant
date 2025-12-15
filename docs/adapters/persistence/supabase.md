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
        
```

### Environment Variables

Set the following environment variables before running:

| Variable        | Required | Description                                                          |
|----------------|:--------:|----------------------------------------------------------------------|
| `SUPABASE_URL` |    ✅     | Your Supabase project URL (e.g. `https://your-project.supabase.co`). |
| `SUPABASE_KEY` |    ✅     | Your Supabase anon/service role key (publishable key).               |
| `SUPABASE_SCHEMA` |    ➖     | PostgreSQL schema name (defaults to `public`).                       |

> **Note**: Only `SUPABASE_URL` and `SUPABASE_KEY` are required. The adapter uses the Supabase REST API client, so no database password or direct PostgreSQL connection is needed.

### Manual Table Setup

**Before using the adapter, you must create the required tables manually in your Supabase database.**

#### Step 1: Access Supabase SQL Editor

1. Go to your Supabase Dashboard
2. Navigate to **SQL Editor** (left sidebar)
3. Click **New Query**

#### Step 2: Run SQL to Create Tables

Copy and paste the following SQL statements. Replace `parlant_sessions_`, `parlant_customers_`, and `parlant_context_variables_` with your actual table prefixes if you customized them:

```sql
-- ============================================
-- SESSIONS TABLES
-- ============================================

-- Create sessions table
CREATE TABLE IF NOT EXISTS parlant_sessions_ (
    id TEXT NOT NULL PRIMARY KEY,
    version TEXT,
    creation_utc TEXT,
    session_id TEXT,
    customer_id TEXT,
    agent_id TEXT,
    data JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Create indexes for sessions table
CREATE INDEX IF NOT EXISTS idx_parlant_sessions__creation_utc 
ON parlant_sessions_(creation_utc);

CREATE INDEX IF NOT EXISTS idx_parlant_sessions__session_id 
ON parlant_sessions_(session_id) WHERE session_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_parlant_sessions__customer_id 
ON parlant_sessions_(customer_id) WHERE customer_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_parlant_sessions__agent_id 
ON parlant_sessions_(agent_id) WHERE agent_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_parlant_sessions__data_gin 
ON parlant_sessions_ USING GIN (data);

-- Create failed migrations table for sessions
CREATE TABLE IF NOT EXISTS parlant_sessions__failed_migrations (
    id TEXT,
    data JSONB DEFAULT '{}'::jsonb
);

-- Create metadata table for sessions
CREATE TABLE IF NOT EXISTS parlant_sessions_metadata (
    id TEXT NOT NULL PRIMARY KEY,
    version TEXT,
    creation_utc TEXT,
    session_id TEXT,
    customer_id TEXT,
    agent_id TEXT,
    data JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_parlant_sessions_metadata_creation_utc 
ON parlant_sessions_metadata(creation_utc);

-- Create events table for sessions
CREATE TABLE IF NOT EXISTS parlant_sessions_events (
    id TEXT NOT NULL PRIMARY KEY,
    version TEXT,
    creation_utc TEXT,
    session_id TEXT,
    customer_id TEXT,
    agent_id TEXT,
    data JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_parlant_sessions_events_creation_utc 
ON parlant_sessions_events(creation_utc);

CREATE INDEX IF NOT EXISTS idx_parlant_sessions_events_session_id 
ON parlant_sessions_events(session_id) WHERE session_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_parlant_sessions_events_data_gin 
ON parlant_sessions_events USING GIN (data);

-- Create failed migrations table for events
CREATE TABLE IF NOT EXISTS parlant_sessions_events_failed_migrations (
    id TEXT,
    data JSONB DEFAULT '{}'::jsonb
);

-- ============================================
-- CUSTOMERS TABLES
-- ============================================

-- Create customers table
CREATE TABLE IF NOT EXISTS parlant_customers_ (
    id TEXT NOT NULL PRIMARY KEY,
    version TEXT,
    creation_utc TEXT,
    session_id TEXT,
    customer_id TEXT,
    agent_id TEXT,
    data JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Create indexes for customers table
CREATE INDEX IF NOT EXISTS idx_parlant_customers__creation_utc 
ON parlant_customers_(creation_utc);

CREATE INDEX IF NOT EXISTS idx_parlant_customers__session_id 
ON parlant_customers_(session_id) WHERE session_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_parlant_customers__customer_id 
ON parlant_customers_(customer_id) WHERE customer_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_parlant_customers__agent_id 
ON parlant_customers_(agent_id) WHERE agent_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_parlant_customers__data_gin 
ON parlant_customers_ USING GIN (data);

-- Create failed migrations table for customers
CREATE TABLE IF NOT EXISTS parlant_customers__failed_migrations (
    id TEXT,
    data JSONB DEFAULT '{}'::jsonb
);

-- Create metadata table for customers
CREATE TABLE IF NOT EXISTS parlant_customers_metadata (
    id TEXT NOT NULL PRIMARY KEY,
    version TEXT,
    creation_utc TEXT,
    session_id TEXT,
    customer_id TEXT,
    agent_id TEXT,
    data JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_parlant_customers_metadata_creation_utc 
ON parlant_customers_metadata(creation_utc);

-- Create customer tag associations table
CREATE TABLE IF NOT EXISTS parlant_customers_customer_tag_associations (
    id TEXT NOT NULL PRIMARY KEY,
    version TEXT,
    creation_utc TEXT,
    session_id TEXT,
    customer_id TEXT,
    agent_id TEXT,
    data JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_parlant_customers_customer_tag_associations_creation_utc 
ON parlant_customers_customer_tag_associations(creation_utc);

CREATE INDEX IF NOT EXISTS idx_parlant_customers_customer_tag_associations_data_gin 
ON parlant_customers_customer_tag_associations USING GIN (data);

-- Create failed migrations table for customer tag associations
CREATE TABLE IF NOT EXISTS parlant_customers_customer_tag_associations_failed_migrations (
    id TEXT,
    data JSONB DEFAULT '{}'::jsonb
);

-- ============================================
-- CONTEXT VARIABLES TABLES
-- ============================================

-- Create context variables table
CREATE TABLE IF NOT EXISTS parlant_context_variables_ (
    id TEXT NOT NULL PRIMARY KEY,
    version TEXT,
    creation_utc TEXT,
    session_id TEXT,
    customer_id TEXT,
    agent_id TEXT,
    data JSONB NOT NULL DEFAULT '{}'::jsonb
);

-- Create indexes for context variables table
CREATE INDEX IF NOT EXISTS idx_parlant_context_variables__creation_utc 
ON parlant_context_variables_(creation_utc);

CREATE INDEX IF NOT EXISTS idx_parlant_context_variables__session_id 
ON parlant_context_variables_(session_id) WHERE session_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_parlant_context_variables__customer_id 
ON parlant_context_variables_(customer_id) WHERE customer_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_parlant_context_variables__agent_id 
ON parlant_context_variables_(agent_id) WHERE agent_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_parlant_context_variables__data_gin 
ON parlant_context_variables_ USING GIN (data);

-- Create failed migrations table for context variables
CREATE TABLE IF NOT EXISTS parlant_context_variables__failed_migrations (
    id TEXT,
    data JSONB DEFAULT '{}'::jsonb
);

-- Create metadata table for context variables
CREATE TABLE IF NOT EXISTS parlant_context_variables_metadata (
    id TEXT NOT NULL PRIMARY KEY,
    version TEXT,
    creation_utc TEXT,
    session_id TEXT,
    customer_id TEXT,
    agent_id TEXT,
    data JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_parlant_context_variables_metadata_creation_utc 
ON parlant_context_variables_metadata(creation_utc);

-- Create variable tag associations table
CREATE TABLE IF NOT EXISTS parlant_context_variables_variable_tag_associations (
    id TEXT NOT NULL PRIMARY KEY,
    version TEXT,
    creation_utc TEXT,
    session_id TEXT,
    customer_id TEXT,
    agent_id TEXT,
    data JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_parlant_context_variables_variable_tag_associations_creation_utc 
ON parlant_context_variables_variable_tag_associations(creation_utc);

CREATE INDEX IF NOT EXISTS idx_parlant_context_variables_variable_tag_associations_data_gin 
ON parlant_context_variables_variable_tag_associations USING GIN (data);

-- Create failed migrations table for variable tag associations
CREATE TABLE IF NOT EXISTS parlant_context_variables_variable_tag_associations_failed_migrations (
    id TEXT,
    data JSONB DEFAULT '{}'::jsonb
);

-- Create values table for context variables
CREATE TABLE IF NOT EXISTS parlant_context_variables_values (
    id TEXT NOT NULL PRIMARY KEY,
    version TEXT,
    creation_utc TEXT,
    session_id TEXT,
    customer_id TEXT,
    agent_id TEXT,
    data JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_parlant_context_variables_values_creation_utc 
ON parlant_context_variables_values(creation_utc);

CREATE INDEX IF NOT EXISTS idx_parlant_context_variables_values_data_gin 
ON parlant_context_variables_values USING GIN (data);

-- Create failed migrations table for values
CREATE TABLE IF NOT EXISTS parlant_context_variables_values_failed_migrations (
    id TEXT,
    data JSONB DEFAULT '{}'::jsonb
);

-- ============================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- ============================================

-- Enable RLS on all tables
ALTER TABLE parlant_sessions_ ENABLE ROW LEVEL SECURITY;
ALTER TABLE parlant_sessions__failed_migrations ENABLE ROW LEVEL SECURITY;
ALTER TABLE parlant_sessions_metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE parlant_sessions_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE parlant_sessions_events_failed_migrations ENABLE ROW LEVEL SECURITY;

ALTER TABLE parlant_customers_ ENABLE ROW LEVEL SECURITY;
ALTER TABLE parlant_customers__failed_migrations ENABLE ROW LEVEL SECURITY;
ALTER TABLE parlant_customers_metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE parlant_customers_customer_tag_associations ENABLE ROW LEVEL SECURITY;
ALTER TABLE parlant_customers_customer_tag_associations_failed_migrations ENABLE ROW LEVEL SECURITY;

ALTER TABLE parlant_context_variables_ ENABLE ROW LEVEL SECURITY;
ALTER TABLE parlant_context_variables__failed_migrations ENABLE ROW LEVEL SECURITY;
ALTER TABLE parlant_context_variables_metadata ENABLE ROW LEVEL SECURITY;
ALTER TABLE parlant_context_variables_variable_tag_associations ENABLE ROW LEVEL SECURITY;
ALTER TABLE parlant_context_variables_variable_tag_associations_failed_migrations ENABLE ROW LEVEL SECURITY;
ALTER TABLE parlant_context_variables_values ENABLE ROW LEVEL SECURITY;
ALTER TABLE parlant_context_variables_values_failed_migrations ENABLE ROW LEVEL SECURITY;

-- Create policies to allow all operations
-- Note: Service role key bypasses RLS automatically, so these policies are for anon key usage
-- If you're using service role key, RLS is bypassed and these policies won't apply

-- Sessions policies - allow all operations for authenticated users
CREATE POLICY "Allow all operations on parlant_sessions_" 
ON parlant_sessions_ FOR ALL 
USING (true)
WITH CHECK (true);

CREATE POLICY "Allow all operations on parlant_sessions__failed_migrations" 
ON parlant_sessions__failed_migrations FOR ALL 
USING (true)
WITH CHECK (true);

CREATE POLICY "Allow all operations on parlant_sessions_metadata" 
ON parlant_sessions_metadata FOR ALL 
USING (true)
WITH CHECK (true);

CREATE POLICY "Allow all operations on parlant_sessions_events" 
ON parlant_sessions_events FOR ALL 
USING (true)
WITH CHECK (true);

CREATE POLICY "Allow all operations on parlant_sessions_events_failed_migrations" 
ON parlant_sessions_events_failed_migrations FOR ALL 
USING (true)
WITH CHECK (true);

-- Customers policies
CREATE POLICY "Allow all operations on parlant_customers_" 
ON parlant_customers_ FOR ALL 
USING (true)
WITH CHECK (true);

CREATE POLICY "Allow all operations on parlant_customers__failed_migrations" 
ON parlant_customers__failed_migrations FOR ALL 
USING (true)
WITH CHECK (true);

CREATE POLICY "Allow all operations on parlant_customers_metadata" 
ON parlant_customers_metadata FOR ALL 
USING (true)
WITH CHECK (true);

CREATE POLICY "Allow all operations on parlant_customers_customer_tag_associations" 
ON parlant_customers_customer_tag_associations FOR ALL 
USING (true)
WITH CHECK (true);

CREATE POLICY "Allow all operations on parlant_customers_customer_tag_associations_failed_migrations" 
ON parlant_customers_customer_tag_associations_failed_migrations FOR ALL 
USING (true)
WITH CHECK (true);

-- Context variables policies
CREATE POLICY "Allow all operations on parlant_context_variables_" 
ON parlant_context_variables_ FOR ALL 
USING (true)
WITH CHECK (true);

CREATE POLICY "Allow all operations on parlant_context_variables__failed_migrations" 
ON parlant_context_variables__failed_migrations FOR ALL 
USING (true)
WITH CHECK (true);

CREATE POLICY "Allow all operations on parlant_context_variables_metadata" 
ON parlant_context_variables_metadata FOR ALL 
USING (true)
WITH CHECK (true);

CREATE POLICY "Allow all operations on parlant_context_variables_variable_tag_associations" 
ON parlant_context_variables_variable_tag_associations FOR ALL 
USING (true)
WITH CHECK (true);

CREATE POLICY "Allow all operations on parlant_context_variables_variable_tag_associations_failed_migrations" 
ON parlant_context_variables_variable_tag_associations_failed_migrations FOR ALL 
USING (true)
WITH CHECK (true);

CREATE POLICY "Allow all operations on parlant_context_variables_values" 
ON parlant_context_variables_values FOR ALL 
USING (true)
WITH CHECK (true);

CREATE POLICY "Allow all operations on parlant_context_variables_values_failed_migrations" 
ON parlant_context_variables_values_failed_migrations FOR ALL 
USING (true)
WITH CHECK (true);

-- Note: For production with anon key, you may want more restrictive policies
-- Example: Allow users to only access their own data
-- DROP POLICY "Allow all operations on parlant_sessions_" ON parlant_sessions_;
-- CREATE POLICY "Users can access own sessions" 
-- ON parlant_sessions_ FOR ALL 
-- USING (auth.uid()::text = customer_id)
-- WITH CHECK (auth.uid()::text = customer_id);
```

#### Step 3: Verify Tables Created

After running the SQL, verify the tables exist:
1. Go to **Table Editor** in your Supabase Dashboard
2. You should see tables like:
   - `parlant_sessions_`, `parlant_customers_`, `parlant_context_variables_` (main tables)
   - `parlant_sessions_metadata`, `parlant_customers_metadata`, `parlant_context_variables_metadata` (metadata tables)
   - `parlant_*_failed_migrations` (failed migrations tables)
3. Check **Authentication** → **Policies** to verify RLS policies were created

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

## Troubleshooting

### Tables Not Found

**Error**: Tables don't exist or can't be found.

**Solution:**
1. Run the SQL statements from "Manual Table Setup" in Supabase SQL Editor
2. Verify table names match your `table_prefix` (default: `parlant_sessions_`, `parlant_customers_`, `parlant_context_variables_`)
3. Check schema is set correctly (default: `public`)

### Authentication Errors

**Error**: "Invalid API key" or authentication failures.

**Solution:**
1. Verify `SUPABASE_URL` is your project URL (e.g. `https://your-project.supabase.co`)
2. Verify `SUPABASE_KEY` is your anon/publishable key (Settings → API in Supabase Dashboard)
3. Ensure your Supabase project is active and not paused
4. Check API key has read/write permissions

### Data Not Persisting

**Error**: Data lost after server restart.

**Solution:**
1. Verify tables exist in Supabase Dashboard → Table Editor
2. Check `SUPABASE_URL` and `SUPABASE_KEY` are set correctly
3. Ensure document stores are configured with Supabase in `configure_container`
4. Verify Supabase project is not paused or over quota

---

## Advanced Configuration

### Custom Table Prefix

```python
database = SupabaseDocumentDatabase(
    logger=container[p.Logger],
    table_prefix="custom_prefix_",
)
```

### Query Operations

Supported query operators:
- **Equality**: `{"field": {"$eq": "value"}}`
- **Comparison**: `{"field": {"$gt": 10, "$lt": 20}}`
- **Membership**: `{"field": {"$in": ["a", "b", "c"]}}`
- **Logical**: `{"$and": [...], "$or": [...]}`
- **Pagination**: `limit` and `cursor` parameters
- **Sorting**: By `creation_utc` or `id` (ASC/DESC)

---

## Requirements

- Python 3.10+
- `pip install parlant[supabase]`
- Supabase project
- `SUPABASE_URL` and `SUPABASE_KEY` environment variables
- Tables created manually (see "Manual Table Setup" above)
