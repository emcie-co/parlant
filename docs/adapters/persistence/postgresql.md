# PostgreSQL Persistence Adapter

The PostgreSQL adapter lets Parlant persist sessions, customers, and
context variables inside a standard PostgreSQL database. This gives you a
self-hosted, production-ready persistence layer without vendor lock-in.

This page covers prerequisites, configuration, and SDK wiring.

## Prerequisites

### 1. Install the optional dependency

```bash
pip install "parlant[postgres]"
```

This pulls in `asyncpg` for async PostgreSQL access.

### 2. Have a PostgreSQL instance running

Any PostgreSQL 14+ instance will work. A few common ways to get one:

**Docker** (quickest):
```bash
docker run -d --name parlant-postgres \
  -e POSTGRES_USER=parlant \
  -e POSTGRES_PASSWORD=parlant \
  -e POSTGRES_DB=parlant \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

> Using the `pgvector/pgvector` image gives you both PostgreSQL and the pgvector
> extension, which you'll need if you also plan to use the
> [pgvector vector adapter](../vector_db/pgvector.md).

**System package** (apt):
```bash
sudo apt install postgresql postgresql-contrib
sudo -u postgres createuser --pwprompt parlant
sudo -u postgres createdb -O parlant parlant
```

**Conda** (useful for isolated environments without root access):
See [this gist](https://gist.github.com/gwangjinkim/f13bf596fefa7db7d31c22efd1627c7a)
for a walkthrough of running PostgreSQL inside a Conda environment.

### 3. Set the connection string

| Variable                    | Required | Description                                                        |
|-----------------------------|:--------:|--------------------------------------------------------------------|
| `POSTGRES_CONNECTION_STRING`|    ✅     | Standard PostgreSQL connection URI.                                |

Example:
```
postgresql://parlant:parlant@localhost:5432/parlant
```

## SDK / Module Setup

Parlant's SDK accepts PostgreSQL connection strings directly for the three
configurable document stores. Pass the connection string to `session_store`,
`customer_store`, and/or `variable_store`:

```python
import parlant.sdk as p

async def main() -> None:
    postgres_url = "postgresql://parlant:parlant@localhost:5432/parlant"

    async with p.Server(
        nlp_service=p.NLPServices.openai,
        session_store=postgres_url,
        customer_store=postgres_url,
        variable_store=postgres_url,
        migrate=True,
    ):
        pass
```

When a `postgresql://` (or `postgres://`) connection string is provided, the SDK
automatically creates a `PostgresDocumentDatabase` instance backed by an
`asyncpg` connection pool.

### Manual container configuration

For more control — for example, to also wire up persistent JSON-file stores for
agents, guidelines, and other entities — use the `configure_container` callback.
See [`examples/run_postgres_server.py`](../../../examples/run_postgres_server.py)
for a full working example.

## What Gets Persisted?

Once the PostgreSQL stores are registered, PostgreSQL becomes the source of truth
for:

- **Sessions** + events + inspections
- **Customers** + their tag associations
- **Context variables** + their values

Other stores (agents, guidelines, glossary, journeys, etc.) continue to use
their default backends unless you explicitly override them in
`configure_container`.

## Table Layout

Each store creates tables with a prefix to avoid collisions:

| Store              | Table prefix               |
|--------------------|----------------------------|
| Sessions           | `parlant_sessions_`        |
| Customers          | `parlant_customers_`       |
| Context variables  | `parlant_context_variables_`|

Each collection becomes a table with the following schema:

| Column         | Type   | Description                          |
|----------------|--------|--------------------------------------|
| `id`           | TEXT   | Primary key                          |
| `version`      | TEXT   | Document version                     |
| `creation_utc` | TEXT   | ISO timestamp, used for cursor pagination |
| `data`         | JSONB  | Full document payload                |

A GIN index on the `data` column supports efficient JSONB queries.

## Common Issues

### Connection refused
Make sure PostgreSQL is running and accepting connections on the host/port in
your connection string. For Docker, verify the container is up and the port
mapping is correct.

### Permission denied / database does not exist
Create the database and user before starting Parlant:
```sql
CREATE USER parlant WITH PASSWORD 'parlant';
CREATE DATABASE parlant OWNER parlant;
```

### Data not persisting across restarts
Verify you're passing the connection string to the SDK (not `None`). When the
store parameters are omitted, Parlant falls back to transient in-memory storage.
