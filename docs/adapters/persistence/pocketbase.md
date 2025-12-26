# PocketBase Persistence Adapter

The PocketBase document adapter lets Parlant persist the long–lived parts of a
deployment—sessions, customers, and context variables—inside your PocketBase
instance. That means you can run the server, stop it, and later resume the exact
same conversation state.

This page walks through the required environment variables and shows how to
wire the stores into PocketBase when booting Parlant via the SDK.

## Requirements

1. Install:

   ```bash
   pip install pocketbase
   ```

2. Set up a PocketBase instance. You can:
   - Run PocketBase locally: Download from [pocketbase.io](https://pocketbase.io/) and run `./pocketbase serve`
   - Use a hosted PocketBase service like [PocketBase Cloud](https://pocketbasecloud.com/) or [PocketHost](https://pockethost.io/)
   - Deploy your own PocketBase instance on your infrastructure

3. Set the credentials that `PocketBaseDocumentDatabase` consumes:

   | Variable                     | Required | Description                                                                         |
   |-----------------------------|:--------:|-------------------------------------------------------------------------------------|
   | `POCKETBASE_URL`             |    ✅     | Base URL of your PocketBase instance (e.g. `http://localhost:8090` or `https://your-domain.com`). |
   | `POCKETBASE_ADMIN_TOKEN`     |   ✅*     | Admin authentication token. Use this for production deployments.                    |
   | `POCKETBASE_ADMIN_EMAIL`     |   ✅*     | Admin email for authentication. Required if `POCKETBASE_ADMIN_TOKEN` is not set.     |
   | `POCKETBASE_ADMIN_PASSWORD` |   ✅*     | Admin password for authentication. Required if `POCKETBASE_ADMIN_EMAIL` is set.       |

   > ✅* Provide **either** `POCKETBASE_ADMIN_TOKEN` **or** both `POCKETBASE_ADMIN_EMAIL` and `POCKETBASE_ADMIN_PASSWORD`.
   >
   > For custom deployments, simply set `POCKETBASE_URL` to your custom endpoint URL.

## Custom Deployment Endpoints

The PocketBase adapter fully supports custom deployment endpoints. Simply set `POCKETBASE_URL` to your custom PocketBase instance URL:

```bash
export POCKETBASE_URL="https://your-custom-pocketbase.example.com"
export POCKETBASE_ADMIN_TOKEN="your-admin-token"
```

The adapter will automatically connect to your custom endpoint and handle all operations through the PocketBase REST API.

## SDK / Module Setup

Parlant's SDK exposes a `configure_container` hook that lets you replace the
default persistence layer. The pattern below shows how to register
PocketBase-backed implementations of the three configurable stores:

- `SessionStore` → `SessionDocumentStore`
- `CustomerStore` → `CustomerDocumentStore`
- `ContextVariableStore` → `ContextVariableDocumentStore`

Each store receives its own collection prefix (`parlant_sessions`,
`parlant_customers`, `parlant_context_variables`) so their metadata never
collides. We also rebind `EventEmitterFactory`, so system events get written into
the same store.

```python
from contextlib import AsyncExitStack

import parlant.sdk as p
from parlant.adapters.db.pocketbase_db import PocketBaseDocumentDatabase
from parlant.core.emission.event_publisher import EventPublisherFactory

EXIT_STACK = AsyncExitStack()


async def _make_session_store(container: p.Container) -> p.SessionStore:
    database = await EXIT_STACK.enter_async_context(
        PocketBaseDocumentDatabase(
            logger=container[p.Logger],
            collection_prefix="parlant_sessions",
        )
    )
    store = p.SessionDocumentStore(database=database, allow_migration=True)
    return await EXIT_STACK.enter_async_context(store)


async def _make_customer_store(container: p.Container) -> p.CustomerStore:
    database = await EXIT_STACK.enter_async_context(
        PocketBaseDocumentDatabase(
            logger=container[p.Logger],
            collection_prefix="parlant_customers",
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
        PocketBaseDocumentDatabase(
            logger=container[p.Logger],
            collection_prefix="parlant_context_variables",
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


async def shutdown_pocketbase() -> None:
    await EXIT_STACK.aclose()
```

### Using the SDK

```python
async def main() -> None:
    try:
        async with p.Server(
            configure_container=configure_container,
        ) as server:
            ...
    finally:
        await shutdown_pocketbase()
```

## Advanced Configuration

### Custom HTTP Client

You can provide a custom `httpx.AsyncClient` instance for advanced configuration (timeouts, retries, etc.):

```python
import httpx

client = httpx.AsyncClient(
    timeout=60.0,
    limits=httpx.Limits(max_keepalive_connections=10),
)

database = PocketBaseDocumentDatabase(
    logger=logger,
    connection_params={"url": "https://your-pocketbase.com", "admin_token": "token"},
    http_client=client,
)
```

### Programmatic Configuration

Instead of environment variables, you can pass connection parameters directly:

```python
database = PocketBaseDocumentDatabase(
    logger=logger,
    connection_params={
        "url": "https://your-custom-pocketbase.example.com",
        "admin_token": "your-admin-token",
    },
    collection_prefix="parlant_custom",
)
```

## What Gets Persisted?

Once the PocketBase stores are registered, PocketBase becomes the source of truth for:

- Sessions + events + inspections
- Customers + their tag associations
- Context variables + their values

Other stores (agents, guidelines, journeys, etc.) continue to use their default
backends. If you define them in code at startup, they will automatically be
recreated each time the server runs. For dynamic authoring flows you can follow
the same module approach to route additional stores into PocketBase.

## Features

The PocketBase adapter provides several advantages:

- **Easy Setup**: No complex database configuration required
- **REST API**: Uses PocketBase's REST API, works with any PocketBase deployment
- **Custom Endpoints**: Fully supports custom PocketBase deployment URLs
- **Automatic Collection Management**: Collections are created automatically with proper schemas
- **Flexible Authentication**: Supports both token-based and email/password authentication
- **Efficient Filtering**: Translates Parlant filters to PocketBase filter syntax
- **Cursor-based Pagination**: Efficient pagination support for large datasets

## Troubleshooting

### Connection Issues

If you're having trouble connecting to your PocketBase instance:

1. Verify the `POCKETBASE_URL` is correct and accessible
2. Check that your admin token or credentials are valid
3. Verify `/api/health` returns 200

### Authentication Errors (404 Not Found)

If you encounter `404 Not Found` errors when authenticating:

1. **PocketBase v0.22+**: The adapter automatically uses the `_superusers` collection endpoint (`/api/collections/_superusers/auth-with-password`)
2. **PocketBase v0.21 and earlier**: The adapter falls back to the legacy admin endpoint (`/api/admins/auth-with-password`)
3. Ensure your email/password corresponds to a **Superuser** account (in PocketBase v0.22+) or **Admin** account (in older versions)
4. Verify your PocketBase instance version (Admin UI footer) if unsure

### Collection Creation Errors

If collection creation fails:

1. Ensure you have admin permissions (using admin token or admin email/password)
2. Check that collection names don't conflict with existing collections
3. Verify PocketBase version compatibility (requires PocketBase 0.8.0+)

### Filter Translation

The adapter automatically translates Parlant's filter syntax to PocketBase filters. If you encounter filter-related errors:

1. Check that field names match your document schema
2. Verify that indexed fields are properly configured
3. Review PocketBase filter syntax documentation for complex queries

