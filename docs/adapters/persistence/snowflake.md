# Snowflake Persistence Adapter

The Snowflake document adapter lets Parlant persist the long–lived parts of a
deployment—sessions, customers, and context variables—inside your Snowflake
account. That means you can run the server (for example inside Snowpark
Container Services), stop it, and later resume the exact same conversation
state.

This page walks through the required environment variables and shows how to
wire the stores into Snowflake when booting Parlant via the SDK.

## Requirements

1. Install the optional dependency (or otherwise provide
   `snowflake-connector-python`):

   ```bash
   pip install "parlant[snowflake]"
   ```

2. Set the credentials that `SnowflakeDocumentDatabase` consumes:

   | Variable                     | Required | Description                                                                         |
   |-----------------------------|:--------:|-------------------------------------------------------------------------------------|
   | `SNOWFLAKE_ACCOUNT`         |    ✅     | Account locator (e.g. `abc-xy123`).                                                  |
   | `SNOWFLAKE_USER`            |    ✅     | Username that Parlant should authenticate as.                                        |
   | `SNOWFLAKE_PASSWORD`        |   ✅*     | Password for password-based auth. Skip when using OAuth (see `SNOWFLAKE_TOKEN`).     |
   | `SNOWFLAKE_TOKEN`           |   ✅*     | OAuth access token. When set, the adapter automatically switches to OAuth.           |
   | `SNOWFLAKE_WAREHOUSE`       |    ✅     | Warehouse to execute queries against.                                                |
   | `SNOWFLAKE_DATABASE`        |    ✅     | Database that will host the Parlant tables.                                          |
   | `SNOWFLAKE_SCHEMA`          |    ✅     | Schema inside the database.                                                          |
   | `SNOWFLAKE_ROLE`            |    ➖     | Optional role override.                                                              |

   > ✅* Provide **either** `SNOWFLAKE_PASSWORD` **or** `SNOWFLAKE_TOKEN`.

## SDK / Module Setup

Parlant’s SDK exposes a `configure_container` hook that lets you replace the
default persistence layer. The pattern below shows how to register
Snowflake-backed implementations of the three configurable stores:

- `SessionStore` → `SessionDocumentStore`
- `CustomerStore` → `CustomerDocumentStore`
- `ContextVariableStore` → `ContextVariableDocumentStore`

Each store receives its own table prefix (`PARLANT_SESSIONS_`,
`PARLANT_CUSTOMERS_`, `PARLANT_CONTEXT_VARIABLES_`) so their metadata never
collides. We also rebind `EventEmitterFactory`, so system events get written into
the same store.

```python
from contextlib import AsyncExitStack

import parlant.sdk as p
from parlant.adapters.db.snowflake_db import SnowflakeDocumentDatabase
from parlant.core.emission.event_publisher import EventPublisherFactory

EXIT_STACK = AsyncExitStack()


async def _make_session_store(container: p.Container) -> p.SessionStore:
    database = await EXIT_STACK.enter_async_context(
        SnowflakeDocumentDatabase(
            logger=container[p.Logger],
            table_prefix="PARLANT_SESSIONS_",
        )
    )
    store = p.SessionDocumentStore(database=database, allow_migration=True)
    return await EXIT_STACK.enter_async_context(store)


async def _make_customer_store(container: p.Container) -> p.CustomerStore:
    database = await EXIT_STACK.enter_async_context(
        SnowflakeDocumentDatabase(
            logger=container[p.Logger],
            table_prefix="PARLANT_CUSTOMERS_",
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
        SnowflakeDocumentDatabase(
            logger=container[p.Logger],
            table_prefix="PARLANT_CONTEXT_VARIABLES_",
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


async def shutdown_snowflake() -> None:
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
        await shutdown_snowflake()
```

## What Gets Persisted?

Once the Snowflake stores are registered, Snowflake becomes the source of truth for:

- Sessions + events + inspections
- Customers + their tag associations
- Context variables + their values

Other stores (agents, guidelines, journeys, etc.) continue to use their default
backends. If you define them in code at startup, they will automatically be
recreated each time the server runs. For dynamic authoring flows you can follow
the same module approach to route additional stores into Snowflake.
