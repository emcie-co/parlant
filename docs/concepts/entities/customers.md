# Customers

In Parlant, a **customer** is a code word for anyone who interacts with an agent—regardless of the real nature of the relationship between them. In other words, a customer can be a real person, a bot, or even a human agent.

While agents can operate anonymously (without knowing who they're talking to), Parlant allows you to track registered customers and provide deeply personalized experiences based on their identity and preferences.

By letting your agents understand who they're talking to, you can tailor interactions for different user segments: high-profile customers might receive premium offers, new users can get focused onboarding guidance, and so forth...

Parlant makes customer registration simple, requiring only minimal identification—a name is enough to get started.

```python
import parlant.sdk as p

async with p.Server() as server:
    # Register a new customer
    customer = await server.create_customer(name="Alice")
```

## Authentication
Parlant aims to live as a backend service, leaving authentication and authorization to the application layer. This means that while you can register customers, you should handle their authentication (e.g., via OAuth, JWT, etc.) in your application code, in whatever way suits your needs.

Once you have identified your customer, then you can pass their ID to the agent, allowing it to personalize interactions based on the registered customer.

## Storage
You can choose where you store customers.

By default, Parlant does not persist customers, meaning that they are stored in memory and will be lost when the server restarts. This is useful for testing and development purposes.

If you want to persist customers, you can configure Parlant to use a database of your choice. For local persistence, we recommend using the integrated JSON file storage, as there's zero setup required. For production use, you can use MongoDB, which comes built-in, or another database.

### Persisting to Local Storage
This will save customers under `$PARLANT_HOME/customers.json`.

```python
import asyncio
import parlant.sdk as p

async def main():
    async with p.Server(customer_store="local") as server:
        # ...

asyncio.run(main())
```

### Persisting to MongoDB
Just specify the connection string to your MongoDB database when starting the server:

```python
import asyncio
import parlant.sdk as p

async def main():
    async with p.Server(customer_store="mongodb://path.to.your.host:27017") as server:
        # ...

asyncio.run(main())
```

## Customer Groups

You can also divide your customers into different groups and control group-specific personalization by using **tags**.

For example, you can create a tag for VIP customers:

```python
# Create a new tag to represent VIP customers
vip_tag = await server.create_tag(name="VIP")

# Register a new customer
customer = await server.create_customer(name="Alice", tags=[vip_tag.id])
```

> **Tip: Learn More**
> To learn more about advanced personalization possibilities for specific customers and groups, check out the [variables](https://parlant.io/docs/concepts/customization/variables) section.

## Adding Metadata
You can also attach custom metadata to customers, which can be used to store additional information about them. This metadata can be used to further personalize interactions or to provide context for tool calls.

```python
customer = await server.create_customer(name="Alice", metadata={
    "external_id": "12345",
    "location": "USA",
})
```

```python
@p.tool
async def get_customer_location(context: p.ToolContext) -> p.ToolResult:
    server = p.ToolContextAccessor(context).server

    if customer := await server.find_customer(id=context.customer_id):
        return p.ToolResult(customer.metadata.get("location", "Unknown location"))

    return p.ToolResult("Customer not found")
```

## Registering Customers
While you can register customers using the SDK itself, it's often more practical to handle customer registration through your application layer. This allows you to integrate customer management with your existing user authentication and authorization systems.

You can do this by using Parlant's REST API or native Client SDKs to create and manage customers.

```python
from parlant.client import ParlantClient

# Change localhost to your server's address
client = ParlantClient("http://localhost:8800")

client.customers.create(
    name="Alice",
    metadata={
        "external_id": "12345",
        "location": "USA",
        "hobby": "reading",
    },
    tags=[TAG_ID]  # Optional: specify tag IDs to assign to the customer
)
```

## Updating Customer Data

You can update customer data at any time, including their name, metadata, and tags. This is useful for keeping customer information up-to-date as your application evolves.

```python
client.customers.update(
    customer_id=CUSTOMER_ID,
    name="Alice Smith",
    metadata={
        "set": {
            "location": "Canada",
        },
        "remove": ["hobby"],
    },
    tags=[NEW_TAG_ID]  # Optional: specify new tag IDs to assign to the customer
)
```
