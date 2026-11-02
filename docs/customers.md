# Customer Management

A customer is the identity Parlant keeps for one end user. Every session
belongs to a customer, tools read the current customer to act on their
behalf, and rules you create may be scoped to specific customer groups.

A customer record carries:

- **`id`** — a stable identifier. Prefer to use the id your own systems already
  know (your CRM or user-database id), so the customer keeps one identity
  across deployments and integrations.
- **`name`** — a display name.
- **`metadata`** — string key-value pairs your tools and hooks can read
  (for example: email, plan, preferred language).
- **`groups`** — group memberships, used to scope rules and filter customers.
  You define groups in code (`server.create_group`), and assign customers to
  them at runtime; ids like `grp_vip` below come from those definitions.
- **`kind`** — `real` or `persona`. Real is the default. See
  [Personas](#personas) below.

## The guest customer

One special, built-in customer always exists: **guest**. It owns all
anonymous conversations (see [Authentication](auth.md) for how guest
identity works).

The guest is a real customer, not a persona. Guest sessions carry live
traffic from people who have not logged in, so Parlant treats the guest as
somebody whose data deserves the same care as any named customer's.

## Personas

A persona is a customer that stands in for a real one while you test. You
test against a persona so that a test run never reaches a real person's
data, and never fires a tool on their behalf.

Mark a customer as a persona when you create it:

```python
import parlant.sdk as p

async with p.Server() as server:
    agent = await server.create_agent(
        name="Support Agent",
        description="Helps customers with orders and returns.",
    )

    test_customer = await server.create_customer(
        id="persona_returns_flow",
        name="Dana Levi (test)",
        metadata={"email": "dana@example.test", "plan": "pro"},
        kind=p.CustomerKind.PERSONA,
    )

    print(test_customer.kind)  # CustomerKind.PERSONA
```

Two rules apply to every persona.

**A persona cannot be changed back, and a real customer cannot be turned
into one.** The kind is set once, when the customer is created. There is no
field for it on any update. Converting a live customer into a persona is the
mistake the whole feature exists to prevent, so Parlant does not offer a way
to do it. If you picked the wrong kind, delete the customer and create it
again.

**Nobody can log in as a persona.** A customer JWT whose subject names a
persona is rejected, so no end user is ever served as one, however the token
was minted. Only an admin can open a session with a persona. See
[Authentication](auth.md#personas-and-authentication).

### Checking whether a customer is a persona

Every way of getting a customer back carries the kind, so you can read it
off whatever the SDK handed you:

```python
customer = await server.get_customer(id="persona_returns_flow")

if customer.is_persona:
    print(f"{customer.name} is safe to test against")

print(customer.kind)  # CustomerKind.PERSONA
```

`is_persona` is the short form of comparing `kind`. It is `False` for
anything that is not explicitly a persona, so code that guards on it treats
an unfamiliar customer as a real one.

### Listing personas

Both the API and the store take a `kind` filter, so you can separate your
test customers from your real ones:

```bash
curl "http://localhost:8800/customers?kind=persona" \
  -H "Authorization: Bearer $PARLANT_ADMIN_API_KEY"
```

A listing filtered to `persona` never includes the guest. A listing filtered
to `real`, and an unfiltered listing, both do.

## Provisioning customers

Provisioning means making sure the customer record exists before it is
needed. There are four ways to do it. If you already have a customer
database, plug it in directly — that is the recommended way. The other
three fit when you'd rather let Parlant keep the records.

### Plugging in your own customer database (recommended)

You almost certainly have a users table already. Don't copy it into
Parlant. Instead, implement the `CustomerStore` interface over your own
database and hand it to the server:

```python
from parlant.core.customers import Customer, CustomerId, CustomerStore


class MyCustomerStore(CustomerStore):
    """Serves customers straight from your existing user database."""

    def __init__(self, db: MyDatabase) -> None:
        self._db = db

    async def read_customer(self, customer_id: CustomerId) -> Customer:
        user = await self._db.users.get(customer_id)

        return Customer(
            id=customer_id,
            name=user.full_name,
            metadata={"plan": user.plan},
            groups=[],
            creation_utc=user.created_at,
            modified_utc=user.updated_at,
        )

    # ... the remaining CustomerStore methods


async with p.Server(customer_store=MyCustomerStore(db)) as server:
    ...
```

With this, there is no provisioning step at all and nothing to keep in
sync: Parlant reads customers live from your database. A plan change in
your database is visible to the very next turn, and deleting the user there
deletes them for Parlant too. (See [Dependency Injection](dependency-injection.md)
for how store binding works.)

**Your store decides what a persona is.** `Customer.kind` defaults to
`CustomerKind.REAL`, so a store that never sets it reports every customer
as real. That is the safe answer, and it means your own store needs no
change. It also means [personas](#personas) only exist in your deployment
if you store the kind and return it. Your users table is unlikely to have a
column for it, so you will either add one or keep your personas somewhere
else.

**How this works with authentication.** Authentication keeps working
exactly as before, because identity is never proven by the store. A
customer arrives with a JWT — the signed login token your backend creates
for them (see [Authentication](auth.md)). Parlant verifies the token first:
signature, audience, expiry. Only then does it ask the store whether the
customer id in the token (its `sub` claim) exists. Three points to get
right:

- **Your tokens and your customers come from the same place, so they can't disagree.**
  The backend that issues login tokens and the store that
  answers "does this customer exist?" both read the same user database.
  Any customer who can log in is a customer your store can find. There is
  nothing to sync and no moment where the two are out of step.
- **You probably want auto-provisioning to stay off** (it is off by
  default). Auto-provisioning tells Parlant: if a valid token names a
  customer you don't know, create them. But with your own store, "create
  them" means writing into your users table — which your store most likely
  doesn't support, and shouldn't. Leave it off, and a token for a customer
  your database doesn't know is simply rejected with a 401.
- **Remember that the guest customer has no row in your database.** The
  built-in store answers `read_customer(CustomerStore.GUEST_ID)` from code,
  not from data — and now that job is yours. Parlant reads the session's
  customer on every turn, including for anonymous sessions, so your store
  must recognize the guest id and return the guest customer itself.
  Otherwise, every anonymous conversation breaks:

  ```python
  async def read_customer(self, customer_id: CustomerId) -> Customer:
      if customer_id == CustomerStore.GUEST_ID:
          return Customer(
              id=CustomerStore.GUEST_ID,
              name="Guest",
              metadata={},
              groups=[],
              creation_utc=GUEST_CREATION_UTC,   # any fixed datetime
              modified_utc=GUEST_CREATION_UTC,
          )
      user = await self._db.users.get(customer_id)
      ...
  ```

The three remaining options use Parlant's own customer store, and differ by
how the customer first meets your agent.

### Auto-provisioning (recommended if you don't implement CustomerStore)

If customers reach Parlant through your app with a JWT your backend
created, you don't need to create them at all. Turn on auto-provisioning,
and the first valid token from an unknown customer creates the record on
the spot. The id comes from the token's `sub` claim, and the name from its
`name` claim.

You can set the environment variable:

```bash
export PARLANT_AUTH_AUTO_PROVISION_CUSTOMERS=true
```

...or set `auto_provision_customers=True` in `AuthConfig` — see
[Authentication](auth.md) for the full token flow.

With this on, your backend never manually syncs users to Parlant.
The customer base builds itself from whoever actually talks to the agent.

### Pre-provisioning via the API

Create the customer explicitly when your backend needs it to exist
*before* the customer ever opens a chat. The main case is outreach that
your side starts, like a delivery update or a renewal reminder, sent to a
customer who has never contacted the agent:

```http
POST /customers
Content-Type: application/json

{
  "id": "crm-90211",
  "name": "Dana Cohen",
  "metadata": {"email": "dana@example.com", "plan": "pro"},
  "groups": ["grp_vip"]
}
```

Then create the session with `customer_id: "crm-90211"` and ask the agent to
speak first (see [Sessions](sessions.md), "AI-Agent Message").

### In code, at startup

For a fixed set of known users, like an internal tool or a demo, create
customers in the same code that defines your agent:

```python
await server.create_customer(
    id="crm-90211",
    name="Dana Cohen",
    metadata={"email": "dana@example.com", "plan": "pro"},
)
```

## Reading and searching

Read one customer:

```http
GET /customers/{customer_id}
```

List customers, with the same cursor pagination as sessions:

```http
GET /customers?limit=25&sort=asc
```

Two filters are worth knowing:

- **`name_prefix`** — case-insensitive prefix search, for support tools
  and search-as-you-type: `GET /customers?name_prefix=dan` finds "Dana
  Cohen" and "Daniel Roth". It stays fast at large customer counts.
- **`min_modified_utc`** — only customers changed since a timestamp, for
  sync jobs that mirror customers into another system.

## Updating customers

`PATCH /customers/{customer_id}` updates the name, and adds or removes
metadata keys and group memberships independently:

```http
PATCH /customers/{customer_id}
Content-Type: application/json

{
  "metadata": {
    "set": {"plan": "enterprise"},
    "unset": ["trial_ends"]
  },
  "groups": {
    "add": ["grp_vip"]
  }
}
```

In practice, this call usually comes from a business event, not from a
person editing a record. A customer upgrades their subscription; your
billing webhook fires; it moves the customer into the VIP group. From the
next turn, every rule scoped to that group applies to their conversations.

In code, the SDK makes the same membership changes through the customer's
`groups` accessor. `create_group` (or `get_group`) gives you a group you
pass by object:

```python
vip = await server.create_group(name="VIP")

customer = await server.get_customer(id="crm-90211")
await customer.groups.add(vip)      # idempotent
await customer.groups.remove(vip)

if vip in customer.groups:          # membership check
    ...
```

You can also set a customer's groups when you create them:

```python
await server.create_customer(
    id="crm-90211",
    name="Dana Cohen",
    groups=[vip],
)
```

## Scoping rules to a customer group

Group membership becomes useful the moment rules depend on it. The
reliable pattern has two parts. First, one **observation** decides
membership. (An observation is a rule that recognizes a situation without
prescribing an action — see [Policy Design](policy-design.md).) Then the
rules that should only apply to the group depend on that observation. The
point of doing it this way: membership is decided by exact code, never by
the model's judgment.

The observation uses an `evaluation_func`, a Python function that Parlant
calls instead of the model to decide whether the rule matches the turn. A
group provides one ready-made: `includes_current_customer` matches on
exactly the turns where the session's customer belongs to the group.

```python
vip = await server.create_group(name="VIP")

vip_customer = await agent.create_observation(
    condition="The customer is a VIP",
    evaluation_func=vip.includes_current_customer,
)
```

The group also provides `excludes_current_customer`, the inverse, for a
rule that should apply only to customers outside the group.

Then scope any rule to the group by depending on the observation:

```python
await agent.create_policy(
    content=(
        "When the customer reports a problem, offer to connect them "
        "to the priority support line"
    ),
    dependencies=[vip_customer],
)

await agent.create_policy(
    content=(
        "When the customer completes a purchase, mention the yearly "
        "VIP anniversary gift"
    ),
    dependencies=[vip_customer],
)
```

For non-VIP customers, the observation evaluates to false and both dependent
policies stay out of the conversation entirely. When the billing webhook
above moves a customer into the VIP group, they start applying from the next
turn — no code change, no restart.

Membership is decided once, in one place. Each dependent policy then only
needs a condition that makes sense *within* the VIP context, because the
question "is this a VIP?" is already answered for it. (See
[Policy Design](policy-design.md) for the dependency pattern in general.)

## Deleting customers (and the right to be forgotten)

Deletion is two calls. Together they let your backend automate the right
to be forgotten: when a user deletes their account, your deletion flow
removes both the customer record and every conversation they ever had:

```http
DELETE /sessions?customer_id=crm-90211
DELETE /customers/crm-90211
```

Delete the sessions first, then the customer. Both are permanent: the
sessions' event logs are gone with them. The guest customer cannot be
deleted; guest sessions are deleted per-session, or by
`DELETE /sessions?customer_id=guest`.

## Reading the customer from a tool

Inside a turn, `p.Customer.current` is the customer of the active session:

```python
@p.tool
async def get_account_summary() -> p.ToolResult:
    plan = await p.Customer.current.metadata.get("plan")
    summary = await billing.summarize(p.Customer.current.id, plan=plan)
    return p.ToolResult(data=summary)
```

`metadata.get()` is an async read that refreshes from the store; plain
`metadata["plan"]` reads the turn's local snapshot. Metadata writes
(`metadata.set()` / `metadata.delete()`) persist immediately.
