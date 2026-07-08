# Authentication & Authorization

This document explains how Parlant's OSS auth infrastructure works and how to
set it up step by step — including wiring it to your existing customer database.

## The model

Parlant distinguishes three questions, answered by three swappable ports:

| Question | Port | OSS default |
|---|---|---|
| **Who is calling?** | `Authenticator` | `TokenAuthenticator` — stateless bearer tokens |
| **May this *kind* of caller do this operation?** | `PermissionGate` | `DefaultPermissionGate` — per-principal allowlist |
| **May this caller touch this *specific* resource?** | `Authorizer` | `OwnershipAuthorizer` — session ownership |

Every request resolves to exactly one **`Principal`**:

- **`AnonymousPrincipal`** — no credentials. May only read agent info and start
  a guest conversation.
- **`GuestPrincipal`** — an unidentified browser holding a Parlant-minted guest
  token. All guests share the singleton `guest` customer, but each guest token
  carries a random `guest_instance_id` that scopes it to *the sessions it
  created* — one guest can never read another guest's conversation.
- **`CustomerPrincipal`** — a specific customer, proven by a JWT that **your**
  backend minted (Parlant only verifies it). Scoped to that customer's own
  sessions.
- **`AdminPrincipal`** — full API access (entity creation, configuration,
  everything), proven by a static API key.

Principals also carry an open `attributes` map (the JWT's claims), which custom
gates/authorizers may consult — the OSS defaults ignore unknown attributes.

### Statelessness — why there are no refresh tokens

Parlant stores **no token state whatsoever**. Every request is judged solely on
the token it presents:

- Customer tokens are issued, refreshed, and expired by **your** systems.
  Parlant cannot "block" a customer: an expired token is rejected with 401, and
  the very next request with a fresh token succeeds.
- Guest tokens are signed (HS256) by Parlant with a configurable TTL (default
  one week) — no server-side session record.
- Revocation, when you need it, lives in *your* identity system (short TTLs,
  key rotation), not in a Parlant block-list.

### HTTP status codes

- **401 Unauthorized** — the credentials failed verification (bad signature,
  expired, unknown customer).
- **403 Forbidden** — valid identity, but the operation or resource is not
  allowed (e.g. a customer reading another customer's session).
- **429 Too Many Requests** — rate limit exceeded (per client IP by default;
  admins are exempt).

---

## Quick start (SDK)

```python
import parlant.sdk as p

async with p.Server(
    auth=p.AuthConfig(
        admin_api_key="your-long-random-admin-key",
        customer_jwt_secret="secret-shared-with-your-backend",  # HS256
        guest_token_secret="another-long-random-secret",
        auto_provision_customers=False,
    ),
) as server:
    ...
```

That's it — the API now enforces the guest/customer/admin model described above.

`AuthConfig` fields:

| Field | Meaning |
|---|---|
| `admin_api_key` | Static key for full API access. **Required.** |
| `customer_jwt_secret` | Shared HMAC secret for verifying your backend's customer JWTs (HS256/384/512). |
| `customer_jwks_url` | *Alternative* to the shared secret: your IdP's JWKS URL for asymmetric JWTs (RS256/ES256 etc.). Mutually exclusive with `customer_jwt_secret`. |
| `guest_token_secret` | Secret for Parlant-minted guest tokens. If unset, an ephemeral one is generated — guest tokens then die with the process (a restart logs guests out). |
| `auto_provision_customers` | If `True`, a valid customer JWT whose `sub` is not yet a Parlant customer creates that customer on first contact. Default `False` (unknown customers are rejected with 401). |

Leaving both `customer_jwt_secret` and `customer_jwks_url` unset disables
customer authentication entirely (guests + admin only).

### Standalone server (env vars)

Running `parlant-server` with `PARLANT_ENV=production` reads the same
configuration from the environment and **fails fast at startup** if the admin
key is missing:

```bash
export PARLANT_ENV=production
export PARLANT_ADMIN_API_KEY="your-long-random-admin-key"          # required
export PARLANT_CUSTOMER_JWT_SECRET="secret-shared-with-backend"    # or:
# export PARLANT_CUSTOMER_JWKS_URL="https://idp.example.com/.well-known/jwks.json"
export PARLANT_GUEST_TOKEN_SECRET="another-long-random-secret"     # recommended
export PARLANT_AUTH_AUTO_PROVISION_CUSTOMERS=false
```

Without `PARLANT_ENV=production` (development mode), everything is allowed —
no credentials needed. Never expose a development-mode server publicly.

---

## Step-by-step: customer auth with your existing customer DB

Parlant is **not** an identity provider. Your backend already knows who's
logged in; it vouches for that identity by minting a small JWT that Parlant
verifies. Three steps:

### Step 1 — Make your customers known to Parlant

The JWT's `sub` claim must be a Parlant `CustomerId`. Pick ONE of:

**(a) Sync your users into Parlant** (explicit ids let `sub` be your own user id):

```python
async with p.Server(auth=...) as server:
    agent = await server.create_agent(name="Support", description=...)
    # e.g. on your user-created webhook, or a one-time backfill:
    await server.create_customer(name="Jane Doe", id="user_1234")
```

**(b) Auto-provision** — set `auto_provision_customers=True` and skip the sync:
the first request with a valid JWT for an unknown `sub` creates the customer
(using the JWT's `name` claim if present). Best when Parlant is the system of
record only for conversations, not for customer profiles.

**(c) Bring your own `CustomerStore`** — if you want Parlant to read customer
profiles *live from your DB* instead of copying them, implement the
`CustomerStore` interface and plug it into the server:

```python
class MyDBCustomerStore(p.CustomerStore):
    async def read_customer(self, customer_id: p.CustomerId) -> Customer:
        row = await my_db.get_user(customer_id)
        if row is None:
            raise ItemNotFoundError(item_id=UniqueId(customer_id))
        return Customer(
            id=customer_id,
            creation_utc=row.created_at,
            name=row.display_name,
            extra={"tier": row.plan},
            tags=[],
        )
    # ...implement the remaining CustomerStore methods...

async with p.Server(
    customer_store=MyDBCustomerStore(),
    auth=p.AuthConfig(...),
) as server:
    ...
```

Note the separation: **identity** (the verified `customer_id`) comes from the
token; the **profile** (name, tier, metadata used for personalization) comes
from the `CustomerStore`. Don't conflate them.

### Step 2 — Mint a JWT in your backend

After *your* login flow establishes who the user is, mint a short-lived token
(any JWT library works; this is PyJWT):

```python
import time, jwt

def make_parlant_token(user_id: str, display_name: str) -> str:
    return jwt.encode(
        {
            "sub": user_id,                    # the Parlant CustomerId
            "name": display_name,              # used by auto-provisioning
            "exp": int(time.time()) + 15 * 60, # your policy
        },
        PARLANT_CUSTOMER_JWT_SECRET,           # the shared secret from Step 1
        algorithm="HS256",
    )
```

Hand this token to your frontend (embed it in the page bootstrap, or expose a
small `/chat-token` endpoint the frontend calls). When it expires, your
frontend fetches a fresh one from *you* — Parlant has no refresh flow.

If you already run an IdP/SSO that issues RS256/ES256 JWTs, skip the minting:
point `customer_jwks_url` at your JWKS endpoint and pass your existing access
token, as long as its `sub` maps to a Parlant `CustomerId`.

### Step 3 — Call Parlant from the browser

```javascript
// create a session bound to the token's customer (no customer_id in the body!)
const session = await fetch(`${PARLANT_URL}/sessions`, {
    method: "POST",
    headers: {
        "Authorization": `Bearer ${parlantToken}`,
        "Content-Type": "application/json",
    },
    body: JSON.stringify({ agent_id: AGENT_ID }),
}).then(r => r.json());

// post a message
await fetch(`${PARLANT_URL}/sessions/${session.id}/events`, {
    method: "POST",
    headers: { "Authorization": `Bearer ${parlantToken}`, "Content-Type": "application/json" },
    body: JSON.stringify({ kind: "message", source: "customer", message: "Hi!" }),
});

// poll for the agent's reply
const events = await fetch(
    `${PARLANT_URL}/sessions/${session.id}/events?min_offset=0&wait_for_data=30`,
    { headers: { "Authorization": `Bearer ${parlantToken}` } },
).then(r => r.json());
```

The session's `customer_id` always comes from the token. A `customer_id` in
the request body that contradicts the token is rejected with 403. The customer
can only create, read, and post into **their own** sessions.

---

## Guest flows (no login)

For visitors without an account, no backend work is needed at all:

1. The browser calls `POST /sessions` with **no** `Authorization` header.
2. Parlant creates a session for the `guest` customer and returns a freshly
   minted guest token in the **`X-Parlant-Guest-Token`** response header.
3. The browser stores that token (e.g. `sessionStorage`) and sends it as
   `Authorization: Bearer <token>` on subsequent requests — including creating
   *additional* sessions, which will belong to the same guest.

A guest can only touch the sessions its own token created: other guests'
sessions, and all customer sessions, return 403. Guest tokens expire after one
week by default (`TokenAuthenticator(guest_token_ttl_seconds=...)` to change).

If you serve the browser from a different origin, note that the default CORS
setup already exposes `X-Parlant-Guest-Token`; if you override
`configure_app`, keep that header in `expose_headers`.

## Admin flows

Backoffice tools and server-to-server integrations authenticate with the
static admin key:

```bash
curl -H "Authorization: Bearer $PARLANT_ADMIN_API_KEY" \
     -X POST $PARLANT_URL/agents -d '{"name": "New Agent"}' -H 'Content-Type: application/json'
```

Admins bypass ownership checks and rate limits, may create sessions for any
customer (`customer_id` in the body is honored), and are the only principals
allowed on WebSocket endpoints (e.g. `/logs` — pass the key as a `?token=`
query parameter, since browsers can't set WebSocket headers).

Treat the admin key like a database password: never ship it to a browser.

---

## Customizing (the enterprise seams)

The composite policy is a pipeline of ports; each can be swapped independently
while reusing the rest:

```python
from parlant.sdk import (
    CompositeAuthorizationPolicy, Authenticator, PermissionGate, Authorizer,
    Principal, CustomerPrincipal, AnonymousPrincipal,
)

class MySSOAuthenticator(Authenticator):
    async def resolve(self, request):
        token = extract_bearer(request)
        if token is None:
            return AnonymousPrincipal()
        claims = await my_sso.introspect(token)   # opaque-token introspection
        return CustomerPrincipal(customer_id=claims["uid"], attributes=claims)

    async def resolve_websocket(self, websocket):
        ...

policy = CompositeAuthorizationPolicy(
    authenticator=MySSOAuthenticator(),
    # permission_gate=..., authorizer=..., rate_limiter=...  (defaults reused)
)

async with p.Server(auth=policy) as server:  # a policy instance binds directly
    ...
```

Typical swaps:

- **`Authenticator`** — SSO/OIDC, mTLS, opaque-token introspection against your
  session store, API-key vaults.
- **`PermissionGate`** — RBAC or scope-based gating; read roles from
  `principal.attributes`.
- **`Authorizer`** — org/team-level tenancy: e.g. allow any principal whose
  `attributes["org"]` matches the session's customer's org.
- **`RateLimiter`** — per-principal or per-plan quotas instead of per-IP.

Two rules keep the seam clean:

1. Custom concepts (org ids, roles, scopes) travel in `Principal.attributes`,
   never as new fields on the core principal types.
2. For total control, implement `AuthorizationPolicy` itself — the API layer
   depends only on that interface.

## Reference: what each principal may do (OSS defaults)

| Operation | Anonymous | Guest | Customer | Admin |
|---|---|---|---|---|
| Read agent info | ✓ | ✓ | ✓ | ✓ |
| Create guest session | ✓ (mints token) | ✓ (own) | — | ✓ |
| Create customer session | — | — | ✓ (own only) | ✓ (any) |
| Read session / list events / read event | — | own only | own only | ✓ |
| Post customer message / status event | — | own only | own only | ✓ |
| List sessions | — | — | own only | ✓ |
| Everything else (agents, journeys, rules, customers, …) | — | — | — | ✓ |

Session "ownership": a guest owns the sessions tagged with its
`guest_instance_id` (stored in session metadata under a reserved key); a
customer owns the sessions whose `customer_id` equals the token's `sub`.
