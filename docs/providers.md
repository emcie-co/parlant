# Configuring Models and Providers

A Parlant agent doesn't use just one kind of model request. Behind each turn
there are several kinds of model work: generating the response, matching
rules, reviewing drafts, embedding, compacting history. The `models` mapping
controls what runs each kind, at whatever level of detail you need:

- **One provider for everything** — `models={"*": "openrouter"}` and you're done;
  each kind of work uses that provider's sensible default model.
- **Override a family of work** — for example, route the reviews to a cheaper model.
- **Pin exact models to exact operations** — down to individual routes and effort levels.

Each kind of work requests models through a named **route** (`react`,
`matching.rank`, `review.tools`, and so on). Your mapping's keys select
routes — exactly, by wildcard, or qualified by context such as the turn's
effort level. The rest of this guide builds this up step by step: quick
start and provider installation, the route system, rate limits, the
built-in providers, credentials, per-agent and live configuration, custom
providers, and troubleshooting.


## Quick Start

Set the credential expected by your provider:

```bash
export OPENROUTER_API_KEY="..."
```

Then configure a provider and create an agent:

```python
import parlant.sdk as p


async with p.Server(
    models={"*": "openrouter"},  # OpenRouter is actually the default anyway
) as server:
    agent = await server.create_agent(
        id="my-agent",
        name="Support Agent",
        instructions="Help customers resolve support requests.",
    )
```

`*` is a special route: It uses this provider's registered default model for
each kind of work. During a normal turn, that can include ReAct (agentic) generation,
rule matching, embedding, review, compaction, moderation, and token estimation.

`p.Server()` defaults to `{"*": "openrouter"}` when `models` is omitted.
Declaring the provider explicitly is recommended because it makes deployment
configuration easier to inspect.

### Installing Provider Support

The base Parlant package includes both OpenAI and OpenRouter support. Gemini,
Anthropic, and LiteLLM use optional dependency groups so their SDKs are only
installed when needed:

| Provider | Installation command |
|---|---|
| OpenAI | `pip install parlant` |
| OpenRouter | `pip install parlant` |
| Gemini | `pip install 'parlant[gemini]'` |
| Anthropic | `pip install 'parlant[anthropic]'` |
| LiteLLM | `pip install 'parlant[litellm]'` |

For example, a new Gemini installation looks like this:

```python
async with p.Server(models={"*": "gemini"}) as server:
    ...
```

```bash
pip install 'parlant[gemini]'
export GEMINI_API_KEY="..."
python my_agent.py
```

OpenRouter needs no extra package after installing Parlant. Set its API
key before selecting an OpenRouter route:

```bash
pip install parlant
export OPENROUTER_API_KEY="..."
```

### Overriding a family of work

You can use a provider's defaults while overriding specific task routes:

In the following example, response generation runs on one Gemini model,
the tool-call and message reviews on a cheaper one, and every other
kind of work falls back to OpenRouter. This example already shows the whole
configuration model. The rest of this guide fills in the details.

```python
async with p.Server(
    models={
        "react": p.ModelConfig("gemini:gemini-3.1-flash-lite"),   # response generation
        "review.*": p.ModelConfig("gemini:gemini-3.5-flash-lite"),  # both review checks
        "*": "openrouter",                                        # everything else
    },
) as server:
    ...
```


## Core Concepts

### Routes

A route is an open string requested by a model-consuming component. Parlant
currently uses these routes:

| Route | Capability | Purpose |
|---|---|---|
| `react` | ReAct | Agent response steps. |
| `indexing` | Schematic | Generate rule signals and other indexes during SDK startup. |
| `matching.rank` | Schematic | Rank or prune rules for the current interaction. |
| `review.tools` | Schematic | Review proposed tool calls. |
| `review.message` | Schematic | Review a proposed message. |
| `compaction` | Schematic | Summarize older session history. |
| `embedding.rules` | Embedding | Embed rule signals and interaction context. |
| `embedding.tools` | Embedding | Embed tools for relevance selection. |
| `moderation.customer` / `moderation.agent` | Moderation | The two opt-in moderation checkpoints (see [Moderation](moderation.md)). |
| `tokenizer` | Tokenizer | Estimate token counts. |

The review routes are requested with the reviewer's boolean-verdict
`ReviewSchema`. Route names remain stable while the turn's `effort` qualifier
changes, so effort-qualified selectors (see [Route
Qualifiers](#route-qualifiers)) are the way to vary review models by effort. A
route that is not exercised by your installed engine does not cause a provider
request.

### Model Configurations

Use `ModelConfig` to bind a route to a concrete model:

```python
p.ModelConfig(
    "openai:gpt-4o",
    args={"api_key": p.SecretRef("models/openai/api-key")},
)
```

The model identifier has two parts separated by the first colon:

```text
provider:model
```

Examples:

```text
openai:gpt-5.6-luna
gemini:gemini-3.5-flash-lite
anthropic:claude-sonnet-5
openrouter:openai/gpt-5.6-terra
```

The provider prefix selects Parlant's adapter. The remaining text is passed to
that provider unchanged.

Instead of a `provider:model` string, `ModelConfig.model` may also be an
application-provided **`ModelProvider`** — for when model selection needs
to happen *dynamically*, with conditional logic that the static route table
can't express (for example, by customer attributes). It is an async
callable that receives the current `p.EngineContext` (or `None` for offline
resolution) and returns the object that will execute the operation:

```python
async def provide_react(context: p.EngineContext | None) -> ReactGenerator:
    if context and await p.Customer.current.metadata.get("tier") == "premium":
        return premium_react
    return standard_react

models={"react": p.ModelConfig(provide_react)}
```

See [Custom Model Providers](#custom-model-providers) for the full contract
(lifecycle, caching, schema binding).

The optional `args` mapping configures only provider-qualified model
strings. It is not a model-request payload. For example, `api_key` selects
credentials, `dimensions` configures an embedder, and `service_tier` picks a
provider processing tier for ReAct routes. Each provider section under
[Built-In Providers](#built-in-providers) lists exactly which arguments it
accepts per capability; anything else is rejected with a clear error. A
callable already owns its construction, so combining a callable with
non-empty `args` is invalid.

### Reasoning Level

`ModelConfig.reasoning` sets the reasoning level for ReAct and schematic routes.
Because it rides on the model route, it follows the exact same resolution as the
model itself — **per agent, and per effort qualifier** — so you control it the
same way you control which model runs.

It accepts a full `ReasoningConfig` value or a depth shorthand:

```python
p.ModelConfig("anthropic:claude-sonnet-5", reasoning="high")            # shorthand
p.ModelConfig("anthropic:claude-sonnet-5", reasoning=p.ReasoningConfig( # full control
    effort="high",        # "minimal" | "low" | "medium" | "high"
    visibility="full",    # "none" | "summary" | "full" — what reasoning is returned
))
```

For ReAct, both `effort` and `visibility` apply. Schematic generation has no
reasoning-visibility channel, so schematic routes use only `effort`. Parlant
currently consumes schematic reasoning configuration on `matching.rank`
(ranking and pruning), `review.tools`, and `review.message`.
Other capabilities (embedding/tokenizer/moderation) ignore `reasoning`.

When `reasoning` is left unset (the default), each engine component's built-in
effort→reasoning mapping is used, so existing routes are unchanged.

To vary reasoning by the agent's effort, attach it to effort-qualified routes
(see [Route Qualifiers](#route-qualifiers) below):

```python
models={
    "react": p.ModelConfig("openai:gpt-5.6-terra", reasoning="low"),
    "react[effort>=high]": p.ModelConfig("openai:gpt-5.6", reasoning="high"),
    "matching.rank": p.ModelConfig("openai:gpt-5.4-nano", reasoning="minimal"),
    "matching.rank[effort>=high]": p.ModelConfig(
        "openai:gpt-5.4-mini",
        reasoning="medium",
    ),
    "review.*": p.ModelConfig("gemini:gemini-3.5-flash-lite", reasoning="low"),
}
```

This is also settable per agent via `create_agent(models=...)`, since agent
routes override the routes configured at the server level:

```python
await server.create_agent(
    id="my-agent-2",
    name="Deep Thinker",
    effort=p.Effort.HIGH,
    models={
        "react": p.ModelConfig("anthropic:claude-sonnet-5", reasoning="high"),
        "review.*": p.ModelConfig("openai:gpt-5.4-mini", reasoning="medium"),
    },
)
```

### Exact and Wildcard Routes

Consumers always request concrete routes, such as `react` or `matching.rank`. Configuration
may use an exact route or a terminal namespace wildcard:

```python
models={
    "matching.*": p.ModelConfig("gemini:gemini-3.1-flash-lite"),
}
```

Only terminal `.*` wildcards are supported. Patterns such as `matching*`,
`*.rank`, and `review.*.tools` are invalid.

### Route Qualifiers

Qualifiers select a model using attributes of the current operation. Parlant
currently provides one qualifier, `effort`:

```python
models={
    "react": p.ModelConfig("gemini:gemini-3.1-flash-lite"),
    "react[effort>=high]": p.ModelConfig("anthropic:claude-sonnet-5"),
    "matching.*[effort<medium]": p.ModelConfig(
        "openai:gpt-5.4-nano"
    ),
}
```

The supported comparisons are `=`, `!=`, `<`, `<=`, `>`, and `>=`. They use
the following ordering:

```text
min < low < medium < high < max
```

Multiple constraints are combined with AND, so this selector matches low and medium effort:

```python
"react[effort>=low,effort<high]"
```

The effort value is resolved when each model operation begins. It is the
maximum of the agent's configured effort and any effort triggered by matched rules
(see [Adaptive Effort](adaptive-effort.md))
so far. A rule discovered during matching cannot retroactively change the model
that discovered it, but the triggered effort applies to subsequent operations.

### The Provider Route

The route `*` is the provider fallback:

```python
models={"*": "gemini"}
```

Its value must be one bare provider name. Parlant chooses that provider's
registered default model for the capability required by the concrete route.
It may carry effort qualifiers to select a provider globally:

```python
models={
    "*": "openrouter",
    "*[effort>=high]": "anthropic",
}
```

In this example, high and maximum-effort operations use Anthropic's defaults.
Other operations use OpenRouter's defaults.

Valid:

```python
{"*": "openai"}
{"*": "gemini"}
{"*": "anthropic"}
{"*": "openrouter"}
{"*[effort>=high]": "anthropic"}
```

Invalid:

```python
{"*": "openai:gpt-5.6-luna"}
{"*": "openrouter:openai/gpt-4.1"}
{"*[effort>=high]": "anthropic:claude-sonnet-5"}
{"react": "openai"}
```

Use `ModelConfig` whenever you specify a model:

```python
{
    "*": "openai",
    "react": p.ModelConfig("openai:gpt-4o"),
}
```

### Route Precedence

Agent routes take precedence over server routes. Within each scope, exact
routes take precedence over namespace wildcards, which take precedence over
the provider route. For the same route pattern, a matching qualified selector
beats its unqualified fallback. A selector matching fewer effort levels is more
specific than one matching more levels.

For `matching.distill` at high effort, resolution checks:

```text
1. agent matching.distill[matching effort constraints, narrowest first]
2. agent matching.distill
3. agent matching.*[matching effort constraints, narrowest first]
4. agent matching.*
5. agent *[matching effort constraints, narrowest first]
6. agent *
7. server matching.distill[matching effort constraints, narrowest first]
8. server matching.distill
9. server matching.*[matching effort constraints, narrowest first]
10. server matching.*
11. server *[matching effort constraints, narrowest first]
12. server *
```

This ordering means an agent's `matching.*` override intentionally beats a more
specific `matching.distill` server default. Remove the agent override when you
want the agent to inherit server configuration again.

Two selectors for the same route pattern cannot overlap with equal qualifier
specificity. For example, these are ambiguous at medium effort and are rejected:

```python
{
    "react[effort>=medium]": p.ModelConfig(...),
    "react[effort<=medium]": p.ModelConfig(...),
}
```

## Rate Limits

Providers set account-level quotas: a maximum number of concurrent
requests, and a maximum request rate. A busy server can exceed them — many
sessions processing at once, or bursts of parallel work inside one
operation. Instead of letting the provider reject requests, you can attach
a `RateLimit` to the same `ModelConfig` that selects the model, so routing
and admission control stay together. Requests beyond the limit wait
in FIFO order; the limiter never returns a fake model response and never
surfaces a rate-limit error to the agent.

```python
async with p.Server(
    models={
        "*": "openai",
        "indexing": p.ModelConfig(
            "openai:gpt-4o-mini",
            rate_limit=p.RateLimit(
                max_concurrency=4,
                requests_per_minute=60,
            ),
        ),
    },
) as server:
    ...
```

The example limits `indexing` because it is the classic burst case: startup
indexing runs several schematic generations in parallel, and every indexing
proposer requests the same route — so they share this limit even though
they use different output schemas. The same mechanism applies to any
schematic or ReAct route.

`RateLimit` has three fields:

| Field | Meaning |
|---|---|
| `max_concurrency` | Maximum admitted requests that may remain in flight. |
| `requests_per_minute` | Maximum request-start rate. Starts are paced evenly, rather than released as a one-minute burst. |
| `bucket` | Optional explicit capacity group shared by every config using the same name. |

At least one numeric constraint is required. Configure either kind separately,
or combine them:

```python
# Concurrency only: useful for protecting connection pools or provider limits.
p.RateLimit(max_concurrency=3)

# Throughput only: admit one request start every two seconds.
p.RateLimit(requests_per_minute=30)

# Both constraints must be satisfied before a request starts.
p.RateLimit(max_concurrency=3, requests_per_minute=30)
```

Without `bucket`, capacity belongs to the selected route binding. This is
usually the right behavior: `indexing` calls share one queue, while `review.*`
and `react` have independent queues. Live replacement of a route creates a new
binding and therefore a new transient bucket.

Use a named bucket when separate routes consume one provider quota:

```python
shared_openai_quota = p.RateLimit(
    max_concurrency=8,
    requests_per_minute=240,
    bucket="openai-production",
)

models = {
    "matching.*": p.ModelConfig(
        "openai:gpt-4o-mini",
        rate_limit=shared_openai_quota,
    ),
    "review.*": p.ModelConfig(
        "openai:gpt-4o",
        rate_limit=shared_openai_quota,
    ),
    "react[effort>=high]": p.ModelConfig(
        "openai:gpt-4o",
        rate_limit=shared_openai_quota,
    ),
}
```

Every use of one named bucket must specify the same constraints. Conflicting
configurations raise `RateLimitConflictError` when that bucket is acquired.

Rate limits currently wrap schematic and ReAct generators. A schematic permit
covers one `generate()` operation, including provider-internal retries or
hedges. ReAct limits are applied at the provider stream-attempt boundary, so a
TTFT hedge consumes another permit. Permits are always released after success,
failure, or cancellation. Embedding, moderation, and tokenizer routes reject
attached rate limits when resolved.

The default `TransientRateLimiter` is process-local. It coordinates every route
resolved by one server process, but it neither coordinates multiple workers nor
preserves quota state across restarts. Replace the `RateLimiter` DI port with a
persistent or distributed implementation when either property is required:

```python
from lagom import Container

from parlant.core.nlp.rate_limits import RateLimiter


async def configure_container(container: Container) -> Container:
    container[RateLimiter] = RedisRateLimiter(...)
    return container


async with p.Server(
    configure_container=configure_container,
    models={...},
) as server:
    ...
```

For durable quotas, use explicit bucket names whose identity is stable across
processes and deployments. A custom limiter implements
`acquire(bucket=..., limit=...)` and returns a permit that is safe to
release more than once.

Valid configurations:

```python
models = {
    "indexing": p.ModelConfig(
        "openai:gpt-5.6-terra",
        rate_limit=p.RateLimit(max_concurrency=4),
    ),
    "react": p.ModelConfig(
        custom_react_provider,
        rate_limit=p.RateLimit(requests_per_minute=60),
    ),
    "review.tools[effort>=high]": p.ModelConfig(
        "openrouter:anthropic/claude-sonnet-5",
        rate_limit=p.RateLimit(
            max_concurrency=2,
            requests_per_minute=20,
            bucket="openrouter-anthropic",
        ),
    ),
}
```

Invalid configurations and their behavior:

| Configuration | Result |
|---|---|
| `p.RateLimit()` | Rejected because it defines no constraint. |
| Zero, negative, boolean, non-finite, or non-numeric numeric fields | Rejected when the `RateLimit` is constructed. |
| An empty or whitespace-only `bucket` | Rejected. |
| `p.ModelConfig(provider_callable, args={...})` | Rejected because callable providers own their own construction. |
| A bare callable as a route value | Rejected; wrap it in `ModelConfig`. |
| A rate limit on embedding, moderation, or tokenizer | Rejected when that route is resolved. |
| Different limits using the same named bucket | Rejected by the limiter as a bucket conflict. |
| `"*": p.ModelConfig(...)` | Rejected; `*` remains a bare provider fallback. Configure limited concrete or namespace routes instead. |

Runtime bucket conflicts and unsupported route capabilities use the
model-routing diagnostics described under Troubleshooting.

## Built-In Providers

The built-in providers are OpenAI, Gemini, Anthropic,
OpenRouter, and LiteLLM.

| Provider | Prefix | Environment variable | Schematic | ReAct | Embedding | Moderation | Tokenizer |
|---|---|---|---|---|---|---|---|
| OpenAI | `openai` | `OPENAI_API_KEY` | Yes | Yes | Yes | Yes (explicit route only) | Yes |
| Gemini | `gemini` | `GEMINI_API_KEY` | Yes | Yes | Yes | No | Yes |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` | Yes | Yes | Voyage AI (`VOYAGE_API_KEY`) | No | Yes |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | Yes | Yes | Yes | No | Yes |
| LiteLLM | `litellm` | per-model (e.g. `OPENAI_API_KEY`) | Yes | Yes | Yes | No | Yes |

Provider adapters are loaded lazily. Starting Parlant does not import every
optional provider package. Configuring a route loads only the selected
provider; a later live route update can load another provider without restarting
the server.

If an optional provider package is missing, route configuration fails with an
actionable installation command such as `pip install 'parlant[gemini]'`. An
unused optional provider does not need to be installed. See [Installing
Provider Support](#installing-provider-support) for all installation commands;
OpenAI and OpenRouter are included in the base package.

Provider defaults are convenient, but explicit `ModelConfig` routes are
better when model selection needs to remain stable across Parlant upgrades.

### OpenAI

Minimal configuration:

```python
server = p.Server(models={"*": "openai"})
```

Explicit mixed configuration:

```python
server = p.Server(
    models={
        "*": "openai",
        "react": p.ModelConfig("openai:gpt-4o"),
        "matching.rank": p.ModelConfig("openai:gpt-5.4-nano"),
        "embedding.*": p.ModelConfig(
            "openai:text-embedding-3-small",
            args={"dimensions": 1536, "max_tokens": 8192},
        ),
        "moderation": p.ModelConfig("openai:omni-moderation-latest"),
        "tokenizer": p.ModelConfig("openai:gpt-4o"),
    }
)
```

Accepted constructor arguments:

| Capability | Arguments |
|---|---|
| Schematic | `api_key`, `tokenizer_model_name` |
| ReAct | `api_key`, `service_tier` |
| Embedding | `api_key`, `dimensions`, `max_tokens` |
| Moderation | `api_key` |
| Tokenizer | None |

OpenAI embedding dimensions default to known dimensions for the standard
embedding models and otherwise default to 1536. Set `dimensions` explicitly for
other models.

`service_tier` selects the processing tier for every ReAct request on the
route: `"standard"` (the default), `"flex"` (cheaper, slower), or
`"priority"` (faster, at a premium — OpenAI's own name for this tier is
`fast`, and Parlant maps it for you):

```python
"react": p.ModelConfig("openai:gpt-5.6-terra", args={"service_tier": "priority"}),
```

### Gemini

```python
server = p.Server(
    models={
        "*": "gemini",
        "react[effort>=high]": p.ModelConfig("gemini:gemini-3.5-flash"),
        "embedding.*": p.ModelConfig(
            "gemini:gemini-embedding-2",
            args={"dimensions": 3072, "max_tokens": 8192},
        ),
    }
)
```

The default embedding model is `gemini-embedding-2`. Stored embeddings are kept
per embedder, so switching models re-embeds your documents on the next start.

Accepted constructor arguments:

| Capability | Arguments |
|---|---|
| Schematic | `api_key` |
| ReAct | `api_key`, `service_tier` |
| Embedding | `api_key`, `dimensions`, `max_tokens` |
| Moderation | Not offered; route moderation to a provider that has it. |
| Tokenizer | `api_key` |

`service_tier` (`"standard"` / `"flex"` / `"priority"`) maps directly onto
Gemini's tiers of the same names, for example
`args={"service_tier": "flex"}`.

### Anthropic

Anthropic does not provide an embedding API, so Parlant's Anthropic provider
fulfills embedding routes through [Voyage AI](https://platform.claude.com/docs/en/build-with-claude/embeddings),
Anthropic's recommended embeddings partner (default model `voyage-3.5`). With
both `ANTHROPIC_API_KEY` and `VOYAGE_API_KEY` set, this is a complete
configuration:

```python
server = p.Server(models={"*": "anthropic"})
```

You may route embeddings to another provider instead — then no Voyage key is
needed:

```python
server = p.Server(
    models={
        "*": "anthropic",
        "embedding.*": p.ModelConfig("openai:text-embedding-3-large"),
    }
)
```

Accepted constructor arguments:

| Capability | Arguments |
|---|---|
| Schematic | `api_key` |
| ReAct | `api_key`, `max_tokens`, `service_tier` |
| Embedding | `api_key` (a Voyage AI key; defaults to `VOYAGE_API_KEY`) |
| Moderation | Not offered; route moderation to a provider that has it. |
| Tokenizer | `api_key` |

`service_tier` accepts the same neutral values as the other providers
(`"standard"` / `"flex"` / `"priority"`) and maps them onto what Anthropic
offers: `"priority"` becomes Anthropic's `auto` (priority capacity when
available), and `"flex"` has no Anthropic equivalent, so it runs as
standard.

You can use Gemini or OpenRouter for `embedding.*` instead of OpenAI.

### OpenRouter

OpenRouter model identifiers contain their own provider/model path. Parlant
splits only at the first colon:

```python
p.ModelConfig("openrouter:openai/gpt-4o")
p.ModelConfig("openrouter:google/gemini-3.5-flash")
p.ModelConfig("openrouter:anthropic/claude-sonnet-5")
```

For `openrouter:google/gemini-3.5-flash`, Parlant selects the OpenRouter adapter
and sends `google/gemini-3.5-flash` to OpenRouter. It does not try to select the
local Gemini adapter a second time.

Example:

```python
server = p.Server(
    models={
        "*": "openrouter",
        "react": p.ModelConfig(
            "openrouter:anthropic/claude-sonnet-5",
            args={
                "site_url": "https://support.example.com",
                "app_name": "Example Support",
                "provider": {
                    "sort": "latency",
                    "preferred_max_latency": {"p90": 2.0},
                    "allow_fallbacks": True,
                },
            },
        ),
        "matching.*": p.ModelConfig(
            "openrouter:google/gemini-3.5-flash"
        ),
        "embedding.*": p.ModelConfig(
            "openrouter:openai/text-embedding-3-large"
        ),
    }
)
```

Accepted constructor arguments:

| Capability | Arguments |
|---|---|
| Schematic | `api_key`, `site_url`, `app_name`, `provider` |
| ReAct | `api_key`, `site_url`, `app_name`, `provider` |
| Embedding | `api_key`, `site_url`, `app_name`, `provider` |
| Moderation | Not offered; route moderation to a provider that has it. |
| Tokenizer | None |

**The `provider` mapping — OpenRouter's routing preferences.** One model id
on OpenRouter is usually served by several backend providers, and the
`provider` mapping controls which of them may serve your requests, and in
what order. Parlant passes the mapping to OpenRouter verbatim on every
inference request, so every field from
[OpenRouter's provider-routing schema](https://openrouter.ai/docs/features/provider-routing)
works with no Parlant-specific translation. The fields you will reach for
most:

| Field | What it does |
|---|---|
| `sort` | Rank candidate backends by `"price"`, `"throughput"`, or `"latency"`. |
| `order` | An explicit list of provider slugs to try, in order. |
| `only` / `ignore` | Allow-list or deny-list of provider slugs. |
| `allow_fallbacks` | Whether OpenRouter may fall back past your `order` list (default `True`). |
| `require_parameters` | Only route to backends that support every request parameter (important for structured output). |
| `max_price` | Price ceiling, e.g. `{"prompt": 1.0, "completion": 4.0}` (USD per million tokens). |
| `quantizations` | Acceptable quantization levels for open-weight models, e.g. `["fp16", "bf16"]`. |
| `data_collection` | `"deny"` to route only to backends that don't retain or train on your prompts. |
| `zdr` | `True` to require zero-data-retention backends. |

Two worked configurations. First, a compliance-focused route: only named
backends, no silent fallback to anyone else, and no data retention:

```python
"react": p.ModelConfig(
    "openrouter:anthropic/claude-sonnet-5",
    args={
        "provider": {
            "order": ["anthropic", "google-vertex"],
            "allow_fallbacks": False,
            "data_collection": "deny",
        },
    },
),
```

Second, a cost-guarded route for high-volume background work: cheapest
backend first, with a hard price ceiling and full-precision weights only:

```python
"matching.*": p.ModelConfig(
    "openrouter:meta-llama/llama-3.3-70b-instruct",
    args={
        "provider": {
            "sort": "price",
            "max_price": {"prompt": 0.5, "completion": 1.0},
            "quantizations": ["fp16", "bf16"],
            "require_parameters": True,
        },
    },
),
```

Because `provider` is part of the route's `args`, different routes can carry
different preferences — a strict pinned configuration for `react` while
`matching.*` shops for price, as above.

**Cost reporting.** OpenRouter's authoritative USD charge is recorded for
schematic, ReAct, and embedding requests as a fixed-point decimal
string in `usage.extra["reported_cost"]`. Usage is keyed
`openrouter:<model path>` (for example `openrouter:anthropic/claude-sonnet-5`),
so a session's `per_provider` costs show what OpenRouter charged as a whole.
A trace made entirely of provider-priced operations gets a cost report
without calling `server.set_pricing()`. Configure an explicit USD price table when a trace also
uses providers or operations that do not report their own charge. Creating a
session budget still requires `server.set_pricing()`; an empty USD table is
enough when every budgeted operation supplies an authoritative cost.

OpenRouter schematic generation sends the requested Pydantic model as a strict
JSON Schema structured-output format. If a backend rejects structured output,
the adapter retries with JSON object mode and then an explicit JSON-only prompt.
The working mode is cached for the process lifetime for each model,
provider-routing configuration, and output schema, so later requests do not pay
the fallback latency again.
Set `provider.require_parameters` to `True` when OpenRouter should only route to
backends that support every requested parameter, including structured output.

OpenRouter ReAct generation uses a provider-portable representation for
mid-conversation system instructions. Application code does not need to detect
whether the model behind OpenRouter supports native mid-conversation system
messages.

### LiteLLM

LiteLLM is a unified gateway over 100+ providers. Unlike OpenRouter (an HTTP
gateway), Parlant's LiteLLM provider drives the **LiteLLM Python SDK** directly
(`litellm.acompletion` / `litellm.aembedding`), so a single provider reaches any
model LiteLLM supports. Install it with `pip install 'parlant[litellm]'`.

The model identifier after the colon is passed to LiteLLM verbatim, so it uses
LiteLLM's own naming (`provider/model`):

```python
p.ModelConfig("litellm:openai/gpt-4o")
p.ModelConfig("litellm:anthropic/claude-sonnet-5")
p.ModelConfig("litellm:gemini/gemini-3.1-flash-lite")
p.ModelConfig("litellm:bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0")
```

Because LiteLLM routes to whichever underlying provider each model names, there
is no single `LITELLM_API_KEY`: set the key(s) for the providers your models use
(e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`). LiteLLM validates
credentials at call time.

Example:

```python
server = p.Server(
    models={
        "*": "litellm",
        "react": p.ModelConfig("litellm:anthropic/claude-sonnet-5"),
        "matching.*": p.ModelConfig("litellm:gemini/gemini-3.1-flash-lite"),
        "embedding.*": p.ModelConfig("litellm:openai/text-embedding-3-large"),
    }
)
```

Accepted constructor arguments:

| Capability | Arguments |
|---|---|
| Schematic | `api_key` |
| ReAct | `api_key` |
| Embedding | `api_key`, `dimensions` |
| Moderation | Not offered; route moderation to a provider that has it. |
| Tokenizer | None |

**Cost reporting.** LiteLLM prices each response it can, and the adapter surfaces
that figure as the authoritative cost for usage/cost reports, budgets, and
cost-control — no Parlant-side pricing table is needed for priced models. If
LiteLLM cannot price a model (it isn't in LiteLLM's cost map), Parlant falls back
to its own token-derived pricing for that route, exactly as for other providers;
the table entry is then `litellm:<model>` (say `"litellm:anthropic/claude-sonnet-5"`)
or the bare model path.

## Credentials and Secrets

### Environment Variables

When `api_key` is omitted, built-in adapters use their standard environment
variable:

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export OPENROUTER_API_KEY="..."
```

This is the simplest server-wide setup. It is insufficient when different
agents need different credentials for the same provider.

### Durable Secret References

`SecretRef` stores a locator, not a credential. Your application supplies a
`SecretLoader` that resolves the locator from its existing secret system,
bound in the container (see
[Dependency Injection](dependency-injection.md)):

```python
import parlant.sdk as p


class VaultSecretLoader(p.SecretLoader):
    async def load(self, ref: p.SecretRef) -> str:
        return await MY_VAULT.read(ref.key)


async def configure_container(container: p.Container) -> p.Container:
    container[p.SecretLoader] = VaultSecretLoader()
    return container


server = p.Server(
    configure_container=configure_container,
    models={"*": "openai"},
)

agent = await server.create_agent(
    id="my-agent-3",
    name="Claims Agent",
    instructions="Help customers manage claims.",
    models={
        "react": p.ModelConfig(
            "openai:gpt-4o",
            args={"api_key": p.SecretRef("agents/claims/openai")},
        )
    },
)
```

Parlant persists the `agents/claims/openai` key, not the loaded key.
The loader is called lazily when the selected implementation is constructed.

For local development and tests:

```python
async def configure_container(container: p.Container) -> p.Container:
    container[p.SecretLoader] = p.MappingSecretLoader(
        {"agents/claims/openai": "local-development-key"}
    )
    return container


server = p.Server(
    configure_container=configure_container,
    models={"*": "openai"},
)
```

`MappingSecretLoader` keeps values in process memory. It is not a production
secret store.

### Transient Secret Values

Use `SecretValue` when application code already has a plaintext credential and
does not want Parlant to persist it:

```python
import os


agent = await server.create_agent(
    id="my-agent-4",
    name="Private Agent",
    instructions="...",
    models={
        "react": p.ModelConfig(
            "gemini:gemini-3.1-flash-lite",
            args={"api_key": p.SecretValue(os.environ["PRIVATE_GEMINI_KEY"])},
        )
    },
)
```

Any agent route containing a `SecretValue`, including a nested one, is kept in
the process-local route source. It disappears on restart and must be registered
again by application code.

Server route bindings are process-local regardless, because they are startup
configuration rather than agent data.

### Do Not Use Ordinary Strings for Agent Credentials

Ordinary argument values in an agent route are durable configuration. This is
correct for values such as `site_url`, `app_name`, `dimensions`, and
`max_tokens`, but not for plaintext credentials.

Avoid:

```python
args={"api_key": "plaintext-key"}
```

Use `SecretRef` or `SecretValue` instead: Parlant cannot infer whether an
arbitrary string is sensitive.

Object representations and model-routing diagnostics redact constructor
argument values.

## Per-Agent Configuration

Server routes are defaults. Agent routes let agents use different models or
credentials without separate Parlant processes.

```python
server = p.Server(
    models={
        "*": "anthropic",
        "embedding.*": p.ModelConfig("openai:text-embedding-3-large"),
    }
)

fast_agent = await server.create_agent(
    id="my-agent-5",
    name="Fast Agent",
    instructions="Resolve simple requests quickly.",
    models={
        "react": p.ModelConfig("gemini:gemini-3.1-flash-lite"),
        "review.*": p.ModelConfig("gemini:gemini-3.5-flash-lite"),
    },
)

quality_agent = await server.create_agent(
    id="my-agent-6",
    name="Quality Agent",
    instructions="Handle complex requests carefully.",
    models={
        "react": p.ModelConfig("anthropic:claude-sonnet-5"),
        "review.*": p.ModelConfig("anthropic:claude-opus-5"),
    },
)
```

Both agents inherit the server's OpenAI embedding route. Their ReAct and review
routes differ.

Agents can also use the same model with separate credentials:

```python
agent_a = await server.create_agent(
    id="my-agent-7",
    name="Tenant A",
    instructions="...",
    models={
        "react": p.ModelConfig(
            "openai:gpt-4o",
            args={"api_key": p.SecretRef("tenants/a/openai")},
        )
    },
)

agent_b = await server.create_agent(
    id="my-agent-8",
    name="Tenant B",
    instructions="...",
    models={
        "react": p.ModelConfig(
            "openai:gpt-4o",
            args={"api_key": p.SecretRef("tenants/b/openai")},
        )
    },
)
```

Model instances are cached by provider, model, sanitized argument identity,
capability, and schematic schema. Different credential references therefore do
not share a generator instance.

## Live Model Changes

Route lookup happens immediately before each model operation. Updating an agent
does not require restarting the server, agent, or session.

```python
await agent.update(
    models={
        "react[effort>=high]": p.ModelConfig("openai:gpt-4o"),
    }
)
```

The updated `agent.models` view is available after the call.

Remove an override by assigning `None`:

```python
await agent.update(
    models={
        "react[effort>=high]": None,
        "review.*": None,
    }
)
```

The next resolution falls back through the remaining agent routes and then the
server routes.

An already-started provider request finishes on its original model. The next
model operation sees the new route, including a later operation in the same
turn.

## Custom Model Providers

A `ModelProvider` is an async callable that receives the current
`EngineContext`, or `None` for an offline resolution, and returns the object that
will execute the routed operation.

```python
import parlant.sdk as p
from parlant.core.nlp.react import ReactGenerator


shared_react = MyReactGenerator(...)


async def provide_react(
    context: p.EngineContext | None,
) -> ReactGenerator:
    return shared_react


agent = await server.create_agent(
    id="my-agent-9",
    name="Custom Agent",
    instructions="...",
    models={"react": p.ModelConfig(provide_react)},
)
```

**When it runs:** the callable is the route's resolution step. Parlant
resolves routes lazily, immediately before each model operation. So the
callable runs every time the route serves an operation — possibly several
times per turn — and never just once at configuration time. Unlike
`provider:model` strings (whose constructed instances are cached), the
callable's result is used as returned, every time. That is what makes
per-request selection possible. It also runs with `context=None` for
offline resolutions, such as validating moderation routes at startup.

Two practical consequences: keep the callable cheap and free of side
effects (it is on the hot path), and have it *select* a prebuilt object
rather than construct a new one. Parlant does not cache or close what the
callable returns, so the application owns concurrency safety, cancellation
behavior, and resource lifecycle.

The provider can select an object from current context:

```python
async def provide_react(context: p.EngineContext | None) -> ReactGenerator:
    if context is None:
        return default_react

    tenant = tenant_for_customer(p.Customer.current.id)
    return react_by_tenant[tenant]
```

This allows application-specific routing by agent, session, customer, tenant,
feature flag, or any other state your application owns.

The callable does not receive route qualifiers as an argument, but it doesn't
need to: they derive from the same turn — its effort, for example, is
`p.Session.current.effort`. (For plain by-effort model
selection, prefer effort-qualified route keys; reach for a provider callable
when the logic goes beyond what selectors express.)

### Provider Identity in Usage Reports

Usage and cost records are keyed by `provider:model`, and the provider part
comes from the object the callable returns: its `provider_name`. A built-in
adapter reports under its own prefix (`openai`, `anthropic`, ...), so the
per-tenant example above would file every tenant's spend under the same
provider. When the objects differ in a way you want to see apart — two
accounts for the same vendor, two regions, two tenants — name them when you
build them:

```python
from parlant.adapters.nlp.anthropic_service import AnthropicReactGenerator

react_by_tenant = {
    "acme": AnthropicReactGenerator(
        model="claude-sonnet-5-5",
        logger=logger,
        api_key=acme_key,
        provider_name="anthropic-acme",
    ),
    "globex": AnthropicReactGenerator(
        model="claude-sonnet-5-5",
        logger=logger,
        api_key=globex_key,
        provider_name="anthropic-globex",
    ),
}
```

Every built-in schematic generator, embedder, ReAct generator, and decision
maker accepts `provider_name`. A custom adapter that implements a port itself
needs a `provider_name` property when it is a ReAct generator (the port
requires one); schematic generators and embedders inherit it from
`BaseSchematicGenerator` and `BaseEmbedder`, which take `provider_name` in
their constructor. The id is free-form and must not contain a colon. Price
each name with a `provider:model` entry in the price table, or let a bare
model entry cover them all; [Usage Reports](usage-reports.md#price-table-keys)
has the rules.

### Schematic Providers Are Schema-Bound

Schematic consumers request a specific output schema. A custom provider must
return a generator bound to that exact schema:

```python
async def provide_ranker(
    context: p.EngineContext | None,
) -> p.SchematicGenerator:
    # This instance's schema must be the RuleRankSchema requested by
    # matching.rank.
    return rule_rank_generator


models={"matching.rank": p.ModelConfig(provide_ranker)}
```

A single callback behind `review.*` is valid only if the returned generator is
compatible with every schema requested through that pattern. Exact schematic
routes are usually clearer and safer for custom providers.

Parlant validates both the returned capability and the schematic schema. A
React route returning a schematic generator, or a review route returning a
generator for the wrong schema, fails before invocation.

### Custom Tokenizers

Token estimation is also routed:

```python
async def provide_tokenizer(
    context: p.EngineContext | None,
) -> p.EstimatingTokenizer:
    return MyEstimatingTokenizer(...)


agent = await server.create_agent(
    id="my-agent-10",
    name="Custom Tokenizer Agent",
    instructions="...",
    models={"tokenizer": p.ModelConfig(provide_tokenizer)},
)
```

Parlant resolves this route when it needs estimation rather than
receiving one process-global tokenizer.

### Provider-Owned Caching

A model provider may choose any lifecycle appropriate for the application:

```python
# One shared instance.
async def shared_provider(context):
    return shared_generator


# One instance per tenant.
async def tenant_provider(context):
    tenant_id = tenant_for(context)
    return generators_by_tenant[tenant_id]


# A fresh instance for every routed operation.
async def fresh_provider(context):
    return MyReactGenerator(...)
```

Callbacks are process-local SDK bindings. Register them again after restart.

## Troubleshooting

### No Model Route Can Satisfy an Operation

Example:

```text
No model route for agent 'claims-agent' can satisfy 'review.tools'
(schematic). Checked: review.tools, review.*, *
```

Configure an exact route, a matching wildcard, or `*` at the agent or server
scope.

### Unknown Provider

```text
Unknown model provider 'gemni'. Did you mean 'gemini'?
```

Use one of `openai`, `gemini`, `anthropic`, or `openrouter` before the first
colon.

### Unsupported Capability

```text
Provider route '*' selected provider 'custom-provider' for route 'embedding.rules',
but provider 'custom-provider' has no default embedding model
```

Add an explicit embedding route using OpenAI, Gemini, Anthropic (Voyage AI),
or OpenRouter.

### Provider Package Is Not Installed

```text
Gemini model support is not installed. Install it with: pip install 'parlant[gemini]'
```

Install the reported extra or change the route to a provider available in the
deployment. Parlant loads provider adapters on demand, so this error is raised
when the provider is first configured or resolved rather than when unrelated
providers start.

### Invalid Provider Route

```text
Provider route '*' must contain one bare provider name
```

Move the concrete model onto a task route:

```python
models={
    "*": "openrouter",
    "react": p.ModelConfig("openrouter:openai/gpt-4o"),
}
```

### Invalid Wildcard

```text
Invalid model route 'react*'; only terminal namespace wildcards and the global
provider route are supported
```

Use `react` or a qualified selector such as `react[effort=low]`.

### Invalid Qualifier

```text
Unknown effort level 'medum'. Did you mean 'medium'?
```

Qualifier names, operators, and values are validated when routes are
configured. Currently `effort` is the only supported qualifier.

### Ambiguous Qualified Routes

```text
Model route selectors 'react[effort>=medium]' and
'react[effort<=medium]' are ambiguous because they overlap with equal
specificity
```

Make one selector narrower, remove the overlap, or add an exact selector for
the overlapping effort level. Route selection never depends on mapping order.

### Misspelled or Reserved Arguments

```text
Unsupported openai model argument(s): api_keey. Did you mean 'api_key'?
```

Framework-owned arguments such as `logger`, `tracer`, `meter`,
`usage_reporter`, and `provider_name` cannot be supplied through
`ModelConfig.args`; a routed model always reports under its provider prefix.

### Secret Reference Cannot Be Loaded

Check that:

1. A `SecretLoader` is bound in the container (via `configure_container`).
2. The reference exists in the application's secret system.
3. The loader is available in every process that may construct the model.
4. The selected route actually uses the expected reference.

Do not replace a failing `SecretRef` with a plaintext agent argument.

### Wrong Custom Provider Result

If a callback returns the wrong object type or schematic schema, configure it on
the correct route or return an implementation matching that route's capability.
Exact routes are preferable while diagnosing custom schematic providers.

### Invalid Rate Limit

Construction rejects missing, non-positive, boolean, and
non-finite constraints. If a request stalls in admission, check whether the
selected route shares a named bucket with another high-volume route.

`RateLimitConflictError` means two selected configs use the same explicit
bucket name with different constraints. Define the `RateLimit` once and reuse
it, or give the independent quotas different bucket names.

`UnsupportedRateLimitCapabilityError` means a rate-limited config was selected
for embedding, moderation, or tokenization. Move the limit to a schematic or
ReAct route, or enforce that provider quota in a custom adapter.
