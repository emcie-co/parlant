# Usage Reporting and Cost Visibility

Parlant answers four cost questions, each with its own layer — raw accounting
is always on, and each further layer is opt-in on top of the previous one:

| Question | Layer |
|---|---|
| How many requests and tokens did each model use? | Usage records (always on) |
| What did that usage cost? | [Pricing](#pricing) |
| Has a session crossed a limit? | [Session budgets](#session-budgets) |
| Should chargeable engine work proceed? | [Cost control](#cost-control) |

## What gets recorded

Every generation and embedding request reports its usage: a request count,
input tokens, cached input tokens, output tokens, and optional
provider-specific counters (e.g. reasoning tokens, cache-write meters).
Parlant aggregates these per turn, by provider and model, and persists the
record with the session once the whole turn has finished.

Usage is keyed by `provider:model` — the same spelling as a model route, such
as `anthropic:claude-sonnet-5-5` or `openrouter:openai/gpt-5.6-terra`. The
provider part is the id the adapter reports under: a built-in provider's
prefix (`openai`, `gemini`, `anthropic`, `openrouter`, `litellm`), or
whatever you named a hand-built adapter (see [Naming a
provider](#naming-a-provider)). Keeping the provider in the key matters
because the same model can be reached through more than one provider at
different rates: `openai:gpt-5.5` and `openrouter:openai/gpt-5.5` are
counted, priced, and reported apart.

One subtlety: to protect responsiveness, Parlant sometimes *hedges* a stalling
model request — it fires an identical duplicate and uses whichever finishes first.
Both attempts are billable, so both are counted: the losing attempt is
allowed to finish and its tokens are included before the turn record is
persisted. Expect a hedged turn's usage to be higher than its visible
output alone would suggest.

## Pricing

Usage records count tokens; to turn tokens into cost, Parlant needs prices.
Give the server a price table (per one million usage units), and every later
turn's record is priced:

```python
from decimal import Decimal
import parlant.sdk as p

pricing = await server.set_pricing(
    {
        "gemini-3.1-flash-lite": p.ModelPricing(
            input_tokens=Decimal("0.10"),
            cached_input_tokens=Decimal("0.025"),
            output_tokens=Decimal("0.40"),
        ),
    },
    currency="USD",  # Default
)
```

Pricing activates cost reporting immediately; no budget is required. Replacing
the pricing affects future turns only — a turn in flight keeps the snapshot it
started with. `ModelPricing.extra` prices provider-specific meters (Anthropic
cache writes, Gemini cache storage, separately billed reasoning tokens).

### Price table keys

A price table entry is keyed one of two ways:

- **A bare model id** (`"gemini-3.1-flash-lite"`) prices that model under
  *any* provider. This is the common case: one rate per model, however it
  is reached.
- **A `provider:model` key** (`"azure-eu:gpt-5.5"`) prices that model when
  served by that one provider. When both forms are present, the exact
  `provider:model` entry wins for that provider, and the bare entry covers
  every other provider.

The model part of the key is spelled exactly as the provider reports it. For
the built-in providers that is the model you route to (`gpt-5.5`,
`claude-sonnet-5-5`); for gateways it includes the gateway's own path
(`openrouter:openai/gpt-5.5`). If a turn uses a model with no matching entry,
the turn is left unpriced and the error names the full `provider:model` key.

A realistic table for a deployment that reaches one model directly and
through a second, more expensive, account:

```python
pricing = await server.set_pricing(
    {
        # The model's rate wherever it is served...
        "gpt-5.5": p.ModelPricing(
            input_tokens=Decimal("1.25"),
            cached_input_tokens=Decimal("0.125"),
            output_tokens=Decimal("10"),
        ),
        # ...except through this provider, which charges more.
        "azure-eu:gpt-5.5": p.ModelPricing(
            input_tokens=Decimal("1.50"),
            cached_input_tokens=Decimal("0.15"),
            output_tokens=Decimal("12"),
        ),
    },
)
```

A provider-reported charge (OpenRouter, LiteLLM) always wins over the table,
so those operations need no entry at all; see [the next
section](#provider-reported-charges).

### Provider-reported charges

Some providers price usage themselves: OpenRouter and LiteLLM report their
authoritative USD charge with each request, so a turn made up entirely of
their operations gets a cost report even without a price table — and, as the
next sections show, [session budgets](#session-budgets) work without one too.
Their usage is still keyed and broken down by provider like everyone else's
(`openrouter:anthropic/claude-sonnet-5`), so a per-provider report shows what
each gateway charged.

### Checking pricing coverage before a conversation

At startup Parlant warns once if configured models have no price table entry
and do not report their own charge. Missing prices do not make the server
unhealthy or stop conversations; their usage remains tokens only.

The authenticated admin `GET /healthz` response and the `system.health`
tunnel response include a live `pricing` section:

```json
{
  "price_table": true,
  "currency": "USD",
  "models": [
    {"key": "gemini:gemini-3.8-flash", "price": "table", "source": "configured"},
    {"key": "openrouter:anthropic/claude-sonnet-5", "price": "provider_reported", "source": "configured"},
    {"key": "openai:text-embedding-3-small", "price": "missing", "source": "observed"}
  ]
}
```

`price_table` means `server.set_pricing(...)` was called; without a table,
`currency` is `null`. Each key appears once. Explicit provider/model routes
are visible before usage; provider defaults are resolved without invoking
models. Opaque custom factories appear only after they report usage.
`observed` keys remain known until this process restarts. Secret values and
model configuration objects are never included.

Coverage is recomputed on each admin request, including route and table
changes. It is separate from health checks and is absent from public health
responses. The Dashboard CLI displays the same missing-price guidance after
local project checks/confirmation and successful managed deployments. Older
servers without coverage remain supported. Adding prices affects future
turns only; it does not re-price historic records.

### Naming a provider

Built-in adapters report under their provider prefix. When you build an
adapter yourself — for a [custom model provider](providers.md#custom-model-providers),
or to reach one vendor through two accounts — give it its own provider id with
the `provider_name` argument. Every built-in generator, embedder, ReAct
generator, and decision maker accepts it:

```python
from parlant.adapters.nlp.openai_service import OpenAIReactGenerator

primary = OpenAIReactGenerator(model="gpt-5.5", logger=logger)
eu_region = OpenAIReactGenerator(
    model="gpt-5.5",
    logger=logger,
    api_key=eu_api_key,
    provider_name="openai-eu",
)
```

Usage from `primary` is keyed `openai:gpt-5.5`; usage from `eu_region` is
keyed `openai-eu:gpt-5.5`, which you can then price on its own
(`"openai-eu:gpt-5.5"` in the table) and filter on in reports. The id is
free-form, but it cannot contain a colon, which separates it from the model.

## Reading usage from the Session API

`GET /sessions/{id}` returns a `usage` mapping keyed by trace ID (one entry
per agent turn):

```json
{
  "usage": {
    "<trace_id>": {
      "timestamp_utc": "2026-07-16T12:34:56Z",
      "requests": {
        "gemini:gemini-3.1-flash-lite": {
          "request_count": 3,
          "input_tokens": 48210,
          "output_tokens": 512,
          "cached_input_tokens": 45800,
          "extra": {"reasoning_tokens": 128}
        },
        "openrouter:anthropic/claude-sonnet-5": {
          "request_count": 1,
          "input_tokens": 9120,
          "output_tokens": 240,
          "cached_input_tokens": 0,
          "extra": {"reported_cost": "0.030960"}
        }
      },
      "costs": {
        "currency": "USD",
        "total": "0.032385",
        "per_model": {
          "gemini:gemini-3.1-flash-lite": "0.001425",
          "openrouter:anthropic/claude-sonnet-5": "0.030960"
        },
        "per_provider": {
          "gemini": "0.001425",
          "openrouter": "0.030960"
        }
      }
    }
  },
  "costs": {
    "currency": "USD",
    "total": "0.032385",
    "per_model": { "...": "summed across every turn" },
    "per_provider": { "...": "summed across every turn" }
  }
}
```

`requests` is always present, keyed by `provider:model`. `costs` is present
when the turn could be priced — through your price table, provider-reported
charges, or both — and `null` when no pricing covered the turn completely.
`per_model` uses the same keys as `requests`; `per_provider` sums them by the
provider part, so a UI can show spend per vendor without parsing model ids.
Decimal costs are fixed-point strings, preserving exact values without
exponent notation.

The top-level `costs` is the whole session: every turn's costs summed, by
model and by provider, or `null` if any turn is unpriced or turns disagree on
currency. `per_provider` is `null` on records written before usage carried a
provider; everything else about such records reads as before.

Session listings omit `usage`; retrieve a specific session for its full trace
history.

## Session budgets

A budget turns cost into a limit: it measures the session's spend and lets
you act when a threshold is crossed. It draws on the same costs the previous
section established — provider-reported charges and/or your price table — so
it works with whichever of those you have:

```python
budget = await server.create_session_budget(
    name="conversation-cap",
    value=Decimal("1.50"),          # in your pricing's currency; USD if no price table
    # window=timedelta(hours=24),   # optional sliding window
)
```

A budget doesn't act on its own — it's an instrument you consult from code
that runs *inside* a conversation context, such as an observation's `on_matched`
callback or a custom evaluator, where a current session exists:

- `await budget.measure()` — the current session's spend so far (within the
  sliding window, if one is set).
- `await budget.reset()` — start this session's measurement over from zero,
  e.g. after a human operator resolved the conversation and handed it back.

The tutorial below shows the standard pattern end to end.

This is safe by construction: if a turn ever includes an operation whose cost
is *not* provider-reported (and not covered by a price table), measuring the
budget raises a `PricingConfigurationError` naming the model — it never
silently under-counts.

The budget adopts the server's effective pricing currency: your price
table's currency if you set one, and otherwise USD — the denomination of
provider-reported charges. (`pricing.create_session_budget` is an equivalent
way to create one from the pricing object directly.)

### Tutorial: hand off to a human when the budget is crossed

The standard way to enforce a budget is an **observation** with an
`on_matched` callback. The observation's applicability is decided by the
budget itself: `budget.exceeded` is a deterministic code evaluator, so no
model judgment is involved. The callback switches the session to manual
mode, where the agent stops responding and a human operator takes over:

```python
import parlant.sdk as p
from decimal import Decimal

budget = await server.create_session_budget(
    name="handoff-cap",
    value=Decimal("2.00"),
)

async def handoff(context: p.EngineContext, match: p.RuleEvaluation) -> None:
    # The frontend (UI) can read the session mode and render
    # it differently — say, with a special status message so
    # that customers are aware the AI agent is no longer involved.
    await p.Session.current.update(mode="manual")

    # A verbatim farewell, posted directly into the session — no model involved:
    await p.Session.current.emit_message(
        "This conversation is being transferred to a human agent. One moment."
    )

    # Alert whatever system routes work to your human operators —
    # a ticketing queue, Slack, a pager — so someone actually picks it up:
    await support_desk.create_ticket(
        session_id=p.Session.current.id,
        customer_id=p.Customer.current.id,
        reason="Session budget exceeded — please take over",
    )

# Run the `handoff` function whenever a session's budget is exceeded
await agent.create_observation(
    evaluation_func=budget.exceeded,
    on_matched=handoff,
)
```

Budget observations are evaluated at the very start of a turn, before any
model work. On the first turn where the session's spend has crossed the
budget, the observation matches, `handoff` runs, and the turn stops cleanly —
**the agent generates nothing**. The customer sees exactly what the
handler emitted: `Session.emit_message` posts your hardcoded farewell
verbatim, with no model involved. The mode switch keeps the agent silent
from there on. The external alert is what brings a real human into the
conversation.

(By the way, `emit_message` has two siblings — `emit_status` and `emit_custom` —
for posting status updates and frontend-specific events the same way, and it
accepts `source=p.EventSource.HUMAN_AGENT` for messages your operator
tooling sends into a manual session.)

One more refinement:

- **Per-day rather than per-conversation caps.**
  You can pass `window=timedelta(hours=24)` so the measured spend slides, and call
  `budget.reset()` from your operator tooling when a human closes out the session.

### The other direction: while the budget still holds

`budget.not_exceeded` is the mirror image of `budget.exceeded` — it matches for
as long as the session has budget left. Use it to gate behavior that is only
affordable while there is room to spend, rather than to react once there isn't:

```python
within_budget = await agent.create_observation(
    evaluation_func=budget.not_exceeded,
)

deep_research = await agent.create_policy(
    content=(
        "When the customer asks a question the knowledge base does not answer, "
        "research it with `search_literature` before replying"
    ),
    tools=[search_literature],
)

await deep_research.depend_on(within_budget)
```

The expensive policy is now considered only on turns where the session can
still afford it. When the spend crosses the budget, the observation stops
matching, the policy goes quiet, and the agent carries on with everything else
— no handoff, no interruption, just the costly option withdrawn.

Both evaluators read the same boundary, so on any turn exactly one of them
matches, and spending *exactly* the budget counts as not exceeding it. Like
`exceeded`, `not_exceeded` is deterministic code rather than model judgment,
and Parlant schedules it at the very start of the turn — so the decision is
made before the turn spends anything.

## Programmatic observation

You can hook a listener to get continuous usage updates.
`UsageReporter.add_listener` receives every provider report as it happens,
with the `provider:model` key the usage is accounted under:

```python
container[UsageReporter].add_listener(
    lambda trace_id, key, usage: metrics.increment(key, usage.input_tokens)
)
```

Note that listener exceptions are swallowed because reporting occurs on
the model hot path. So keep listeners fast and queue up any expensive work.

## Cost control

`CostControlPolicy` is the low-level admission port used at TURN, STEP, and
BACKGROUND boundaries. It does not perform token pricing; it decides whether
work may proceed at each boundary, and you can implement and register your own
policy through the DI container.
