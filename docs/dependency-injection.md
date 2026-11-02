# Configuring the Container (Dependency Injection)

Parlant is a unique harness, because it's built to offer flexibility for
engineers. Almost any part of a Parlant server can be replaced with your
own implementation. You can route logs into your observability stack, keep
sessions in your database, enforce your own authorization rules, or tune
Parlant's cost and latency trade-offs. None of this requires forking or
patching. You write a class, bind it in one place, and the whole stack
picks it up.

The mechanism behind this is Parlant's dependency-injection container: every
component — engines, stores, policies, loggers — is resolved by type from a
single `Container`, which the SDK hands to you at well-defined points in the
server's lifecycle. This guide covers when the container is available to
you, how bindings work, and a cookbook of the replacements deployments most
commonly make.

## Motivation

What makes the one-place swap possible is that the codebase follows the
hexagonal (ports and adapters) architecture: components depend on *ports* —
abstract types like `Logger`, `AuthorizationPolicy`, or `SessionStore` — and
never on concrete implementations. The container binds each port to an
implementation at startup, so when you rebind `Logger` to a Datadog-backed
implementation or `SessionStore` to your database, the rest of the stack
keeps working — nothing else ever knew which implementation it was talking
to.

Some frequently-needed configurations have dedicated `p.Server` parameters
(`models=`, `auth=`, `session_store=`...). Those parameters are shortcuts:
under the hood they produce container bindings. The container hooks are the
general mechanism behind all of them.

## Where you get the container

`p.Server` offers two lifecycle hooks, and the container remains accessible
at runtime. They serve different purposes:

### `configure_container` — replace implementations

Runs during startup, *before* the application is assembled. It receives the
container and returns it (usually after rebinding some types). This is the
hook for swapping implementations — anything constructed later will resolve
your bindings:

```python
async def configure_container(container: p.Container) -> p.Container:
    container[p.CompassOptimizationPolicy] = MyOptimizationPolicy()
    return container


async with p.Server(configure_container=configure_container) as server:
    ...
```

### `initialize_container` — act on the finished container

Runs *after* the container is finalized, with every binding in place and every
singleton ready to resolve. Unlike `configure_container`, it doesn't change
*which* implementations are bound — it acts on the ones already there. It returns
nothing; it exists for side effects that need resolved services: registering
listeners, warming caches, and **tuning the runtime configuration of finalized components**.

That last use is the common one. Many components expose plain settings —
properties you assign or setter methods you call — that adjust their behavior
without replacing them. Because `initialize_container` runs after construction,
the instance you resolve here is the *same singleton* Parlant will use, so a
setting you change on it sticks. For example, to raise how many canned responses
Parlant's responder and reviewer surface per turn (see
[Canned Responses](canned-responses.md)):

```python
from parlant.core.engines.compass.responder.responder import Responder
from parlant.core.engines.compass.reviewer import Reviewer


async def initialize_container(container: p.Container) -> None:
    # How many canned responses are surfaced per turn — as style references by
    # the responder, and as review guidance by the reviewer. Both default to 5.
    container[Responder].max_canned_responses = 8
    container[Reviewer].max_canned_responses = 3

    # The last-resort text shown if an agent message stays blocked by moderation
    # through every retry (see [Moderation](moderation.md)).
    container[Responder].moderation_fallback = (
        "Sorry, I can't respond to that right now. How else can I help?"
    )


async with p.Server(initialize_container=initialize_container) as server:
    ...
```

Rebinding types here, by contrast, is a mistake: components that already resolved
their dependencies keep the instances they got, so a late `container[X] = ...`
won't reach them. Change *which* implementation is bound in `configure_container`
(before construction); change the *settings* of a constructed component in
`initialize_container` (after).

### Runtime access — `server.container`

Inside the `async with` block, `server.container[SomeType]` resolves any
bound type. Useful for wiring your application to running services — for
example, subscribing to usage reports (see
[Usage Reports](usage-reports.md)).

### Not a replacement: engine hooks

Engine hooks register *behavioral callbacks* on engine events (message
generated, tool calls reviewed, and so on) rather than replacing
implementations, and they use this same mechanism: resolve
`container[p.CompassHooks]` in `initialize_container` and append your
callbacks to its hook lists. Reach for hooks when you want to observe or
intervene in a turn; rebind a port when you want a different implementation
of a component. See
[Tool & Message Review](review.md#observing-reviews-with-engine-hooks) for a
worked hooks example.

## Binding mechanics

The container maps *types* to implementations. The forms you'll actually use:

```python
# Bind an instance — every resolution gets this exact object.
container[p.CompassOptimizationPolicy] = MyOptimizationPolicy()

# Bind a class as a lazily-constructed singleton (dependencies of its
# __init__ are themselves resolved from the container).
container[p.RateLimiter] = p.Singleton(MyRateLimiter)

# Bind a factory — runs per resolution.
container[p.Logger] = lambda c: build_logger()
```

Two rules of thumb:

- Resolution is by exact type. Bind the *port* (the type components ask
  for), not your subclass:
  `container[p.CompassOptimizationPolicy] = MyOptimizationPolicy()`,
  never `container[MyOptimizationPolicy] = ...`.
- Bind early. Singletons capture their dependencies when first constructed,
  which is why implementation swaps belong in `configure_container` and not
  later.

Bindings are validated as they're made: binding an instance that isn't
actually an instance of the port raises a `TypeError` immediately, so a
mistake fails at startup with a clear message rather than at some distant
resolution site. Factory callables receive the `p.Container` itself, so a
factory can resolve its own dependencies from it.

## Cookbook

What deployments most commonly rebind, with links to the fuller guides where
one exists.

### Engine tuning: the optimization policy

`p.CompassOptimizationPolicy` collects Parlant's cost and latency settings
in one place: request-hedging timeouts per model route, consecutive-message
processing delays, the review retry budget, and when a shared prompt prefix is
worth prefilling. Subclass it, override what you need, and rebind it:

```python
class MyOptimizationPolicy(p.CompassOptimizationPolicy):
    def get_max_review_restarts(self, effort: p.Effort) -> int:
        if effort >= p.Effort.HIGH:
            return 8
        return super().get_max_review_restarts(effort)


async def configure_container(container: p.Container) -> p.Container:
    container[p.CompassOptimizationPolicy] = MyOptimizationPolicy()
    return container
```

See [Working with Sessions](sessions.md#consecutive-message-processing-delay) for
process-delay semantics and [Tool & Message Review](review.md#the-retry-budget)
for retry-budget semantics.

#### When prefill pays for itself

Some evaluations fan out one request per rule, every one of them repeating the
same prompt prefix. Parlant can warm that prefix first, so the repeats arrive
as cache reads instead of full prompts.

The warm-up is itself a request, and providers charge more for a cache *write*
than for ordinary input tokens — so warming to serve a couple of requests costs
more than not warming at all. `get_min_prefill_fan_out` is the count at which
it starts paying:

```python
class MyOptimizationPolicy(p.CompassOptimizationPolicy):
    def get_min_prefill_fan_out(self, hints: dict = {}) -> int:
        return 3   # this provider's write premium is mild; warm sooner
```

The default is 5. Return `0` to switch prefill off entirely. Each evaluator is
judged on its own fan-out, so a turn that spreads a few rules across two
evaluators warms neither.

One thing the threshold cannot fix: the prefix is warmed at the end of a turn,
and provider caches expire on their own schedule. If your conversations have
long gaps between messages, the write may be paid for and never read, whatever
the fan-out.

### Stores

The common stores have constructor shortcuts — `session_store=`,
`customer_store=`, and `variable_store=` on `p.Server` — accepting
`"transient"`, `"local"`, a MongoDB connection string, or a store instance.
Passing an instance creates a container binding under the hood, so a custom
implementation works without touching the container:

```python
async with p.Server(session_store=MySessionStore()) as server:
    ...
```

For stores without a constructor shortcut, bind the port directly in
`configure_container` (`p.SessionStore`, `p.CustomerStore`,
`p.ContextVariableStore` are exported; other store ports live under
`parlant.core.*` — see the stability note below).

### Model plumbing: rate limits and secrets

Model *routing* is not container work — models are configured with the
`models={...}` parameter (see
[Configuring Models and Providers](providers.md)). Two adjacent concerns are:

- **Rate limiting** — `p.RateLimiter` governs concurrency and throughput for
  model requests. `p.BasicRateLimiter` is the default; rebind to integrate a
  distributed limiter (see the
  [providers guide](providers.md#rate-limits) for the worked
  Redis example).
- **Secrets** — durable model credentials are expressed as
  `p.SecretRef("locator")` values inside `ModelConfig` args; a `SecretRef`
  stores a locator, never the credential. The `p.SecretLoader` port resolves
  locators against your secret system (Vault, cloud secret managers, ...) —
  bind yours in `configure_container`:

```python
class VaultSecretLoader(p.SecretLoader):
    async def load(self, ref: p.SecretRef) -> str:
        return await my_vault.read(ref.key)


async def configure_container(container: p.Container) -> p.Container:
    container[p.SecretLoader] = VaultSecretLoader()
    return container


async with p.Server(
    models={
        "*": "openai",
        "react": p.ModelConfig(
            "openai:gpt-5.4",
            args={"api_key": p.SecretRef("models/openai/api-key")},
        ),
    },
    configure_container=configure_container,
) as server:
    ...
```

See [providers.md](providers.md#credentials-and-secrets) for the full secret model,
including transient secrets.

### Logging

Everything logs through the `p.Logger` port — the console, `logs.jsonl`, and
diagnostic bug-report archives all go through it (see
[Diagnostics](diagnostics.md)). Rebind it to route logs into your own
observability stack; implementations usually wrap the existing logger rather
than replace it, so the built-in sinks keep working.

### Authorization

`p.Server(auth=...)` accepts an `AuthConfig` or a full
`p.AuthorizationPolicy` — again a shortcut over the container binding. Custom
policies (who may call which API operation, at what rate) are covered in
[Auth](auth.md).

### Health thresholds

`p.HealthThresholds` holds every cutoff behind the `/healthz` status: engine
turn latency and success rate, model request latency per route, and event loop
lag. Bind your own value to tune them; the defaults reproduce the built-in
behavior. See [Diagnostics](diagnostics.md#status-thresholds).

```python
async def configure_container(container: p.Container) -> p.Container:
    container[p.HealthThresholds] = p.HealthThresholds(
        engine=p.EngineHealthThresholds(degraded_p95_ms=45_000.0),
    )
    return container
```

### Compaction

Session compaction is configured by *calling* the bound `Compacter` rather
than rebinding it — an `initialize_container`-style customization
(`container[Compacter].set_policy({...})`). See
[Session Compaction](compaction.md#configuring-compaction).


## What is *not* configured through the container

Three surfaces that look like DI but have dedicated, better-suited
parameters:

- **Model routing** — `models={...}` on `p.Server` / `create_agent`.
  See [providers.md](providers.md).
- **Prompt customization** — `prompts={...}` route-keyed providers.
  See [Prompt Customization](prompt-customization.md).
- **Per-rule callbacks** — `on_matched=` / `on_discovered=` and per-rule
  review triggers are parameters of rule creation, not container bindings.
  See [Tool & Message Review](review.md).

## Stability note

Types exported from `parlant.sdk` (everything reachable as `p.X`) are the
stable container keys — they are API, and rebinding them is supported.
Anything imported from `parlant.core.*` or `parlant.adapters.*` is internal:
it works as a container key today (the compaction example above does exactly
this), but its location and shape may move between releases. Prefer the
exported ports where one exists, and expect imports from `parlant.core` to
need updates when you upgrade Parlant.
