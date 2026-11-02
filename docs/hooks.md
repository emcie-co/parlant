# Engine Hooks

Engine hooks are callbacks you register at fixed points in an agent turn's lifecycle —
when a message is generated, when tool calls are reviewed, when the session is
compacted, and so on. The engine calls them as it runs, handing each one the live
turn context and a payload for the event. You use them to *observe* a turn (audit
logs, metrics, tracing) and, when you need to, to *steer* it — skip a stage,
reject a step so it regenerates, or stop the turn and answer it your own way.

Hooks are the counterpart to the dependency-injection container. The container is
where you swap *which* implementation a component uses; hooks are where you watch
or steer the engine's *behavior* without replacing anything. You write async
functions, append them to the hook points you care about, and the engine picks
them up.

This guide covers why hooks exist, how to register them, the full set of hook
points and their payloads, which engine fires which, and a cookbook of common
uses.

## Motivation

Much of what you want to do around an agent isn't about changing how it works —
it's about knowing what it did and occasionally stepping in to add custom logic:

- **Observability.** Emit a metric every time a message is generated, trace how
  long each reasoning step takes, or write an audit record whenever the reviewer
  rejects a message. These need to run *inside* the turn, at the moment the event
  happens, with access to the turn's state.
- **Intervention.** Shortcut an expensive stage when you can (a detected greeting
  needs no rule matching), reject a drafted message or tool call so the step
  regenerates under your correction, or stop a turn outright and answer it your own
  way. Each needs a way to tell the engine what to do at a specific point.

Note that neither fits the use case of modifying the dependency injection container:
you're not replacing the responder or the reviewer, you're reacting to them.
Hooks give you exactly that: a stable, typed callback at each lifecycle
point. You don't need to fork Parlant's code or subclass its components.

## Registering hooks

### `initialize_container`

Hooks are registered through the container (see
[Dependency Injection](dependency-injection.md)): resolve the bound
`CompassHooks` in `initialize_container` and append your callbacks to
the points you want:

```python
import parlant.sdk as p


async def initialize_container(container: p.Container) -> None:
    hooks = container[p.CompassHooks]

    async def on_message_generated(
        context: p.EngineContext,
        message: str,
        exc: Exception | None,
    ) -> p.CompassHookResult:
        my_metrics.increment("messages.generated")
        return p.CompassHookResult.CALL_NEXT

    hooks.on_message_generated.append(on_message_generated)


async with p.Server(initialize_container=initialize_container) as server:
    ...
```

Each hook point is a plain `list`, so you can append more than one callback to the
same point; they run in registration order as a chain (see
[The return value](#the-return-value) below). Hooks are shared across the server,
not per-agent — a callback fires for every agent's turns.

### The hook signature

Every hook has the same shape:

```python
async def hook(
    context: p.EngineContext,
    payload: <depends on the hook>,
    exc: Exception | None,
) -> p.CompassHookResult: ...
```

- **`context`** is the live `p.EngineContext` for the turn — session, agent,
  customer, interaction history, working state, and the event emitters. A hook can
  read all of it, and may even emit its own events. (Its `state` is typed as
  `Any`, since hooks are engine-agnostic.)
- **`payload`** carries what the hook is *about*. Its type depends on the hook —
  an `int` step number, the message `str`, a `Sequence` of tool calls, or one of
  the payload dataclasses under `p.CompassHookPayload` (`GaveUp`, `ToolCallOutcome`,
  `ToolCallsReviewed`, `MessageReviewed`, `Compacted`, and the two-way review
  channels `ReviewingToolCalls` / `ReviewingMessage`). Many hooks carry no data and
  pass `None`. Annotate your handler with the matching type — e.g.
  `payload: p.CompassHookPayload.MessageReviewed`.
- **`exc`** is `None` for every hook except `on_error`, which receives the
  exception the engine caught.

Hooks run **inline** and are awaited within the turn, so a slow hook adds latency
to the response. Keep them fast, or hand off heavy work to a background task.

### The return value

A hook returns a `p.CompassHookResult` that decides what happens next:

- **`CALL_NEXT`** — run the next hook registered on this point, if any. This is the
  normal return for anything that only observes.
- **`RESOLVE`** — skip this point's normal engine work and jump ahead, without
  running the remaining hooks on this point. What gets skipped depends on the hook
  (see the table below): `RESOLVE` on `on_preparing` skips discovery/matching and
  responds directly (a dev-detected-greeting shortcut); on `on_reviewing_tool_calls`
  / `on_reviewing_message` it skips the reviewer, treating the calls/message as
  approved/passed; on `on_giving_up` it skips the forced final message, because your
  handler has answered instead. A hook point that defines no `RESOLVE` behavior
  treats it like `CALL_NEXT`.
- **`BAIL`** — take the negative/stop branch of whatever this hook brackets. At the
  early turn gates (`on_acknowledging` / `on_acknowledged` / `on_preparing` /
  `on_generating_messages`) it **stops the turn** — the engine does nothing further,
  so the hook should first emit whatever it wants (e.g. a canned fallback via
  `context.session_event_emitter`). At the review hooks (`on_reviewing_tool_calls` /
  `on_reviewing_message`) it **rejects** the calls/message — the step regenerates
  with the payload's `note`, or gives up immediately if the hook set
  `payload.give_up`. Other Compass engine points ignore it for now — including
  `on_giving_up`, which has no stop branch to take (the turn is ending either way);
  use `RESOLVE` there to take the response over.

If no hook resolves or bails, the turn proceeds normally. `RESOLVE` and `BAIL` are
the levers that change the outcome, so an observer should always return `CALL_NEXT`.

## The hook points

Grouped by where they fire in a turn. Unless noted, the payload is `None` and
`exc` is `None`.

### Turn lifecycle

| Hook | Fires | Payload |
| --- | --- | --- |
| `on_error` | the engine caught a runtime error | — (the error is in `exc`) |
| `on_acknowledging` | just before the acknowledgement status event | — |
| `on_acknowledged` | just after the acknowledgement status event | — |
| `on_preparing` | just before preparation (rule matching, tool selection) begins | — |
| `on_generating_messages` | just before message generation begins | — |
| `on_message_generated` | a message was generated (blocking output: not yet emitted) | `str` (the message) |
| `on_messages_emitted` | all messages have been emitted into the session | `Sequence[str]` (the turn's delivered messages) |

**`on_messages_emitted`** fires once per turn, after the reply is already on the
customer's wire, and its payload is every message the turn delivered, in the order
the customer saw it. A turn that calls a tool sometimes delivers two: the preamble
("let me check that…") and the reply that follows. Because it runs after delivery
and knows the whole turn, it is the natural place for work that reacts to a finished
exchange — writing to an external memory, auditing, analytics — where
`on_message_generated` would fire once per message and, in blocking mode, before
the customer has seen anything.

`RESOLVE` on **`on_preparing`** skips discovery/matching and responds directly —
useful when you can cheaply detect a trivial input (a greeting) and want a fast,
ungoverned reply. `BAIL` on `on_acknowledging` / `on_acknowledged` / `on_preparing`
/ `on_generating_messages` stops the turn.

`BAIL` on **`on_message_generated`** intercepts the outgoing message — but only
under **blocking** output mode, where the hook runs *before* the message event
is emitted: the emission is suppressed, and your handler may emit a replacement
itself (via `ctx.session_event_emitter.emit_message_event`), e.g. for
last-minute validations. Under **streaming** output the text is already on the
customer's wire by the time the step completes, so there the hook is purely
observational and the verdict is ignored. Tool-call preamble messages
("let me check that…") are likewise not interceptable.

### The Compass engine reasoning loop

| Hook | Fires | Payload |
| --- | --- | --- |
| `on_step_started` | just before a ReAct step runs | `int` (step number) |
| `on_step_completed` | a step finished (not on a reviewer-forced restart) | `int` (step number) |
| `on_message_updated` | a streamed message grew (streaming output only) | `str` (full text so far) |
| `on_giving_up` | the loop ran out of attempts and is about to force a final message | `CompassHookPayload.GaveUp` (`reason`, `steps`, `kind`) |

**`on_giving_up`** is the one place you can replace Parlant's own words. By default
the turn spends one more model call on an "I couldn't help with that" message.
Return `RESOLVE` and that message is never generated: your handler owns the
response and emits whatever it wants (or nothing). See
[Take over the give-up message](#take-over-the-give-up-message).

`kind` says why the turn stopped, so a handler can treat the causes differently:

| `GiveUpKind` | Means |
|---|---|
| `RETRIES_EXHAUSTED` | Output was rejected by review or blocked by moderation once too often. |
| `MAX_STEPS` | The turn used up `agent.max_steps` without preparing a response. |
| `HOOK_REQUESTED` | One of your review hooks rejected a step with `payload.give_up` set. |

### Tool calls (Compass engine)

| Hook | Fires | Payload |
| --- | --- | --- |
| `on_reviewing_tool_calls` | just before the reviewer checks proposed tool calls | `CompassHookPayload.ReviewingToolCalls` (`tool_calls`; editable `note`, `give_up`) |
| `on_tool_calls_reviewed` | right after that review, before any regenerate | `CompassHookPayload.ToolCallsReviewed` (`tool_calls`, `passed`, `todo`, `adjusted_reasoning`) |
| `on_calling_tools` | just before the approved tool calls run | `Sequence[ToolCallPart]` |
| `on_tool_calls_completed` | the step's tool calls finished | `Mapping[str, list[CompassHookPayload.ToolCallOutcome]]` (by tool name) |

On **`on_reviewing_tool_calls`**, `RESOLVE` skips the reviewer and approves the
calls as-is; `BAIL` skips it and rejects them — the step regenerates with
`payload.note` (default *"The review was rejected - please look into it."*), or
gives up immediately if the hook set `payload.give_up = True`.

### Message review (Compass engine)

| Hook | Fires | Payload |
| --- | --- | --- |
| `on_reviewing_message` | just before the reviewer checks a drafted message | `CompassHookPayload.ReviewingMessage` (`message`; editable `note`, `give_up`) |
| `on_message_reviewed` | right after that review, before any regenerate | `CompassHookPayload.MessageReviewed` (`message`, `passed`, `adjusted_reasoning`) |

On **`on_reviewing_message`**, `RESOLVE` skips the reviewer and accepts the message
as passed; `BAIL` skips it and rejects the message — the step regenerates with
`payload.note`, or gives up immediately if the hook set `payload.give_up = True`.

### Compaction (Compass engine)

| Hook | Fires | Payload |
| --- | --- | --- |
| `on_compacting` | just before session-history compaction | `int` (token count before) |
| `on_compacted` | right after compaction | `CompassHookPayload.Compacted` (`before`, `after`) |



## Per-rule handlers

A few members of `CompassHooks` are not general hook points but **handler maps**,
keyed by the id of a specific rule, and they use a different signature —
`(context, RuleEvaluation) -> None`, with no `CompassHookResult`:

- `on_rule_matched_handlers` — called when a specific rule matches (both engines).
- `on_rule_discovered_handlers` — called when a rule enters Compass's session
  working set (Compass only).

You rarely touch these maps directly. The easier way is the SDK's per-rule
callbacks, which register into them for you:

```python
async def on_matched(context: p.EngineContext, evaluation: p.RuleEvaluation) -> None:
    audit_log.info(f"Rule matched: {evaluation.rule.id}")


await agent.create_policy(
    content="When the customer asks for a refund, explain the refund policy",
    on_matched=on_matched,
)
```

## Cookbook

### An audit trail of review decisions

The review points bracket every reviewer pass, so they're the natural place to
record why a step regenerated:

```python
async def initialize_container(container: p.Container) -> None:
    hooks = container[p.CompassHooks]

    async def on_message_reviewed(
        context: p.EngineContext,
        payload: p.CompassHookPayload.MessageReviewed,
        exc: Exception | None,
    ) -> p.CompassHookResult:
        if not payload.passed:
            audit_log.warning(f"Message rejected: {payload.adjusted_reasoning}")
        return p.CompassHookResult.CALL_NEXT

    hooks.on_message_reviewed.append(on_message_reviewed)
```

See [Tool & Message Review](review.md#observing-reviews-with-engine-hooks) for the
full set of review hooks.

### Timing the reasoning loop

`on_step_started` / `on_step_completed` share the step number, so a hook can
bracket each Compass step for latency metrics — for example, by stashing a start
time on `context` keyed by step and reading it back on completion.

### Fast-path a trivial input

If you can cheaply recognize an input that needs no rules or tools — a bare
greeting, say — `RESOLVE` from `on_preparing` skips discovery/matching and goes
straight to the responder, which replies from the agent's base instructions alone:

```python
async def on_preparing(
    context: p.EngineContext,
    payload: None,
    exc: Exception | None,
) -> p.CompassHookResult:
    if looks_like_a_greeting(context):
        return p.CompassHookResult.RESOLVE  # skip matching, respond directly
    return p.CompassHookResult.CALL_NEXT
```

### Take over the give-up message

When a turn runs out of attempts, Parlant's default is to spend one more model call
saying it couldn't help. Often you'd rather do something about it:

```python
async def initialize_container(container: p.Container) -> None:
    hooks = container[p.CompassHooks]

    async def hand_off_instead_of_apologizing(
        ctx: p.EngineContext,
        payload: p.CompassHookPayload.GaveUp,
        exc: Exception | None,
    ) -> p.CompassHookResult:
        ticket = await support_desk.open_ticket(
            customer_id=ctx.customer.id,
            session_id=ctx.session.id,
            summary=payload.reason,
            steps_taken=payload.steps,
        )

        # Stop the agent from auto-replying; a human owns this conversation now.
        await p.Session.current.update(mode="manual")

        await ctx.session_event_emitter.emit_message_event(
            trace_id=ctx.tracer.trace_id,
            data=(
                f"I can't sort this one out myself. I've opened ticket #{ticket.id} "
                "and a colleague will pick it up right here."
            ),
        )

        return p.CompassHookResult.RESOLVE

    hooks.on_giving_up.append(hand_off_instead_of_apologizing)
```

`RESOLVE` tells Parlant the response is handled, so it generates nothing of its own:
the customer sees only your message, and the extra model call is saved. Return
`CALL_NEXT` instead and Parlant's own message still goes out, which is what you want
from a handler that only records the event.

The handler doesn't have to say anything. Returning `RESOLVE` without emitting ends
the turn in silence, which suits a session a human is already watching.

Use `payload.kind` to treat the causes differently — escalating a turn that hit
`MAX_STEPS` on a genuinely hard task is not the same as one whose output kept
failing review:

```python
if payload.kind is p.CompassHookPayload.GiveUpKind.RETRIES_EXHAUSTED:
    ...
```

### Reject a bad message without the reviewer

When you can detect a breach directly, `BAIL` from `on_reviewing_message` rejects
the drafted message and regenerates the step — no reviewer call. Edit the payload
to steer the retry, or force a give-up:

```python
async def on_reviewing_message(
    context: p.EngineContext,
    payload: p.CompassHookPayload.ReviewingMessage,
    exc: Exception | None,
) -> p.CompassHookResult:
    if leaks_another_customers_data(payload.message):
        payload.note = "Never reveal another customer's data; apologize and offer help with their own account."
        # payload.give_up = True  # or end the turn now instead of retrying
        return p.CompassHookResult.BAIL
    return p.CompassHookResult.CALL_NEXT
```

This is the place to put a guardrail on the agent's own output. Note that `on_reviewing_message`
only fires when message review is active for the turn (an agent default or a
matched rule requesting it), so enable review where you rely on it. There is no
post-generation "drop the message" hook: under streaming the text is already on
its way to the customer by the time a message exists, so a guardrail belongs
*before* emission — at review.

## Hooks vs. the container

Hooks and the container solve different problems, and it's worth keeping them
straight:

- Reach for **hooks** (`container[p.CompassHooks]` in `initialize_container`) to *observe or steer a turn* — react to
  an event, record it, or bail.
- Reach for the **container** (`configure_container`) to
  *replace an implementation* — a different logger, store, or optimization policy.

See [Dependency Injection](dependency-injection.md) for the container side.

## Stability note

The hook points, their payloads, and `CompassHookResult` are exported from the SDK
(`p.CompassHooks`, `p.CompassHook`, `p.CompassHookResult`, `p.CompassHookPayload`,
`p.EngineContext`) and are the supported surface. The set may gain points as the
engines evolve; a hook your engine doesn't fire stays dormant rather than erroring,
so it's safe to register defensively.
