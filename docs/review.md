# Integrated Tool & Message Reviews

Reviews are one of the most powerful features of Parlant.

A review is a separate check that Parlant runs on what a turn is about to do:
the tool calls it proposes to execute, or the message it is about to send.
If that output would breach the instructions that govern the agent,
Parlant blocks it. The verdict is enforceable because it lands before anything happens: a
rejected tool call is never run, a rejected message is never seen, and the
step regenerates with guidance on how to avoid the violation.

You request reviews declaratively — as an agent-wide default, only when certain
observations or policies apply, or (for specific tools) by marking a tool as consequential.

## Motivation

Generating a response and judging its compliance are two different jobs, and
they often conflict.

The generating model tries to be helpful in the flow of conversation. Under
that pressure it can sometimes drift: call a tool the situation doesn't
justify, claim an action succeeded when no tool actually ran, or phrase a
message in a way a policy forbids. Asking the same generation pass to catch
its own mistake is not reliable enough in practice — the reasoning (and
model) that produced the mistake is the same one asked to catch it.

Review separates the two jobs. A separate model pass — with its own model
route, its own prompt, and no attachment to the draft — checks the step's output
against the agent's instructions before that output takes effect. Because it runs
*before* execution and *before* delivery, its verdict can be enforced: a rejected
tool call is never run, and a rejected message is never seen.

This matters most where mistakes are expensive: tools with real-world consequences
(payments, cancellations, record updates), and conversations governed by hard policy
(compliance, legal, brand). Review lets you pay for this protection only where you
need it, instead of raising the whole agent's effort.

## Tutorial

To test the feature, start with an agent that reviews everything it does:

```python
import parlant.sdk as p


async with p.Server(models={"*": "openai"}) as server:
    agent = await server.create_agent(
        id="my-agent",
        name="Banking Agent",
        instructions="Help customers manage their accounts.",
        message_review=True,   # review every drafted message before delivery
        tool_review=True,      # review every proposed tool call before execution
    )
```

That is the whole setup. From now on, whenever a turn drafts a message, Parlant
holds it back, reviews it against the agent's instructions, and only delivers
it if it passes. Whenever a turn proposes tool calls, they are reviewed
before any of them runs. Watching a session, you'll see this as short status
updates: "Reviewing message" and "Reviewing tool use".

Agent-wide review is often more than you need, though, because it adds
latency and cost to every turn. The two finer-grained triggers usually serve
better:

**Per-rule review** — a rule can request review only for the turns where it
matters:

```python
await agent.create_policy(
    title="Refund limits",
    content=(
        "Refunds above $500 require a manager approval reference. "
        "Never promise such a refund without one."
    ),
    trigger_message_review=True,
)
```

Now message review runs only on turns where this policy actually matched.
Ordinary small talk stays on the fast path.

**Consequential tools** — a tool author can mark a tool as having real-world
impact, which triggers tool review whenever that tool appears in a proposed
call set, regardless of any agent or rule flags:

```python
@p.tool(consequential=True)
async def cancel_order(order_id: str) -> p.ToolResult:
    ...
```

The three triggers combine: review runs if *any* of them requests it. A rule
can add review to a turn, but nothing can cancel a review that something else
requested.

## The two review paths

Tool review and message review are independent — a turn can run either, both,
or neither:

| | Tool review | Message review |
|---|---|---|
| Reviews | The step's reasoning plus its proposed tool calls | The complete drafted message |
| Runs | After a step proposes tool calls, before any of them executes | After a pure message step completes, before the message is emitted |
| Triggered by | `tool_review=True` on the agent, a matched rule with `trigger_tool_review=True`, or a consequential tool in the call set | `message_review=True` on the agent, or a matched rule with `trigger_message_review=True` |
| On pass | The calls commit and run (optionally with a TODO note carried forward) | The message is delivered |
| On breach | The calls are discarded and the step restarts with corrected reasoning | The message is discarded and the step regenerates; the user never sees it |

Points worth knowing up front:

- **Effort floor.** At `Effort.MIN`, no reviews run at all, whatever was
  requested — MIN is Parlant's cost floor (see
  [Adaptive Effort](adaptive-effort.md)). Every other effort level honors
  review requests, and higher effort allows more regeneration attempts after
  rejections (see [Mechanism](#mechanism)).
- **Streaming.** When message review is active for a turn, the message is
  buffered and reviewed whole rather than streamed out token by token — a
  reviewed message can't be half-shown.
- **Matching implications.** Setting `trigger_message_review` or
  `trigger_tool_review` on a rule also upgrades how carefully that rule is
  *matched*. Review fires only on turns where the rule genuinely applies, so
  Parlant gives the rule an individual "does this apply right now?" check
  each turn, instead of relying on similarity search alone. In other words,
  requesting review declares that the rule's matching itself matters — and
  it is treated accordingly.
- **Unsupported action claims.** Message review always enforces one built-in
  rule: a message that states or implies an action was performed (an order
  cancelled, an email sent, a payment processed) must be backed by an actual
  tool result. Promising that an action is about to happen, with no tool run
  behind it, is a breach even if everything else is compliant.

## Usage examples

Review flags are accepted anywhere rules are created — `create_policy`,
and `create_observation` alike.

An observation that makes any turn it matches carefully reviewed:

```python
await agent.create_observation(
    condition="the customer is discussing account closure",
    trigger_message_review=True,
    trigger_tool_review=True,
)
```

A tool-carrying policy whose calls are always reviewed:

```python
await agent.create_policy(
    title="Wire transfers",
    content="Execute wire transfers only after verbal confirmation of the amount and recipient.",
    tools=[execute_wire_transfer],
    trigger_tool_review=True,
)
```

Combining with `trigger_effort` (see [Adaptive Effort](adaptive-effort.md)) —
raise the whole turn's effort *and* review its output when the situation is
serious:

```python
await agent.create_observation(
    condition="the customer disputes a charge or threatens legal action",
    trigger_effort=p.Effort.HIGH,
    trigger_message_review=True,
)
```

Per-agent defaults with rule-level additions:

```python
agent = await server.create_agent(
    id="my-agent-2",
    name="Claims Agent",
    instructions="Process insurance claims.",
    tool_review=True,          # every tool call is reviewed...
)

await agent.create_policy(
    title="Claim denial wording",
    content="When denying a claim, cite the specific policy clause.",
    trigger_message_review=True,       # ...and messages are reviewed on denial turns
)
```

## Advanced usage

### Choosing the review model

The reviewer requests its own model routes — `review.tools` and
`review.message` — so you can pin the checks to a cheaper or stricter model
independently of generation, and vary them by effort. See
[Configuring Models and Providers](providers.md) for the full grammar:

```python
async with p.Server(
    models={
        "*": "openai",
        "review.*": p.ModelConfig("gemini:gemini-3.5-flash-lite"),
        "review.message[effort>=high]": p.ModelConfig(
            "openai:gpt-5.4-mini",
            reasoning="medium",
        ),
    },
) as server:
    ...
```

When `ModelConfig.reasoning` is not set, the reviewer picks its own reasoning
depth from the turn's effort, and from whether any HIGH-weight rules matched.
A high-stakes turn is therefore reviewed more thoroughly, automatically.

### Observing reviews with engine hooks

Four engine hooks bracket the review paths — useful for audit trails,
metrics, or debugging why a step regenerated:

```python
async def initialize_container(container: p.Container) -> None:
    hooks = container[p.CompassHooks]

    async def on_tool_calls_reviewed(
        context: p.EngineContext,
        payload: p.CompassHookPayload.ToolCallsReviewed,
        exc: Exception | None,
    ) -> p.CompassHookResult:
        if not payload.passed:
            audit_log.warning(f"Tool calls rejected: {payload.adjusted_reasoning}")
        return p.CompassHookResult.CALL_NEXT

    async def on_message_reviewed(
        context: p.EngineContext,
        payload: p.CompassHookPayload.MessageReviewed,
        exc: Exception | None,
    ) -> p.CompassHookResult:
        if not payload.passed:
            audit_log.warning(f"Message rejected: {payload.adjusted_reasoning}")
        return p.CompassHookResult.CALL_NEXT

    hooks.on_tool_calls_reviewed.append(on_tool_calls_reviewed)
    hooks.on_message_reviewed.append(on_message_reviewed)


async with p.Server(initialize_container=initialize_container) as server:
    ...
```

The hooks are `on_reviewing_tool_calls` / `on_tool_calls_reviewed` and
`on_reviewing_message` / `on_message_reviewed`. Every engine hook receives
`(context, payload, exc)` and returns a `p.CompassHookResult`. The payload of
`on_tool_calls_reviewed` is a `p.CompassHookPayload.ToolCallsReviewed` (`tool_calls`, `passed`,
`todo`, `adjusted_reasoning`); `on_message_reviewed` carries a
`p.CompassHookPayload.MessageReviewed` (`message`, `passed`, `adjusted_reasoning`). The
`...reviewing` hooks fire before the verdict: `on_reviewing_tool_calls` with
the proposed calls as its payload, `on_reviewing_message` with the drafted
message text.

### Customizing the review prompts

The review prompts are built through the standard prompt-customization
mechanism under the `review.tools` and `review.message` route keys, with
sections named `reviewer.tools.*` and `reviewer.message.*` (see
`p.PromptSections`). See [Prompt Customization](prompt-customization.md).

## Mechanism

### Tool review

When a response step ends with pending tool calls and review is triggered,
the reviewer receives the step's reasoning and the proposed calls, together
with the instructions that govern the turn. It answers with a compact
boolean verdict (`ReviewSchema`): did the reasoning or the calls break or
diverge from the instructions?

- **Clean verdict** — the calls are committed and executed in order. The
  reviewer may still return a `todo`: a summary of what the agent must still
  do before responding. The TODO is stored in the turn state and rendered
  into the next step's instructions, which keeps a multi-step task on track
  without rejecting anything.
- **Breach verdict** — the reviewer returns `adjusted_reasoning`: a complete
  replacement for the step's reasoning, written as if the agent had reasoned
  correctly from the start. The proposed calls are discarded without
  running, the faulty reasoning is replaced by the adjusted version, and the
  step restarts. The next attempt never sees the rejected calls — only the
  corrected reasoning.

### Message review

When a step ends with a message and no pending tool calls, and review is
triggered, the drafted message is held back and reviewed as a whole. If the
agent has canned responses, the most relevant ones (by similarity to the
draft) are rendered into the review prompt as style and compliance guidance,
so review also checks that the draft follows your approved phrasing. The
built-in unsupported-action-claims rule is always part of the prompt.

A passing message is delivered normally. On a breach, the reviewer's
`adjusted_reasoning` replaces the step's reasoning, the step regenerates,
and the rejected draft is never delivered. The regenerating step may not see
the rejected message at all, so the adjusted reasoning must stand on its
own: the user's request, the constraint that applies, and what a compliant
response must do.

### The retry budget

Each reviewer rejection restarts the step (this is called a *semantic failure*).
How many restarts a turn tolerates depends on its effort level:

| Effort | Max restarts |
|---|---|
| MIN | 1 (reviews don't run at MIN anyway) |
| LOW | 2 |
| MEDIUM | 3 |
| HIGH | 5 |
| MAX | 10 |

If the budget runs out, Parlant stops attempting the task and produces a
final message telling the user that it couldn't complete it. It will not
deliver output that repeatedly failed review. That final message is not itself
reviewed: it is the turn's last word, so a rejection there would leave the
customer with nothing.

You can replace that message with your own logic — hand off to a human, emit a
canned response, or say nothing at all — by taking over the `on_giving_up` hook.
See [Take over the give-up message](hooks.md#take-over-the-give-up-message).

The defaults above live on
`p.CompassOptimizationPolicy.get_max_review_restarts`. To change them,
subclass the policy and rebind it in the container:

```python
class MyOptimizationPolicy(p.CompassOptimizationPolicy):
    def get_max_review_restarts(self, effort: p.Effort) -> int:
        if effort >= p.Effort.HIGH:
            return 8
        return super().get_max_review_restarts(effort)


async def configure_container(container: p.Container) -> p.Container:
    container[p.CompassOptimizationPolicy] = MyOptimizationPolicy()
    return container


async with p.Server(configure_container=configure_container) as server:
    ...
```

### Where the verdict comes from

Both paths use the same compact schema (`ReviewSchema`:
`breaches_or_discrepancies` plus optional `adjusted_reasoning`) over the
`review.tools` / `review.message` model routes. The reviewer's reasoning
depth follows the turn's effort, and rises when HIGH-weight rules matched
the turn; an explicit `reasoning` setting on the route's `ModelConfig`
overrides it. Review requests are resolved once per turn, from the agent
default plus this turn's matched rules. A rule that matched earlier in the
session, but not on this turn, does not trigger review.
