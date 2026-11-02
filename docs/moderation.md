# Moderation

Moderation is an optional safety layer with two independent checkpoints: the
customer's incoming messages, and the agent's outgoing ones.

Two separate settings control it:

1. **Whether to moderate** is an agent property — the `moderation` setting on
   each agent (`auto`/`on`/`off`).

2. **What model does the moderating** is deployment configuration — the same
   `models={...}` mapping that selects your chat models also carries two
   moderation entries, one per checkpoint: `moderation.customer` for incoming
   messages and `moderation.agent` for outgoing ones (details in
   [Routing moderation models](#routing-moderation-models) below).

The two settings are separate on purpose: an agent can *declare* that it must
be moderated without knowing which model, or which provider, will do the
checking in a given deployment.

## The agent's moderation property

Every agent carries a `moderation` setting:

```python
import parlant.sdk as p

agent = await server.create_agent(
    id="my-agent",
    name="Support Agent",
    instructions="...",
    moderation=p.Moderation.AUTO,
)
```

- **`auto`** (the default) — moderation follows your configuration: each checkpoint runs if you
  configured a moderation model for it, and is skipped silently if you
  didn't. No provider enables moderation implicitly, so under `auto`,
  moderation is exactly as present as your configuration says. Partial
  coverage is fine: configuring only customer-message moderation gives
  input-side checking and nothing else.

- **`on`** — required: both checkpoints must have a model configured. This is
  validated at server startup and whenever an agent is created or updated
  with `on` — a missing model fails fast with a configuration error rather
  than silently skipping a safety layer you declared mandatory. To require
  one checkpoint while deliberately waiving the other, configure the waived
  one with the `null` provider (below).

- **`off`** — no moderation for this agent, regardless of what's configured.

Moderation is not configurable per message: callers of the API cannot opt a
particular message in or out.

## Customer-message moderation

When a customer message is posted to a moderating agent's session, it runs
through the model routed at **`moderation.customer`** before it enters the
session. A flagged message is stored with `flagged: true` and its moderation
tags (`harassment`, `hate`, `jailbreak`, …) in `groups`. From there, Parlant
treats the message as censored everywhere it matters:

- The flagged text never reaches any prompt. Every consumer — matching,
  response generation, canned-response selection — sees a censor notice
  carrying the reasons ("this message was censored: harassment") instead of
  the content.
- Policies can condition on it ("when the customer's last message is
  censored, …"), and the agent can explain why the message was not addressed.

## Agent-message moderation

The model routed at **`moderation.agent`** checks the agent's own finished
responses. In block message-output mode, the response passes the check before
anything is shown. A flagged response is **regenerated**, not shown: the
moderation reason (never the flagged text) is fed back as guidance and the
agent tries again, sharing the same retry budget as message review. If it is
still flagged after those attempts, the turn gives up and produces a final
message explaining that it couldn't respond. Only if that give-up message is
*also* flagged does Parlant emit a fixed fallback string. Whenever a message is
blocked, the verdict (tags included) is recorded on the emitted event's
internal metadata for frontends and audit.

The last-resort fallback is configurable on the responder via the DI container:

```python
from parlant.core.engines.compass.responder.responder import Responder


async def initialize_container(container: p.Container) -> None:
    container[Responder].moderation_fallback = (
        "Sorry, I can't respond to that right now. How else can I help?"
    )
```

Notes:

- Block mode only: streamed output is already on the wire by the time the
  message is complete, so the streaming loop does not gate. Parlant logs a
  warning at startup and on agent creation/update when an agent has
  agent-message moderation configured but a non-block output mode — waive
  the checkpoint with the `null` provider if that's intentional.
- The check fails open: if the moderation model errors, the original message
  is emitted and a warning is logged. Moderation is a safety layer on top of
  generation, not a delivery gate.

## Routing moderation models

The two checkpoints are ordinary model routes and resolve independently —
they will typically point at different models:

```python
models={
    "moderation.customer": p.ModelConfig("openai:omni-moderation-latest"),
    "moderation.agent": p.ModelConfig("openai:omni-moderation-latest"),
}
```

Moderation is strictly opt-in: no provider registers a default moderation
model, so a provider-wide fallback like `models={"*": "openai"}` never turns
it on — only the explicit routes above do. Among the built-in providers, only
OpenAI offers a moderation model (`omni-moderation-latest`); deployments on
other providers route the moderation checkpoints to a provider that has one
(or a custom one), or leave them unconfigured.

### The `null` provider

Routing a checkpoint to the built-in `null` provider configures it with a
service that never flags:

```python
models={
    "moderation.customer": p.ModelConfig("openai:omni-moderation-latest"),
    "moderation.agent": p.ModelConfig("null"),
}
```

This exists for `moderation=on` agents: the route is genuinely configured,
so the requirement is satisfied, and the configuration shows clearly that
this checkpoint is waived on purpose.

## Writing a custom moderation model

A moderation model implements `ModerationService`:

```python
from parlant.core.nlp.moderation import (
    AgentModerationContext,
    CustomerModerationContext,
    ModerationCheck,
    ModerationService,
)

class MyGuard(ModerationService):
    async def moderate_customer(self, context: CustomerModerationContext) -> ModerationCheck:
        verdict = await my_classifier(context.message)
        return ModerationCheck(flagged=verdict.bad, tags=verdict.tags)
```

Both context objects carry the session and, for agent-side checks, the full
generic engine context of the turn (`context.engine_context`) — so a
context-aware model can weigh the whole conversation, the matched policies,
and tool results, not just the message text. Customer-side moderation runs at
the API boundary before a turn starts, so `engine_context` is `None` there;
the session (and its history) is still available.

`moderate_agent` defaults to running the same check as `moderate_customer`;
override it when agent-side moderation should behave differently.
