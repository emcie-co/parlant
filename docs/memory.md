# Memory

An agent needs two very different kinds of memory. It needs to remember **this conversation** — what was said a moment ago, including what happened earlier in a long
session. And it needs to remember **this customer** — the durable facts, preferences,
and history that should carry from one conversation to the next, even when a different
agent picks up the thread.

Parlant provides the first kind for you. You connect the second kind yourself,
and this guide shows how, using [mem0](https://github.com/mem0ai/mem0), an
open-source memory layer, as the worked example.

## The two kinds of memory

**Session Memory is built in.** Every turn, Parlant loads the
session's interaction history into the prompt, so the agent always sees the
conversation so far. When a session grows past a threshold, Parlant compacts the
earlier part into a durable summary and keeps going on top of it — the agent still
"remembers" the beginning of a long conversation without paying for the whole
transcript (see [Session Compaction](compaction.md)). None of this requires code or
configuration; it is how every session works.

**Customer Memory is the subject of this guide.** Session Memory ends with the
session. What it can't do is recall, in *tomorrow's* conversation, that this customer
is allergic to peanuts, prefers email over phone, or is midway through a return they
started last week — especially if tomorrow's conversation is with a different agent.
That is **Customer Memory**: durable knowledge about a specific *customer* that spans
sessions and agents. Parlant doesn't ship a memory store for this — you bring your
own — but it gives you the two connection points you need to plug one in.

## Motivation

Why does Parlant ask you to bring your own store instead of shipping one?
Because Customer Memory raises hard product and compliance questions: what
you're allowed to retain, for how long, where it's stored, how it's deleted
on request, and how it's shared across an organization's agents. A framework
shouldn't make those decisions for you. So Parlant does not store this data
itself. Instead, it splits the job into two parts:

- **Memory retrieval** — before the agent responds, pull the memories relevant to the moment and
  put them in front of the model.
- **Memory update** — as the conversation reveals things worth keeping, write them to your store.

Point both halves at your memory system of choice — mem0, a vector database, your own
CRM — keyed by the customer, and the agent gains a memory that persists.

## How it works in Parlant

### The key: customer identity

A Parlant `Customer` is a global entity with a stable `id`. A `Session` merely
*references* a `customer_id` (and an `agent_id`) — the customer is not owned by any
session or agent. So **memory keyed by `customer.id` naturally spans every session and every agent** that talks to that customer. You can let Parlant assign the id, or pin
your own (e.g. your CRM's id) so it lines up across systems:

```python
customer = await server.create_customer(name="Alex", id="crm-42")
```

Inside both connection points below you read the current customer as
`p.Customer.current.id`. That's the value you hand your memory store as its
per-user key. (`p.Customer.current`, `p.Session.current`, and
`p.Agent.current` are valid anywhere inside a turn — hooks, retrievers,
field providers — with no need to pass context objects around.)

### Memory retrieval — a retriever

A **retriever** is an async function attached to an agent that runs during a turn's
preparation and injects extra context into the prompt. It's the natural place to fetch
memories: it receives the customer and the latest message, so it can query your store
and return what it finds.

```python
async def recall(ctx: p.RetrieverContext) -> p.RetrieverResult | None:
    if msg := ctx.interaction.last_customer_message:
        if facts := mem0_store.search(msg.content, user_id=p.Customer.current.id):
            return p.RetrieverResult(data="Memory about this customer:\n" + "\n".join(facts))
    return None

await agent.attach_retriever(recall)
```

`RetrieverResult.data` (any JSON-serializable value) is injected into that turn's
prompt as retrieved context. Return `None` to inject nothing. Retrievers are kicked off
in parallel at the start of preparation and awaited before generation, so a retriever
is free to `await` (or offload) an external call.

### Memory update — a hook that runs after the agent replies

To *write* memories you need a callback that sees the **whole exchange** — what the
customer said and what the agent answered. Half of what is worth remembering only
exists in the reply: the agent confirms the appointment, quotes the price, or promises
the callback. A callback that runs before the reply cannot see any of that.

Parlant's **engine hooks** give you the right moment. `on_messages_emitted` fires once
per turn, after everything the agent said has reached the customer, and hands you
those messages. You register it on the container (see [Engine Hooks](hooks.md)):

```python
async def initialize_container(container: p.Container) -> None:
    hooks = container[p.CompassHooks]

    async def remember(
        ctx: p.EngineContext,
        messages: Sequence[str],
        exc: Exception | None,
    ) -> p.CompassHookResult:
        if (customer_message := ctx.interaction.last_customer_message) and messages:
            await memory.add(
                [
                    {"role": "user", "content": customer_message.content},
                    {"role": "assistant", "content": "\n".join(messages)},
                ],
                user_id=ctx.customer.id,
                run_id=ctx.session.id,
            )

        return p.CompassHookResult.CALL_NEXT

    hooks.on_messages_emitted.append(remember)


async with p.Server(initialize_container=initialize_container) as server:
    ...
```

A turn can deliver more than one message — a tool preamble and then the reply — so
the payload is a list, joined here into the assistant's half of the exchange.

Three things make this pattern reliable:

- **Store both sides of the exchange.** This is what mem0 asks for: `add` takes an
  ordered list of `user`/`assistant` turns, and its fact extraction reads both roles.
  The reply is what resolves references in the customer's message — mem0's own example
  stores *"User's dog Biscuit turned 5"* from a message that only said *"He turned 5
  today"*, because the surrounding turn said who "he" was. Send only the customer's
  half and you lose both the facts the agent stated and the context that makes the
  customer's half readable later.
- **Pass only the new messages, not the whole conversation.** A good memory layer keeps
  its own thread and pulls prior context by key, so resending the full history each turn
  is wasteful and eventually overflows the context window. mem0's guidance is explicit:
  send the new turns and let it resolve the rest.
- **Let the memory layer decide what is worth keeping.** mem0 does its own extraction —
  deciding what matters, removing duplicates of what it already knows, and storing
  nothing when there is nothing new. You hand it the exchange; it decides what to keep.
  (To write only at specific moments and cut write cost, gate the hook on a rule — see
  [Notes](#notes).)

That's the whole pattern: `recall` reads, `remember` writes, both keyed by `customer.id`.

## Tutorial: mem0

[mem0](https://github.com/mem0ai/mem0) is a memory layer that uses an LLM to
extract important facts from a conversation, embeds them, and retrieves the
relevant ones on demand. It fits the two connection points above naturally.

### Install

```bash
pip install mem0ai
```

The self-hosted memory runs in your process. By default it uses OpenAI for both
fact-extraction and embeddings and a local, embedded vector store — so it needs only an
`OPENAI_API_KEY` and no external database. (This is independent of the model your
*agent* runs on.) mem0 ships an async client, `AsyncMemory`, which you can `await`
directly. Pin the models explicitly — mem0's default chat model can reject the low
temperature it uses internally:

```python
from mem0 import AsyncMemory

memory = AsyncMemory.from_config({
    "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini", "temperature": 0.1}},
    "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
})
```

mem0's surface is small. `memory.add(messages, user_id=...)` stores extracted facts,
where `messages` is an ordered list of `{"role": "user" | "assistant", "content": ...}`
turns — the same shape the chat APIs use. And
`memory.search(query, filters={"user_id": ...}, limit=...)` returns the relevant ones as
`{"results": [{"memory": "...", "score": 0.49, ...}, ...]}`.

### Wire it into an agent

Use `AsyncMemory` so the mem0 calls await cleanly inside the async retriever
and callback.

```python
import asyncio
from collections.abc import Sequence

import parlant.sdk as p
from mem0 import AsyncMemory

memory = AsyncMemory.from_config({
    "llm": {"provider": "openai", "config": {"model": "gpt-4o-mini", "temperature": 0.1}},
    "embedder": {"provider": "openai", "config": {"model": "text-embedding-3-small"}},
})


# Memory update — once the agent has replied, hand mem0 the full exchange; it
# extracts and stores anything memorable (and nothing when there's nothing new).
async def initialize_container(container: p.Container) -> None:
    hooks = container[p.CompassHooks]

    async def remember(
        ctx: p.EngineContext,
        messages: Sequence[str],
        exc: Exception | None,
    ) -> p.CompassHookResult:
        if (customer_message := ctx.interaction.last_customer_message) and messages:
            await memory.add(
                [
                    {"role": "user", "content": customer_message.content},
                    {"role": "assistant", "content": "\n".join(messages)},
                ],
                user_id=ctx.customer.id,  # owns the memory (spans sessions)
                run_id=ctx.session.id,  # scopes mem0's extraction context to this session
            )

        return p.CompassHookResult.CALL_NEXT

    hooks.on_messages_emitted.append(remember)


async def main() -> None:
    async with p.Server(initialize_container=initialize_container) as server:
        agent = await server.create_agent(
            id="my-agent",
            name="Support",
            instructions="You are a helpful assistant. Use what you know about the customer.",
        )

        # Memory retrieval — pull the customer's memories relevant to their latest message.
        async def recall(ctx: p.RetrieverContext) -> p.RetrieverResult | None:
            if msg := ctx.interaction.last_customer_message:
                hits = await memory.search(msg.content, filters={"user_id": p.Customer.current.id})
                if facts := [h["memory"] for h in hits["results"]]:
                    return p.RetrieverResult(data="Memory about this customer:\n" + "\n".join(facts))
            return None

        await agent.attach_retriever(recall)


if __name__ == "__main__":
    asyncio.run(main())
```

### What this gives you

Give a stable customer the same `id` across conversations, and the two
connection points work together across sessions and agents. In one session
the customer mentions *"I'm allergic to peanuts"* — `remember` fires and
mem0 stores *"Alex is allergic to peanuts."* In a later,
entirely separate session, that customer asks *"can you suggest a snack?"* — `recall`
searches mem0, finds the allergy, injects it, and the agent recommends peanut-free
options. Neither session shared any conversation state; the memory carried
over because both sessions used the same customer id.

## Notes

- **Offload the write anyway.** `on_messages_emitted` runs after the reply has reached
  the customer, so a slow write doesn't delay what they see. It does still hold the turn
  open until it returns, and `memory.add` is LLM-backed (it extracts facts), so for
  production spawn it as a background task rather than awaiting it inline.
- **Hooks are server-wide, not per-agent.** A callback appended to `CompassHooks` fires
  for every agent's turns. If only one of your agents should write memories, check
  `ctx.agent.id` at the top of the hook and return early otherwise.
- **Scope with the right keys.** `user_id=customer.id` is what makes a memory belong to
  the customer and span every session and agent — so `search` filters on it alone. On
  `add`, also pass `run_id=session.id`: it scopes mem0's *extraction context* (the prior
  messages it resolves references against) to the current session, which is why sending
  just the new turns is enough. mem0 also accepts `agent_id` if you want memory private
  to one agent.
- **Choose what to capture.** Writing on every turn and letting mem0 extract is the
  simplest robust default. If memory-write cost matters, let Parlant decide when to
  write: create an observation for the moments worth remembering, and have the hook
  write only when that observation matched this turn.

  ```python
  memorable = await agent.create_observation(
      condition="the customer shares a personal fact, preference, or constraint",
  )

  # ...inside the hook:
  if not any(m.rule.id == memorable.id for m in ctx.state.ordinary_rule_matches):
      return p.CompassHookResult.CALL_NEXT
  ```
- **Session Memory is separate and automatic.** This whole guide is about
  *Customer Memory*; the running conversation and its compaction summary are always present
  regardless (see [Session Compaction](compaction.md)).
- `on_messages_emitted` is one of many lifecycle points, and the retriever is built on
  the same mechanism underneath — see [Engine Hooks](hooks.md) for the full set.
