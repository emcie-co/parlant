# Tools

A tool is a Python function the agent can call while composing a response. It
can look something up, act on a system, or fetch the data an instruction needs.
You attach tools to rules — a policy or an observation. When the rule applies
to the current turn, its tools are offered to the model, and the model calls
them natively as part of its response loop.

This guide covers the mental model, how to author and attach tools, parameter
options, and the advanced knobs.

## Overview

```python
import parlant.sdk as p

@p.tool
async def check_balance(account_name: str) -> p.ToolResult:
    return p.ToolResult(
        data=await bank.get_balance(
            p.Customer.current.id,
            account_name,
        )
    )

await agent.create_observation(
    condition="You need to get the customer's current account balance",
    tools=[check_balance],
)
```

The rule says when its tools matter. Here, an observation describes the
situation ("you need the balance"), and when it matches a turn, its tools join
the model's tool catalog for that turn. Parlant runs the calls the model makes,
folds the results back into the loop, and the next step reasons with them. A
policy works the same way: its tools are offered on the turns it matches.

A single rule can carry several tools, and a tool can serve several rules.
Tool selection happens on every turn: only tools whose rules apply are
offered. This keeps the catalog small and the model focused.

### The execution loop

The response loop moves between reasoning, tool calls, and message writing:

1. Parlant matches rules and assembles the turn's tool catalog.
2. The model reasons and may call one or more tools. It may also call the same
   tool several times with different arguments.
3. Results are added to the loop's working state. Rules that depend on tool
   results are evaluated again, and any terms or rules the results introduce
   join the turn.
4. The model either calls more tools or writes the response.

Results are saved to the session as tool events. The result's `lifespan`
control decides whether they stay there — see
[How long a result lives](#how-long-a-result-lives).

## Tool results

`ToolResult` carries more than data:

- `data` — what the model sees and reasons with.
- `metadata` — attached to the tool event for frontends. The model never sees it.
- `control` — processing options (e.g. `lifespan`, session `mode`).
- `canned_responses` — literal response texts the tool supplies for this turn
  (see [Canned Responses](canned-responses.md), "Tool-provided responses").
- `canned_response_fields` — values for canned-response field substitution.
- `rules` — transient rules ("do X for this response", optionally with a
  priority). They compete like any matched instruction, for the current turn only.

## How long a result lives

A tool result is prompt material. A result saved to the session is replayed on
every later turn of that conversation, so a large one is not a one-time cost —
it is a cost you pay again on every turn until the session ends.

`lifespan` decides that, and takes one of three values:

```python
@p.tool
async def summarize_policy_document(topic: str) -> p.ToolResult:
    return p.ToolResult(
        data=await docs.render(topic),
        control={"lifespan": "response"},
    )
```

- **`"response"`** — the result serves the current response and is not saved.
  It stays fully available for the rest of the turn, including later steps, so
  the agent can reason over it and call more tools. It simply does not come
  back on the next turn.
- **`"session"`** — the result is saved and replayed on later turns, so the
  agent still knows it without calling the tool again.
- **`"auto"`** — the default. Parlant estimates the result's size and decides:
  over about 1,000 tokens it is treated as `"response"` and a warning is
  logged; smaller results are saved.

Reach for `"response"` when the result is large, or when it is only true right
now — a rendered document, a search result set, today's queue. It is the
answer to "my tool returns a lot of data": the agent still gets all of it for
this turn's reasoning, and the conversation doesn't carry the weight
afterwards. Reach for `"session"` when it is a small fact the agent should
carry through the conversation, like the account it just looked up.

Parlant warns when a result looks too heavy for its lifespan: above 2,000
tokens for `"response"` and `"auto"`, and above 1,000 for `"session"`, where
the cost repeats. A `"session"` warning suggests `"response"` explicitly:

> Tool local:fetch_report returned a session-lifespan result with 3,400 tokens.
> Consider compacting it, or setting ToolResult(control={"lifespan":
> "response"}) to avoid excessive token accumulation throughout the session.

Treat those warnings as real. They are the difference between a conversation
that stays cheap and one whose prompt grows every turn.

### The payload limit

A tool result travels to Parlant over HTTP, and a single response chunk is
capped at **16KB** by default. A tool that returns more fails with a
`ToolResultError` naming the limit:

```
url='http://localhost:8818/...', arguments='...', Response exceeds 16KB limit
```

Raise it with an environment variable, read once at startup:

```sh
export PARLANT_TOOL_RESULT_MAX_PAYLOAD_KB=64
```

Raising the cap is the right move when the data genuinely has to reach the
agent whole — a long document it must quote from, say. It is the wrong move
when a tool is returning a raw payload that nobody reads: the model pays for
every token of it. Prefer returning what the agent actually needs, and pair a
large result with `"response"` lifespan so it doesn't repeat on later turns.

## Parameters

Tool parameters are inferred from the function signature. Supported types
include scalars, enums, lists, dates/datetimes, `Optional`s, and pydantic
models. Each parameter can carry `ToolParameterOptions`:

```python
@p.tool
async def transfer(
    amount: int,
    to_account: Annotated[str, p.ToolParameterOptions(
        description="The recipient's account name",
        source="customer",
    )],
) -> p.ToolResult: ...
```

- `description` — how to extract the argument. Shown to the model.
- `source` — where the value may come from: `"customer"` (must be stated by the
  customer, never guessed), `"context"` (taken from session data, not asked),
  or `"any"`.
- `significance` — why the parameter is needed. It helps the model explain
  itself when asking for the value.
- `examples` — example values, folded into the parameter description.
- `hidden` — internal-only: the model never reveals, mentions, or asks the
  customer about it.
- `display_name` — the customer-facing name for the parameter. The model uses
  it, and never the internal name, whenever it talks to the customer about the
  parameter.

When a required argument is missing, the agent asks for it conversationally.
It uses display names, respects `source`, and never surfaces hidden
parameters.

## Reaching turn context from a tool

A tool receives its arguments, and nothing else is required. But sometimes a
tool needs to know what is happening in the conversation right now. Inside a
running turn, the `.current` accessors give the full picture:

- `p.Session.current` — the live session: `.interaction` (the conversation so
  far), `.effort` (the turn's resolved effort level — see
  [Adaptive Effort](adaptive-effort.md)), metadata, and more.
- `p.Customer.current` / `p.Agent.current` — who is talking, and which agent
  is serving them.
- `p.Server.current` — the running server, for SDK operations from inside a
  tool. It also exposes `.logger`.

**When to use them:** when the tool's behavior should depend on the live
turn, not only on its arguments. Here are two examples.

A tool that reacts to what the customer just said. Here, it picks up an
attachment from the last message:

```python
@p.tool
async def file_complaint(summary: str) -> p.ToolResult:
    last = p.Session.current.interaction.last_customer_message
    attachments = last.attachments if last else []
    return p.ToolResult(data=await crm.open_complaint(summary, attachments))
```

A tool that scales its work with the turn's effort. It searches more widely
when the turn is demanding:

```python
@p.tool
async def search_knowledge_base(query: str) -> p.ToolResult:
    limit = 20 if p.Session.current.effort >= p.Effort.HIGH else 5
    return p.ToolResult(data=await kb.search(query, limit=limit))
```

**One caution:** the `.current` accessors only work for tools that run inside
the Parlant server process. A tool hosted on a separate, remote service
cannot use them. Such tools should rely on their arguments instead.

For that remote case, a tool can declare an optional first parameter:
`context: p.ToolContext`. It carries the basics: the session, agent, and
customer ids, plus callbacks for emitting events. Parlant sends it over the
network to remote plugin services. Tools that run in-process don't need it.
For them, prefer the `.current` accessors.

```python
@p.tool
async def remote_lookup(context: p.ToolContext, query: str) -> p.ToolResult:
    return p.ToolResult(data=await search(query, customer_id=context.customer_id))
```

## Consequential tools

Mark a tool `consequential=True` when running it has real-world effects the
customer should approve first: a payment, a booking, an action that cannot be
undone.

```python
@p.tool(consequential=True)
async def book_flight(destination: str, date: str) -> p.ToolResult: ...
```

The agent must confirm with the customer before running a consequential tool.
Routine lookups should not be marked consequential. The agent runs those
directly as part of serving the request, without asking permission.

## Telling the customer what the agent is doing

While a tool runs, the customer sees a status. By default that status is a
generic *Running tools*, which says nothing about what is actually happening.
Give the tool a `narration` and Parlant emits your message as the tool starts,
so that is what the customer reads instead:

```python
@p.tool(narration="Looking up your order…")
async def check_order_status(order_id: str) -> p.ToolResult: ...
```

Pass several and Parlant picks one at random per call, so a customer who waits
through the same lookup twice doesn't read the same line twice:

```python
@p.tool(narration=["Checking with the warehouse…", "Looking up your order…"])
async def check_order_status(order_id: str) -> p.ToolResult: ...
```

For a message that depends on who is asking, pass a function. It is resolved
on the server, once per call, and only the resulting string is sent to the
client. Its parameters are bound by name, the way a parameter's
`choice_provider` is: declare `context: ToolContext` to receive the agent,
session and customer ids, and name any `plugin_data` keys you need. It may be
sync or async:

```python
async def narrate_lookup(context: p.ToolContext) -> str:
    customer = await customers.find(context.customer_id)
    return f"Looking up {customer.first_name}'s order…"


@p.tool(narration=narrate_lookup)
async def check_order_status(order_id: str) -> p.ToolResult: ...
```

The function never receives the engine's turn context — narration is resolved
across the plugin boundary, so it only gets what crosses that line.

Narration is best-effort. If resolving it fails, Parlant logs a warning, keeps
the default status, and runs the tool anyway: a status message must never cost
a customer their answer.

Your message arrives as a `status` event with `status="processing"`, emitted
once per tool call, right after the generic one:

```
processing | Running tools
processing | Checking with the warehouse…
```

Both events are in the stream; a client that shows the latest status shows
yours. Over AG-UI they arrive as `CUSTOM` events named `parlant.status`; see
[AG-UI](ag-ui.md).

## Advanced: overlapping tools

Sometimes two tools cover similar ground: a specialized search next to a
general one, or the same operation exposed by two services. Calling both for
the same item would duplicate work — or worse, duplicate a side effect.

Declare the overlap and Parlant tells the model the tools are alternatives:

- Per tool, via the `overlap` option:
  - `ToolOverlap.AUTO` (default) — overlaps where a relationship declares it.
  - `ToolOverlap.ALWAYS` — treated as an alternative to every other tool
    offered alongside it.
  - `ToolOverlap.NONE` — never treated as overlapping, even if a relationship
    names it.

When any of the turn's offered tools overlap (and only then), Parlant adds a
note to the response instructions. The note lists the overlapping pairs and
directs the model to choose the single most fitting tool per item. Different
items in the same request may still each use a different tool: "transfer 200
shekels to Alisse and $300 to Bob" correctly uses `transfer_shekels` for one
and `transfer_money` for the other. With no overlap configured, nothing is
rendered and no prompt tokens are spent.
