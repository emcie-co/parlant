# Retrievers

A **retriever** is an async function that fetches information and puts it into
the agent's context for a turn. You attach it to an agent, or to a single
policy or observation, and Parlant runs it as part of preparing the response.

The agent never decides to call a retriever. That is the difference between a
retriever and a tool, and it is the whole point of the feature.

## Motivation

A tool call costs a round trip. The model reads the conversation, decides a
tool applies, produces arguments, waits for the result, and only then writes
the reply. That is the right shape when the decision matters — when the agent
must choose *whether* to look something up, or *what* to look up.

Often it doesn't matter. If every conversation about an order needs the
customer's order history, there is nothing to decide. Making the model decide
adds a step to the loop and a chance of getting it wrong.

A retriever runs on its own. Parlant starts agent-wide retrievers as soon as
the customer's message is acknowledged, in parallel with the rest of
preparation, and awaits them before the agent writes. The data is simply
there, and no loop step was spent getting it.

Retrievers are also **contextual**. Attach one to a policy or an observation
and it only runs on turns where that rule matched, so the work happens exactly
when its situation holds.

## Tools, retrievers, and variables

Three features put outside data in front of the agent. They differ in who
decides, and when:

| | Who decides to fetch | When it runs | Best for |
|---|---|---|---|
| **Tool** | the agent | mid-turn, in the loop | actions, and lookups that need arguments from the conversation |
| **Retriever** | you | during preparation, in parallel | context the turn always needs, or that a matched rule always needs |
| **Variable** | Parlant | before the turn, on a schedule | slow-changing facts about the customer |

See [Tools](tools.md) and [Context Variables](variables.md).

## An agent-wide retriever

```python
async def recall_recent_orders(ctx: p.RetrieverContext) -> p.RetrieverResult | None:
    orders = await orders_service.recent(ctx.customer.id)

    if not orders:
        return None

    return p.RetrieverResult(
        data=[f"Order {o.id}: {o.status}, expected {o.eta}" for o in orders],
    )


await agent.attach_retriever(recall_recent_orders)
```

`data` is any JSON-serializable value and reaches the agent as retrieved
context for that turn. Returning `None` adds nothing — that is the normal way
to say "nothing relevant here", and it costs the turn nothing.

Attach as many as you need. They run concurrently, and each one decides
independently whether it has anything to contribute.

## A retriever bound to a rule

Attach a retriever to a policy or an observation and it runs only when that
rule matches:

```python
refund_questions = await agent.create_observation(
    condition="the customer asks about a refund they are expecting",
)

async def load_refund_status(ctx: p.RetrieverContext) -> p.RetrieverResult | None:
    refund = await billing.pending_refund(ctx.customer.id)

    if refund is None:
        return None

    return p.RetrieverResult(
        data=f"Pending refund of ${refund.amount:.2f}, issued {refund.issued_on}.",
    )


await refund_questions.attach_retriever(load_refund_status)
```

Now the billing call happens on refund turns and nowhere else. The rule
decides the situation; the retriever supplies the facts for it.

Attaching a retriever changes how the rule is evaluated. A rule with a
retriever must get a real per-turn verdict, so Parlant marks it to be ranked
rather than merely recalled. See [Rule Discovery](rule-discovery.md).

## What a retriever can return

`RetrieverResult` carries more than data:

```python
return p.RetrieverResult(
    data="Order 1234 shipped on Tuesday.",
    metadata={"source": "orders-api"},
    canned_responses=["Your order shipped on {{ ship_date }}."],
    canned_response_fields={"ship_date": "Tuesday"},
    rules=[{"content": "Offer expedited shipping on the next order"}],
)
```

- **`data`** — the retrieved context itself.
- **`metadata`** — carried alongside the result, for clients and observability.
- **`canned_responses`** — literal example messages for this turn. A response
  supplied this way skips relevance ranking, because it was chosen for this
  exact moment. See [Canned Responses](canned-responses.md).
- **`canned_response_fields`** — values that fill `{{ placeholders }}` in this
  turn's canned responses.
- **`rules`** — transient rules that apply to this response only, phrased like
  a policy's content, with an optional `condition`, `priority`, and `weight`.

A result with none of these is treated as nothing retrieved, and Parlant adds
nothing to the turn.

## Deciding late: deferred retrievers

An agent-wide retriever starts early, before Parlant knows which rules matched.
Sometimes the work should start early but the *decision* needs the later
picture. Return a callable instead of a result, and Parlant calls it after
matching, with the full engine context.

This form belongs to agent-wide retrievers. A rule-attached one already runs
after matching — that is what "attached to a rule" means — so it has nothing
to defer:

```python
import asyncio

billing_questions = await agent.create_observation(
    condition="the customer asks about billing, an invoice, or a charge",
)


async def load_account_health(ctx: p.RetrieverContext) -> p.DeferredRetriever:
    # Starts immediately, in parallel with the rest of preparation.
    report = asyncio.create_task(analytics.account_health(ctx.customer.id))

    async def decide(engine_ctx: p.EngineContext) -> p.RetrieverResult | None:
        # Runs once the turn's rules are known.
        rules_in_play = {rule.id for rule in engine_ctx.state.discovered_rules}

        if billing_questions.id not in rules_in_play:
            report.cancel()
            return None

        return p.RetrieverResult(data=await report)

    return decide


await agent.attach_retriever(load_account_health)
```

Read the working set from `discovered_rules`. An observation with nothing
attached to it is admitted there and rendered as a standing instruction; it
never gets a per-turn match verdict, so looking for it among the turn's
matches would find nothing.

The expensive call overlaps with preparation either way. What the deferred
callable adds is the option to drop the result when the turn turns out not to
need it — so cancel the work you started before returning `None`.

## What it costs the turn

A retriever's result enters the turn as a tool result under the retriever's
id, with a response lifespan: it belongs to this response, and does not
linger in later turns.

That has two consequences worth knowing. Everything a retriever returns is
prompt text, so a retriever that returns a large document spends real tokens
on every turn it runs. And because the agent didn't ask for it, it cannot
decline it — the data is in front of the model whether or not this particular
turn needed it. Prefer returning `None` over returning something marginal.

## Choosing between a tool and a retriever

Reach for a **retriever** when:

- the data is needed on every turn, or on every turn where a rule matches
- there is nothing to decide and no arguments to extract
- you want the fetch to overlap with the rest of preparation

Reach for a **tool** when:

- the agent must decide whether the situation calls for it
- the call needs arguments from the conversation, like an order number
- it acts on the world rather than reading from it, and may need confirmation
  (see [Tools](tools.md))

The two compose. A retriever can load the customer's open orders so the agent
starts the conversation informed, while a tool files the dispute once the
customer asks for one.
