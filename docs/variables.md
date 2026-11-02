# Context Variables

A **context variable** is a named value that Parlant loads into the agent's
context before a turn begins. It is how the agent knows who it is talking to
without asking, and without spending a tool call to find out.

```python
await agent.create_variable(
    name="subscription_plan",
    description="The plan this customer is on, which decides what they can be offered.",
)
```

The agent sees the variable's name, its description, and its current value on
every turn, so a policy can say "offer the annual discount to monthly
customers" and the agent already knows which one this is.

## Setting a value

A variable holds a different value per customer, per group, per agent, or one
value for everyone:

```python
plan = await agent.create_variable(name="subscription_plan")

await plan.set_value_for_customer(customer, "premium")
await plan.set_value_for_tag(vip_group.id, "premium")
await plan.set_value_for_agent(agent, "trial")
await plan.set_global_value("free")
```

Values are JSON-serializable, so a variable can hold a string, a number, or a
whole object.

### Which value the agent sees

Parlant looks for a value in this order and stops at the first one it finds:

1. the customer's own value
2. a value set for any group the customer belongs to
3. a value set for the agent
4. the global value

So the global value acts as a default, a group value overrides it for those
customers, and a customer's own value wins over both. A variable with no value
at any level is simply absent from the turn.

## Variables that keep themselves current

A value you set by hand stays as you set it. To have Parlant fetch it instead,
attach a tool and say how fresh the value has to be:

```python
@p.tool
async def fetch_open_orders(context: p.ToolContext) -> p.ToolResult:
    return p.ToolResult(data=await orders.open_for(context.customer_id))


await agent.create_variable(
    name="open_orders",
    description="The customer's orders that have not been delivered yet.",
    tool=fetch_open_orders,
    freshness_rules="*/15 * * * *",
)
```

`freshness_rules` is a cron expression describing when the stored value goes
stale. Parlant measures it from the moment the value was last written: if the
next scheduled time has not arrived, the stored value is used as-is; if it has,
Parlant calls the tool and stores what comes back.

Two things follow from this. The tool is called with **no arguments** — only a
`ToolContext` carrying the agent, session, and customer ids — because there is
no conversation to draw arguments from. And with no `freshness_rules`, a
tool-backed variable is refreshed on **every** turn, which is rarely what you
want.

## Keep values small

A variable's value is in the prompt on every single turn, so its size is a
recurring cost, not a one-time one. Parlant warns when a tool-backed variable
produces more than about a thousand tokens:

> Tool-enabled context variable 'open_orders' produced 4,300 tokens for key
> 'guest'. Consider compacting it or using a tool with a response-lifetime
> result instead.

Take the advice literally. Return a summary rather than a payload — three
order lines rather than the full order objects — or move the data to a tool or
a retriever, where it only costs the turns that actually use it.

## Variables, tools, and retrievers

All three put outside data in front of the agent, and they differ in when the
fetching happens:

| | When it runs | Lives for | Best for |
|---|---|---|---|
| **Variable** | before the turn, on your schedule | until the next refresh | slow-changing facts: plan, tier, entitlements |
| **Retriever** | during preparation, in parallel | this response | context this turn needs, decided by you |
| **Tool** | mid-turn, when the agent calls it | this response | actions, and lookups needing conversation arguments |

A useful test: if the answer would be the same for this customer an hour from
now, it is a variable. If it is true only for this conversation, it is a
retriever. If the agent has to decide whether to ask at all, it is a tool.

See [Tools](tools.md) and [Retrievers](retrievers.md).

## Writing good descriptions

The description is not a comment. It reaches the agent along with the value,
and it is what tells the model how to read that value.

```python
# Don't: the model sees "tier: 2" and has to guess what 2 means.
await agent.create_variable(name="tier")

# Do: the value now carries its meaning.
await agent.create_variable(
    name="tier",
    description=(
        "Loyalty tier from 1 to 3. Tier 3 customers get free returns; "
        "tier 1 customers pay return shipping."
    ),
)
```

Where the meaning drives behavior, put the behavior in a policy and let the
description explain only what the value *is*. See
[Policy Design](policy-design.md).
