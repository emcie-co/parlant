# Behavior

## BHV001 (HIGH): Split observations with three or more clauses

An observation should describe one situation. Flag every observation whose
condition joins three or more separate clauses, meaning distinct situations
listed together, usually with commas, "or" or "and/or". Such a condition asks
the engine to judge several situations as one, so it matches less reliably
and is harder to test and to change.

Split it: create one observation per clause, then add a "combined"
observation with `evaluation_func=p.MATCH_ALWAYS` that depends on any of the
split ones. The combined observation matches whenever one of them does,
without a model call of its own, so whatever depended on the original
observation can depend on the combined one instead.

Before:

```python
asks_about_returning = await agent.create_observation(
    "the customer asks for a refund, wants to return an item, or wants to exchange it",
)
await refund_window_policy.depend_on(asks_about_returning)
```

After:

```python
asks_for_refund = await agent.create_observation("the customer asks for a refund")
wants_to_return = await agent.create_observation("the customer wants to return an item")
wants_to_exchange = await agent.create_observation("the customer wants to exchange an item")

asks_about_returning = await agent.create_observation(
    "the customer asks for a refund, a return or an exchange",
    evaluation_func=p.MATCH_ALWAYS,
)
await asks_about_returning.depend_on_any(asks_for_refund, wants_to_return, wants_to_exchange)

await refund_window_policy.depend_on(asks_about_returning)
```

## BHV002 (MEDIUM): Split observations with two clauses

An observation should describe one situation. Flag every observation whose
condition joins two separate clauses, meaning distinct situations listed
together, usually with a comma, "or" or "and/or". Such a condition asks the
engine to judge several situations as one, so it matches less reliably and is
harder to test and to change. (Three or more clauses are BHV001.)

Split it: create one observation per clause, then add a "combined"
observation with `evaluation_func=p.MATCH_ALWAYS` that depends on any of the
split ones. The combined observation matches whenever one of them does,
without a model call of its own, so whatever depended on the original
observation can depend on the combined one instead.

Before:

```python
asks_about_returning = await agent.create_observation(
    "the customer asks for a refund or wants to return an item",
)
await refund_window_policy.depend_on(asks_about_returning)
```

After:

```python
asks_for_refund = await agent.create_observation("the customer asks for a refund")
wants_to_return = await agent.create_observation("the customer wants to return an item")

asks_about_returning = await agent.create_observation(
    "the customer asks for a refund or a return",
    evaluation_func=p.MATCH_ALWAYS,
)
await asks_about_returning.depend_on_any(asks_for_refund, wants_to_return)

await refund_window_policy.depend_on(asks_about_returning)
```
