# Behavior

## BHV001 (HIGH): Split observations with three or more alternative clauses

An observation should describe one situation. Flag every observation whose
condition joins three or more separate clauses that are alternatives, meaning
distinct situations listed together, any one of which should match, usually
with commas, "or" or "and/or". (Clauses that must all hold at once are
BHV003.) Such a condition asks the engine to judge several situations as
one, so it matches less reliably and is harder to test and to change.

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

## BHV002 (MEDIUM): Split observations with two alternative clauses

An observation should describe one situation. Flag every observation whose
condition joins two separate clauses that are alternatives, meaning distinct
situations listed together, either of which should match, usually with a
comma, "or" or "and/or". Such a condition asks the
engine to judge several situations as one, so it matches less reliably and is
harder to test and to change. (Three or more clauses are BHV001; clauses
that must both hold at once are BHV004.)

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

## BHV003 (HIGH): Split observations with three or more clauses that must all hold

An observation should describe one situation. Flag every observation whose
condition joins three or more separate clauses that must all hold at once,
usually with "and", "while" or "who". Such a condition asks the engine to
judge several situations as one, so it matches less reliably and is harder to
test and to change. (Clauses that are alternatives are BHV001.)

Split it: create one observation per clause, then add a "combined"
observation with `evaluation_func=p.MATCH_ALWAYS` that depends on all of the
split ones with `depend_on(...)`, which requires every one of them, unlike
`depend_on_any(...)`. The combined observation matches once all of them have,
without a model call of its own, so whatever depended on the original
observation can depend on the combined one instead.

Before:

```python
returning_pro_asks_price = await agent.create_observation(
    "the customer is a returning customer on the Pro plan and asks about pricing",
)
await loyalty_discount_policy.depend_on(returning_pro_asks_price)
```

After:

```python
is_returning = await agent.create_observation("the customer is a returning customer")
is_on_pro = await agent.create_observation("the customer is on the Pro plan")
asks_about_pricing = await agent.create_observation("the customer asks about pricing")

returning_pro_asks_price = await agent.create_observation(
    "a returning Pro customer asks about pricing",
    evaluation_func=p.MATCH_ALWAYS,
)
await returning_pro_asks_price.depend_on(is_returning, is_on_pro, asks_about_pricing)

await loyalty_discount_policy.depend_on(returning_pro_asks_price)
```

## BHV004 (MEDIUM): Split observations with two clauses that must both hold

An observation should describe one situation. Flag every observation whose
condition joins two separate clauses that must both hold at once, usually
with "and", "while" or "who". Such a condition asks the engine to judge
several situations as one, so it matches less reliably and is harder to test
and to change. (Three or more such clauses are BHV003; clauses that are
alternatives are BHV002.)

Split it: create one observation per clause, then add a "combined"
observation with `evaluation_func=p.MATCH_ALWAYS` that depends on both of the
split ones with `depend_on(...)`, which requires every one of them, unlike
`depend_on_any(...)`. The combined observation matches once both of them
have, without a model call of its own, so whatever depended on the original
observation can depend on the combined one instead.

Before:

```python
returning_asks_price = await agent.create_observation(
    "the customer is a returning customer and asks about pricing",
)
await loyalty_discount_policy.depend_on(returning_asks_price)
```

After:

```python
is_returning = await agent.create_observation("the customer is a returning customer")
asks_about_pricing = await agent.create_observation("the customer asks about pricing")

returning_asks_price = await agent.create_observation(
    "a returning customer asks about pricing",
    evaluation_func=p.MATCH_ALWAYS,
)
await returning_asks_price.depend_on(is_returning, asks_about_pricing)

await loyalty_discount_policy.depend_on(returning_asks_price)
```
