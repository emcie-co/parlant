# Behavior

## BHV001 (HIGH): Split observations with three or more alternative clauses

An observation should describe one situation. Flag every observation whose
condition joins three or more separate clauses that are alternatives, meaning
distinct situations listed together, any one of which should match, usually
with commas, "or" or "and/or". (Clauses that must all hold are BHV003.) Such a
condition asks the engine to judge several situations as one, so it matches
less reliably and is harder to test and to change.

Split it: create one observation per clause. Then add a "combined"
observation whose condition is a more generic description that covers all of
them, and make it depend on any of the split ones with `depend_on_any(...)`.
The engine still judges the combined condition on every turn, so it applies
only while the conversation is about it, and the dependency admits it only
once one of the specific situations has actually matched. Whatever depended
on the original observation can depend on the combined one instead.

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
    "the customer asks about undoing a purchase",
)
await asks_about_returning.depend_on_any(asks_for_refund, wants_to_return, wants_to_exchange)

await refund_window_policy.depend_on(asks_about_returning)
```

## BHV002 (MEDIUM): Split observations with two alternative clauses

An observation should describe one situation. Flag every observation whose
condition joins two separate clauses that are alternatives, meaning distinct
situations listed together, either of which should match, usually with a
comma, "or" or "and/or". (Three or more such clauses are BHV001; clauses that
must both hold are BHV004.) Such a condition asks the engine to judge several
situations as one, so it matches less reliably and is harder to test and to
change.

Split it: create one observation per clause. Then add a "combined"
observation whose condition is a more generic description that covers both,
and make it depend on either of the split ones with `depend_on_any(...)`. The
engine still judges the combined condition on every turn, so it applies only
while the conversation is about it, and the dependency admits it only once
one of the specific situations has actually matched. Whatever depended on the
original observation can depend on the combined one instead.

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
    "the customer asks about undoing a purchase",
)
await asks_about_returning.depend_on_any(asks_for_refund, wants_to_return)

await refund_window_policy.depend_on(asks_about_returning)
```

## BHV003 (HIGH): Split observations with three or more clauses that must all hold

An observation should describe one situation. Flag every observation whose
condition joins three or more separate clauses that must all hold, usually
with "and", "while" or "who", when at least one of them is a lasting fact
about the customer or the conversation (who they are, what they have, what
already happened). (Clauses that are alternatives are BHV001.) Such a
condition asks the engine to judge several situations as one, so it matches
less reliably and is harder to test and to change.

Split it: create one observation per lasting fact. Then add a "combined"
observation whose condition is only what must hold on the current turn, and
make it depend on all of the lasting facts with `depend_on(...)`, which
requires every one of them, unlike `depend_on_any(...)`. A matched
dependency stays satisfied for the rest of the session, which is right for a
lasting fact; the engine still judges the combined condition on every turn,
so it applies only while that part is true. Whatever depended on the original
observation can depend on the combined one instead.

If every clause must hold on the same turn, keep them in one condition; this
rule doesn't apply.

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

returning_pro_asks_price = await agent.create_observation(
    "the customer asks about pricing",
)
await returning_pro_asks_price.depend_on(is_returning, is_on_pro)

await loyalty_discount_policy.depend_on(returning_pro_asks_price)
```

## BHV004 (MEDIUM): Split observations with two clauses that must both hold

An observation should describe one situation. Flag every observation whose
condition joins two separate clauses that must both hold, usually with "and",
"while" or "who", when at least one of them is a lasting fact about the
customer or the conversation (who they are, what they have, what already
happened). (Three or more such clauses are BHV003; clauses that are
alternatives are BHV002.) Such a condition asks the engine to judge several
situations as one, so it matches less reliably and is harder to test and to
change.

Split it: create an observation for the lasting fact. Then add a "combined"
observation whose condition is only what must hold on the current turn, and
make it depend on the lasting fact with `depend_on(...)`. A matched
dependency stays satisfied for the rest of the session, which is right for a
lasting fact; the engine still judges the combined condition on every turn,
so it applies only while that part is true. Whatever depended on the original
observation can depend on the combined one instead.

If both clauses must hold on the same turn, keep them in one condition; this
rule doesn't apply.

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

returning_asks_price = await agent.create_observation(
    "the customer asks about pricing",
)
await returning_asks_price.depend_on(is_returning)

await loyalty_discount_policy.depend_on(returning_asks_price)
```
