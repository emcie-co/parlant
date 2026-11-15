# Wording and Tone of Voice

## WTV001 (HIGH): Don't repeat a canned response's wording in its policy

Flag every policy that has canned responses attached (`canned_responses=[...]`)
and also spells out their wording in its `content`. The canned response
already carries the wording; repeating it in the policy makes two sources of
truth, which drift apart the first time one of them is edited, and the
policy's copy competes with the template when the agent writes its reply.

In the policy, refer to the relevant "reference wording" instead of quoting it.

Before:

```python
refund_started = await agent.create_canned_response(
    template="I've started your refund of {{ amount }}. It should land in 5 to 10 business days.",
)
await agent.create_policy(
    content=(
        "When a tool result confirms the refund was issued, say: "
        "\"I've started your refund of {{ amount }}. It should land in 5 to 10 business days.\""
    ),
    canned_responses=[refund_started],
)
```

After:

```python
await agent.create_policy(
    content=(
        "When a tool result confirms the refund was issued, tell the customer "
        "it's on the way, using the reference wording"
    ),
    canned_responses=[refund_started],
)
```

## WTV002 (MEDIUM): Shape wording and tone with canned responses, not instructions or policies

Flag wording and tone-of-voice guidance written into the agent's
`instructions=` or into policies: phrases to use, greetings, sign-offs,
register ("be warm and casual"), and the like. Prose about style is
interpreted loosely and applied unevenly, while canned responses show the
agent the exact voice in the context where it is needed.

Instead, attach canned responses to the right context:

- To the agent itself, as a global style reference: create the canned
  response with `agent.create_canned_response(...)` and attach it to no rule.
- To specific observations or policies (`canned_responses=[...]`), when the
  responses should be called upon in that context.

Before:

```python
agent = await server.create_agent(
    name="Ava",
    id="ava",
    instructions=(
        "You help customers of Acme Outdoor. Be warm and casual. "
        "Greet people with 'Hey there!' and sign off with 'Happy trails!'"
    ),
)
```

After:

```python
agent = await server.create_agent(
    name="Ava",
    id="ava",
    instructions="You help customers of Acme Outdoor.",
)

# Agent-wide: a style reference for every reply.
await agent.create_canned_response(template="Hey there! What can I help you find today?")
await agent.create_canned_response(template="Glad I could help. Happy trails!")

# Context-specific: called upon when this observation matches.
await agent.create_observation(
    "the customer is planning a trip",
    canned_responses=[
        await agent.create_canned_response(
            template="Ooh, where are you headed? I can suggest gear for the trip.",
        ),
    ],
)
```
