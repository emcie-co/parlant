# Wording and Tone of Voice

## WTV001 (HIGH): Don't repeat a canned response's wording in its policy

Flag a policy with attached canned responses (`canned_responses=[...]`) only
when its `content` repeats the response's exact or nearly exact phrasing as a
script for the agent to say. The canned response already carries the wording.
Repeating it in the policy creates two copies that can drift apart and compete
when the agent writes its reply.

When fixing a finding, keep the trigger and every required disclosure, fact,
and action explicit in the policy. Summarize any requirements that were only
in the quoted reply. Use the canned response as a phrasing example.

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
        "the refund was issued, give the amount, and say it should arrive "
        "within 5 to 10 business days."
    ),
    canned_responses=[refund_started],
)
```

Leave this policy alone: `If the information is unavailable, tell the customer
you don't have it.` It states a required disclosure. An attached response such
as `I don't have that information right now.` shows how to phrase it. The policy
must still require the disclosure if the canned response is edited or removed.

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
