# Adaptive Effort

Effort controls how much work an agent invests in a turn. It affects which
models run, how deeply they reason, how carefully rules are matched, and how
many chances a review gets.

There are five levels: `MIN`, `LOW`, `MEDIUM`, `HIGH`, and `MAX`. You can set
effort in two places:

- **On the agent** — the baseline that every turn starts from.
- **On a rule** — a matched policy or observation can raise the effort of the
  current turn.

Why both? Most turns are routine and should run cheap and fast. But you can't
know in advance which turns are important. With adaptive effort, the
conversation itself signals when the situation is serious — and the agent
spends more on compute only in those cases.

## The agent's baseline

```python
agent = await server.create_agent(
    id="my-agent",
    name="Support Agent",
    instructions="...",
    effort=p.Effort.MEDIUM,   # MEDIUM is the default anyway
)
```

The baseline is where every turn starts. Rules can raise it for a single
turn, but never lower it.

## Raising effort from a rule

Any policy or observation can carry a `trigger_effort`:

```python
await agent.create_observation(
    condition="the customer disputes a charge or threatens legal action",
    trigger_effort=p.Effort.HIGH,
)
```

As long as this observation matches, the turn's **dynamic effort level** becomes
the highest value among the agent's baseline and all matched rules' triggers.
For example, a HIGH trigger on a MEDIUM agent gives a HIGH turn. Two
triggers do not add up beyond the highest one. A rule that matched on an
earlier turn, but not on the current one, has no effect on effort.

The raise takes effect immediately, in the same turn. In the example above,
as soon as the observation matches, the rest of the turn — tool selection,
response generation, reviews — runs at HIGH effort.

## What effort changes

**Which models run.** Effort reaches model routing as the `effort`
qualifier, so effort-qualified routes can select a different model per
level. The built-in providers' defaults already work this way: for example,
OpenAI's react default uses a smaller model at `min`/`low` and its largest
model at `max`. See [Configuring Models and Providers](providers.md).

```python
models={
    "react": p.ModelConfig("openai:gpt-5.6-luna"),
    "react[effort>=high]": p.ModelConfig("openai:gpt-5.6-terra", reasoning="high"),
}
```

**How deeply the responder reasons.** If a route does not set `reasoning`
explicitly, the responder's reasoning depth follows the effort level: minimal
and hidden at `MIN`, deeper at higher levels.

**How carefully rules are matched.** At higher effort, more rules get an
individual "does this actually apply right now?" check from a model, instead
of similarity search alone.

Which rules get the individual check depends on their weight:

| Effort | Individually checked | Similarity search only |
|---|---|---|
| `MIN` / `LOW` | — | all weights |
| `MEDIUM` | HIGH-weight rules | LOW, MEDIUM |
| `HIGH` / `MAX` | MEDIUM- and HIGH-weight rules | LOW |

Some rules always get the deeper check, at any effort level, because their
verdict affects other things, so a wrong match costs more.
For example: rules with labels, rules that other rules depend on, rules in priority
relationships, rules with tool associations, and rules that trigger reviews.

**Whether and how reviews run.** At `MIN` effort, no reviews run at all. At every
other level, review requests are honored, and higher effort allows more retries
after a review rejects a draft (2 at `LOW`, up to 10 at `MAX`).
See [Integrated Tool & Message Reviews](review.md).

**How history is compacted.** Higher effort keeps more raw history before
compacting it, and writes more detailed summaries when it does. See
[Session Compaction](compaction.md).

**Whether the turn is re-matched after a tool runs.** Above `LOW`, a step that
follows a tool call re-evaluates the rules that were gated on it, so rules the
tool's result made applicable can still join the turn. At `MIN` and `LOW` the
turn keeps the match it started with.

**How canned responses are chosen.** Above `LOW`, candidates are ranked by
relevance to the moment and the best few are shown. At `MIN` and `LOW` there is
no ranking — ranking embeds the customer's message and searches the vector
store — so every eligible response is shown instead, up to a larger cap. See
[Canned Responses](canned-responses.md).

## MIN: the cost floor

`Effort.MIN` is not just "a bit less than LOW". It is the level where Parlant
skips optional work completely: no reviews (even if requested), similarity
search only for all rule matching, and minimal hidden reasoning. Use it when
latency and cost matter most and the agent's instructions alone are enough.

## Reading the level

Inside a running turn, the resolved level is available as
`p.Session.current.effort`. This is the value that every effort-based
decision above uses, and your own code can use it too.

For example, a tool can scale its own work with the turn's effort, reading
it from `p.Session.current` (see [Tools](tools.md)):

```python
@p.tool
async def search_knowledge_base(query: str) -> p.ToolResult:
    limit = 20 if p.Session.current.effort >= p.Effort.HIGH else 5
    return p.ToolResult(data=await kb.search(query, limit=limit))
```

On a routine turn the tool fetches a few results. On a demanding turn it
searches more widely — without any change to the policy that calls it.

A [custom model provider](providers.md#custom-model-providers) can combine
the level with other conditions that route selectors can't express — for
example, use the expensive model only for premium customers on demanding
turns:

```python
async def provide_react(context: p.EngineContext | None) -> ReactGenerator:
    if context is None:
        return standard_react

    premium = await p.Customer.current.metadata.get("tier") == "premium"
    demanding = p.Session.current.effort >= p.Effort.HIGH
    return premium_react if premium and demanding else standard_react
```

(For plain by-effort model selection, prefer effort-qualified route keys like
`"react[effort>=high]"` — reach for code only when the logic needs more than
the selector grammar offers.)
