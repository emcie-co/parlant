# Canned Responses

A canned response is a business-approved example message — the exact wording,
tone, and structure you want the agent to use for given situations. You write it
once, mark where it applies — to specific rules, or across the whole agent — and
Parlant puts it in front of the model for consideration as the agent composes its reply.

Parlant uses a canned response as a **guiding example**. When one applies to the
situation at hand, the agent reads the approved message while writing its own
reply and reproduces its form and function — its tone, its phrasing, what it says
and leaves unsaid — shaped to fit the live conversation. The customer receives
the agent's own words, guided by your example rather than a fixed script.

This guide covers why the feature exists, the mental model, how to author canned
responses, the advanced knobs, and how the feature works conceptually — enough
to design for it and troubleshoot it.

## Motivation

Two forces pull in opposite directions when you build a customer-facing agent:

- **Control.** Compliance, brand, and legal teams need the agent to say certain
  things a certain way — a disclosure worded exactly so, a refusal that stays
  polite and on-message, an escalation that follows the script.
- **Fluency.** Real conversations don't fit a script. A rigid, always-verbatim
  agent reads as robotic, can't adapt to what the customer actually asked, and
  breaks the moment the situation is slightly off-template.

Fixed, always-verbatim messages give you the first at the cost of the second.
Free generation gives you the second at the cost of the first. Canned responses
aim for both: you supply the approved wording as a strong exemplar, and the model
reproduces its *form and function* while still writing a reply that fits the live
conversation. When an example fits perfectly, the model will naturally track it
almost word-for-word; when the situation has drifted, the model adapts rather
than forcing a mismatched script.

Because the same examples are shown during review, they also become a
**compliance yardstick**: the review stage can flag a drafted message that
diverges from the approved form and send it back for regeneration.

## Overview

### The mental model

Think of a canned response as a **worked example handed to two people**: the
writer (the component that composes the agent's response messages) and the editor
(the component that reviews it). Both are told "here is how the business wants this
kind of message to sound — match its form and function, honor its note, but you
don't have to copy it verbatim." Neither is told "send this text": the message the
customer receives is always freshly written, never a canned example pasted in directly.

Those two people act at two moments in a turn: when the agent **writes** its
reply, and when that reply is **reviewed** before it goes out. Both draw on the
same candidate set, chosen in three steps:

1. **Scope** decides the *pool*. A canned response is a candidate on a turn when it is
   either available **agent-wide** or attached to a **rule that matched** on that turn.

2. **Precedence** orders the pool into two tiers. Responses attached to a turn-matched
   rule come first; agent-wide responses are the fallback tier. A response that is both
   turn-matched to a rule, and assigned to the agent in general, counts as rule-scoped.

3. **Relevance** decides the *shortlist*. Parlant keeps the few most semantically
   relevant to the moment — the customer's latest message when generating, the
   drafted reply when reviewing — filling the budget from the rule-scoped tier
   first and letting agent-wide responses take only the slots that remain. The
   budget defaults to ten per turn for the responder and five for the reviewer,
   and is configurable (see below). Relevance is
   also what keeps an agent-wide example out of turns it doesn't fit.

### Key terms

**Canned response**
: A stored, business-approved example message with a `template` (its text), an
  optional `note`, optional retrieval `signals`, and
  optional fields it interpolates. Created via the SDK and made available
  agent-wide or attached to specific rules.

**Template**
: The canned response's text — a string that may contain `{{ field }}`
  placeholders for values resolved at render time (customer name, order id, …).

**Style reference**
: How the **message generator** sees canned responses — a prompt section of
  examples to take inspiration from while writing the reply.

**Canned guidance**
: How the **reviewer** sees them — a prompt section used to judge whether the
  drafted message conforms to the approved form.

**Field provider**
: A callback (or a tool result) that supplies values for a template's
  placeholders at turn time.

## Tutorial

### 1. Author a canned response and attach it to a rule

The most common case is a canned response tied to a specific situation. Create it
on the agent, then attach it to the rule whose situation it covers, so it is
offered when that rule matches.

```python
import parlant.sdk as p


async def main() -> None:
    async with p.Server() as server:
        agent = await server.create_agent(
            id="my-agent",
            name="Support",
            instructions="Handle billing questions.",
        )

        refund_reply = await agent.create_canned_response(
            template=(
                "I'm sorry for the trouble. I've started your refund of "
                "{{ amount }} — it should land in 5–10 business days."
            ),
            note=(
                "Use a warm, apologetic tone. Always state the 5–10 business "
                "day window. Never promise an exact date."
            ),
        )

        await agent.create_policy(
            content=(
                "When the customer is owed a refund and it has been approved, "
                "tell them their refund is on the way"
            ),
            canned_responses=[refund_reply],
        )
```

On any turn where that rule matches, the generator receives the refund reply as a
style reference and writes a message that mirrors its tone and keeps the "5–10
business days, no exact date" constraint — adjusting naturally to whatever the
customer actually said.

The `note` field is optional, but it does real work: it is shown next to the example,
and both the generator and the reviewer are instructed to honor it. Put any hard
constraints there, or explain anything about the response that needs special
attention from the agent.
If there is nothing special to say about the response, prefer to leave the note out,
since it increases costs via more input tokens.

### 2. Make a canned response available agent-wide

Some example messages aren't tied to one situation — a house greeting, a standard
sign-off, a way of phrasing "I can't help with that." Create these on the agent
and simply don't attach them to a rule; they become candidates on every turn:

```python
sign_off = await agent.create_canned_response(
    "Is there anything else I can help you with today?",
)
```

An agent-wide response is still only *shown* when it fits the moment — relevance
filtering (below) keeps it out of turns it doesn't belong in — so it stays out of
the way without you scoping it to a rule.

### 3. Filling in dynamic values in the templates through fields

Canned response templates can interpolate `{{ field }}` placeholders.
There are two ways to supply the values.

**A tool result.** A tool can return field values on its `ToolResult`, and they
are available to any canned response rendered on that turn:

```python
@p.tool
async def look_up_refund(order_id: str) -> p.ToolResult:
    refund = await billing.get_refund(order_id)
    return p.ToolResult(
        data={"status": refund.status},
        canned_response_fields={"amount": f"${refund.amount:.2f}"},
    )
```

**A field provider.** Attach a callback to the rule; it runs at turn time and
returns a mapping of field values:

```python
async def provide_refund_fields() -> dict[str, object]:
    amount = await billing.pending_refund_amount(p.Customer.current.id)
    return {"amount": f"${amount:.2f}"}


await agent.create_policy(
    content=(
        "When the customer is owed a refund and it has been approved, "
        "tell them their refund is on the way"
    ),
    canned_responses=[refund_reply],
    canned_response_field_provider=provide_refund_fields,
)
```

Either way, `{{ amount }}` is replaced before the example reaches the model. A
placeholder with no value renders back as the literal `{{ amount }}` text — a
useful signal that a field went unfilled.

## Advanced usage

### Add signals to improve retrieval

> **Not to be confused with** policy/observation `signals`
> ([Rule Discovery](rule-discovery.md#managing-custom-signals)). Those train
> the rule-discovery classifier and pair with `anti_signals`. Canned response
> signals, covered here, are simpler: alternate phrasings that help an example
> message get found — no training step, no anti-signals.

**The problem signals solve.** During shortlisting, Parlant matches the current
moment against each candidate's *template text*. That works when the template
reads like the situation it belongs to — but many good templates don't. Consider:

```python
await agent.create_canned_response(
    template="Your refund should arrive within 5–10 business days.",
)
```

A customer asking "where's my money??" shares almost no wording — and little
semantic overlap — with that answer-shaped sentence. The response is eligible,
but it can lose the relevance ranking to candidates that happen to read closer
to the customer's phrasing, and never make the shortlist.

`signals` fix this by giving the response alternate retrieval phrasings — the
things customers actually say when this response is the right one:

```python
refund_status = await agent.create_canned_response(
    template="Your refund should arrive within 5–10 business days.",
    signals=[
        "Where is my refund?",
        "When will I get my money back?",
        "My refund has not arrived yet",
        "It's been a week and I still don't see the money",
    ],
)
```

Now the customer's message matches a signal even though it doesn't match the
template, and the response surfaces.

**When to add them.** The practical flow is reactive: author the response
without signals, converse with the agent the way a customer would, and add
signals when you catch a turn where the response *should* have guided the reply
but didn't. That said, you can predict the misses — add signals proactively when
the template:

- **starts directly with the answer** ("Your refund should arrive…") — the
  question it answers isn't in the text;
- **uses internal or formal terminology** ("Your RMA has been initiated") when
  customers say "I want to send this back";
- **is too short to carry meaning** ("Done!", "You're all set") — nothing for
  relevance to grab onto.

Conversely, a template that already narrates its situation ("I'm sorry your
delivery is late — let me check what's holding it up") usually needs none.

**What to put in them.** Write each signal as a realistic customer message:
first person, phrased the way customers actually write — typos and frustration
included. Cover the *distinct ways* the situation gets phrased rather than
adding many variants of one phrasing:

- Good: `"I want to send this back"`, `"How do I return something I ordered?"`,
  `"This came damaged, I don't want it"` — three genuinely different routes into
  a returns response.
- Weak: `"return item"`, `"returns"`, `"refund policy return"` — these are
  search keywords, not messages. Signals are matched by meaning against real
  customer messages, so keywords give them little to match on.
- Wrong place: `"Always mention the 30-day window"` — that's behavioral
  guidance, which belongs in `note`. Signals never instruct the model; they
  only affect whether the response is shortlisted.

**What signals don't do.** They don't widen eligibility — a response still needs
its rule to match, or agent-wide scope, to be in the running at all — and they
don't change how the reply is written once the response surfaces. Scope decides
*if it can appear*, signals improve *whether it's found*, and `note` shapes
*how it's followed*.

### Other advanced controls

- **`field_dependencies` disqualify a response whose data premise doesn't hold.**
  A canned response can declare the fields its content *presumes*, independently
  of the `{{ placeholders }}` its template happens to use:

  ```python
  await agent.create_canned_response(
      template="Your order is ready for pickup.",
      field_dependencies=["order"],
  )
  ```

  Before shortlisting (both when writing and when reviewing), every declared
  dependency must be resolvable in the current context —
  from the `std` bundle, a tool result's `canned_response_fields`, or a matched
  rule's field provider. If any dependency is unmet, the response is excluded from
  the turn entirely: it never reaches the prompt, so the model can't assert facts
  nobody staged. Use this for responses that state data-backed outcomes ("your
  order is ready") so they only appear on turns where a tool or rule actually
  supplied that data.

- **Canned responses always guide writing; they only gate delivery when message review runs.** Review is active on a turn when the agent enables it by default
  or a matched rule requests it (and never at `Effort.MIN`). If you rely on
  canned responses as a *compliance gate*, make sure message review is enabled
  where it matters; otherwise they act purely as writing-time style guidance.

- **Tools and retrievers can supply literal responses for the turn.** A
  `ToolResult` / `RetrieverResult` can carry `canned_responses` — a list of
  **literal response texts** (not store ids). These become the turn's
  highest-precedence tier: they skip relevance ranking entirely (the tool
  supplied them for THIS turn, so they're already maximally relevant), fill
  the budget before rule-scoped responses, and carry a use-verbatim note.
  Everything else about candidate selection is scope-based: the pool is the
  matched rules' responses plus the agent-wide ones, and a tool's
  `canned_response_fields` (the value map) feeds rendering, not selection.

- **Agent-wide is the default scope for `agent.create_canned_response`.** Because
  that method also files the response under the agent, an SDK-created canned
  response is in scope on every turn even when you also attach it to a rule. Rule
  attachment therefore does not *restrict* where a response can appear — precedence
  and relevance decide that. In practice this rarely matters (a refund example
  simply won't be relevant on a billing-address turn), but don't rely on
  attachment as a hard gate.

- **A large library is fine, because precedence protects specific responses over general ones.** Irrelevant responses fall away in shortlisting, so there is no
  cost to keeping many. And because rule-scoped responses fill the budget before
  agent-wide ones, a big agent-wide library can't crowd out the response you
  attached to the rule that actually matched — even if an agent-wide one happens
  to read closer to the customer's phrasing.

- **The per-turn budget is configurable.** The responder surfaces at most ten canned
  responses per turn by default; the reviewer surfaces at most five. Raise or lower this through
  the DI container in the SDK's `initialize_container`, independently for each:

  ```python
  from parlant.core.engines.compass.responder.responder import Responder
  from parlant.core.engines.compass.reviewer import Reviewer


  async def initialize_container(container: p.Container) -> None:
      container[Responder].max_canned_responses = 8
      container[Reviewer].max_canned_responses = 3
  ```

  See [Dependency Injection](dependency-injection.md) for how `initialize_container`
  tunes finalized components.

- **The note travels with the example.** The `note` field is shown in both
  stages. This is the right place for constraints that aren't obvious from the
  wording ("never promise an exact date", "must include the reference number").
  The model is instructed to honor them even when it adapts the phrasing.

- **They are never shown to the customer directly.** The rendered example text
  exists only inside the generation and review prompts. Even a perfectly matching
  canned response reaches the customer only as text the model wrote — so a
  template that fails to interpolate (leaving a raw `{{ field }}`) will not leak
  that placeholder to the customer, but it *will* degrade the example the model is
  working from.

## How it works

On each turn, a canned response passes through four conceptual stages:
**eligibility → shortlisting → rendering → use** (in generation, and again in
review when it runs). Understanding these is enough to design for the feature
and to diagnose why a response did or didn't show up.

### Eligibility: is it in the running this turn?

A canned response is a candidate on a turn if it's available agent-wide or
attached to a rule that matched — and its declared `field_dependencies` (if any)
are all resolvable right now. That last check is a hard gate: a response that
presumes data nobody staged this turn is out entirely, so the agent can't assert
facts it doesn't have.

**Troubleshooting lens:** if a response never appears, check this stage first.
Did the rule it's attached to actually match? If it declares field dependencies,
did a tool or field provider actually supply those fields this turn?

### Shortlisting: which candidates make the cut?

Candidates compete for a small per-turn budget (default ten for the responder,
five for the reviewer), filled in strict
precedence order:

1. **Tool-provided literal responses** — a tool supplied them for this exact
   turn, so they skip ranking and go first.
2. **Rule-scoped responses**, ranked by semantic relevance to the moment.
3. **Agent-wide responses** take only the slots that remain.

Relevance is judged against a *query* that differs by stage: when the agent is
writing, candidates are compared to the customer's latest message; when the
draft is being reviewed, they're compared to the drafted reply itself.

**Troubleshooting lens:** if an eligible response still doesn't surface, it lost
on relevance — its wording is semantically far from how the customer actually
phrased the situation. Add `signals` with realistic customer phrasings to widen
its recall, or check whether a crowd of rule-scoped responses is exhausting the
budget ahead of it.

#### At MIN and LOW effort, nothing is ranked

Ranking embeds the customer's message and searches the vector store. That is
per-turn work the lowest effort levels exist to avoid, so on those turns
Parlant skips it: every eligible response goes to the agent, in the same
precedence order, up to a larger cap (20 by default). Nothing is left out for
being less relevant — only for not fitting the cap, which Parlant warns about,
naming how many were dropped.

The cap is a property of the responder, so you can set it wherever you have the
container:

```python
from parlant.core.engines.compass.responder.responder import Responder


async def initialize(container: p.Container) -> None:
    container[Responder].max_unranked_canned_responses = 8


server = p.Server(initialize_container=initialize)
```

The trade is that an example can appear on a turn it doesn't suit — a sign-off
during a dispute, say — because relevance filtering is what normally keeps it
out. Examples are style references rather than instructions, so the effect is
mild, but if your agent is on `LOW` and has a large agent-wide catalog, scope
some of it to rules so those responses only compete when their rule matches.
See [Adaptive Effort](adaptive-effort.md).

### Rendering: filling in the blanks

Shortlisted responses are rendered by a plain, deterministic template pass — no
model involved. `{{ field }}` values are resolved from the built-in standard
bundle, the turn's tool results, and field providers, in that order. A
placeholder nobody filled renders as its literal `{{ field }}` text.

**Troubleshooting lens:** a literal `{{ ... }}` can never leak to the customer —
rendered examples are prompt material only — but if you see one while debugging,
it means a field went unfilled, and the example the agent is learning from is
degraded. Trace which source was supposed to supply that field.

### Use: guiding the writer, calibrating the editor

The rendered examples reach the model twice, in two roles:

- **While writing**, they appear as style references: "here is how the business
  wants messages like this to sound — match their form and function, honor their
  note; copying verbatim is allowed but not required." The reply the customer
  gets is always the agent's own writing, guided by the examples.
- **While reviewing** (when message review is active for the turn), the same
  kind of shortlist — now matched against the *draft* — serves as the yardstick
  for whether the draft conforms to the approved form. A breach doesn't swap in
  a canned response; it sends the draft back for regeneration with the
  reviewer's feedback attached.

One design rule is worth remembering: **the message the customer receives always comes from the agent's own generation.** Canned responses
guide tone and compliance, but the model always writes the final text itself.
This is what lets an agent stay both on-message and genuinely
conversational. It also means canned responses are only a *compliance gate* on
turns where message review actually runs; on other turns they act as
generation-time guidance alone.
