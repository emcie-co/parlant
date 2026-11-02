# Writing Parlant Docs

This guide collects the rules we apply to everything under `docs/`. It exists
because the same problems keep coming back in review: prose that reads like
marketing, examples that can't actually run, terms used before they're
introduced, and claims that no longer match the code.

Read it before writing or editing a doc. The checklist at the end is the
short version.

## Who you are writing for

Two audiences read these docs, and both matter:

- **Integrators** building on Parlant. They know the product as "Parlant".
  They do not know our module names, class names, or internal architecture.
- **AI agents** writing Parlant code for those integrators. They copy our
  snippets and imitate our patterns.

The second audience raises the stakes on examples. A half-designed snippet
does not just confuse one reader; it gets reproduced at scale.

## Voice: Parlant is the actor

In user-facing docs, Parlant does things. Internal component names are
architecture, not vocabulary.

```
Don't:  The matcher routes every usable rule by cost tier.
        Compaction is the compass engine's answer to long sessions.
Do:     Parlant routes every usable rule by cost tier.
        Compaction is how Parlant handles long sessions.
```

Class and function names stay as they are inside code blocks and when you
name a real API (`p.CompassOptimizationPolicy`, `CompassHooks`). The rule is
about prose.

Internal docs (`docs/internal/`) may name components freely — that is their
subject.

**Never frame behavior against a version the reader never used.** "There is
no longer a `condition` field" tells a new reader nothing except that they
missed something. State what is true now.

**Document only what ships here.** `docs/*.md` describes this repository. If a
capability is not in this tree, it does not belong in these docs.

## Register: plain, human, and confident

Write in plain English, roughly CEFR B2: short sentences, common words, one
idea per sentence. Many readers are not native English speakers, and dense
idiomatic prose costs them comprehension.

Explain like a person explaining something simply — not like a tech marketer.
The failure mode to avoid is the clipped fragment used as a slogan:

```
Don't:  Same process, same PID, brand-new server.
Do:     The process and PID stay the same; the server inside it starts
        completely fresh.

Don't:  Usage is tokens; cost needs prices.
Do:     Usage records count tokens. To turn tokens into cost, Parlant needs
        prices.

Don't:  **Same database, no drift.** Your store reads the same user database
        that issues the tokens.
Do:     **Your tokens and your customers come from the same place, so they
        can't disagree.** The backend that issues login tokens and the store
        that answers "does this customer exist?" both read the same user
        database. Any customer who can log in is a customer your store can
        find.
```

Note what changed in that last one: the heading became a full statement, and
the body explains *why* instead of asserting a benefit.

**This is not a ban on confidence.** Strong positive claims in full sentences
are wanted:

- "Reviews are one of the most powerful features of Parlant."
- "Parlant is a unique harness, because it's built to offer flexibility for
  engineers."
- "Parlant stores **no token state whatsoever**."

Keep the strength and the personality. Cut the fragments and the hype words
(`seamless`, `blazing`, `effortless`, `just works`, `magic`, and `simply`
used as filler).

Other register habits worth breaking:

- **Em-dash asides that carry the core meaning.** If the point is inside the
  dashes, it deserves its own sentence.
- **Jargon where a plain word exists**: "seams" → "connection points",
  "knobs" → "settings", "sugar" → "shortcuts", "by construction" → just give
  the reason, "the stalest" → "the least relevant".
- **Invented verbs and metaphors**: "an evicted rule is *ledgered*", "the set
  would *flap*", "resurrect every evicted rule". Say what happens.
- **Rhetorical questions nobody asked.** Do not open a section with "So what
  happens when effort rises mid-turn?" if the reader had no reason to wonder
  yet. Lead with the fact.

## Structure

The default shape of a feature doc is
**motivation → mental model → tutorial → reference**.
Readers arrive wanting to know what a thing is for before they can absorb how
it works.

**Cover when to use it, and the trade-offs, in the intro.** A feature doc
that only explains mechanics leaves the reader unsure whether they need the
feature at all. Prompt customization opens by saying it is for the rare case
where a built-in prompt says something you need said differently, then names
what you gain (precise control through a stable API) and what you take on
(you stop receiving upstream improvements to that section).

**General case first, special cases after.** The rate-limits section used to
open on indexing, which is one specific application. It now starts with what
a rate limit is and how it's configured, and indexing appears later as the
notable case.

**Don't dive too deep too early.** An overview that jumps into "the most
common customization is replacing the responder's general instructions" is
already three levels down. The overview establishes the concept; specifics
come next.

**Lead with the recommended path and rank the options.** `docs/customers.md`
lists four ways to provision customers, and says outright that plugging in
your own `CustomerStore` is the recommended one and why, before covering the
other three.

**Put the load-bearing distinction where the reader needs it.**
"Attachment means the inbound side only; the outbound side is just data a
tool returns" started life as a closing footnote, after a whole outbound
section a reader could spend confused. It belongs in the overview.

## Introduce concepts before using them

Read your doc as someone who arrived from a search engine. Every term you use
must already be defined, or defined right there.

Real misses we fixed:

- `docs/customers.md` used group ids like `grp_vip` in HTTP examples, but only
  explained where groups come from near the end. Now the intro says groups are
  defined in code with `server.create_group`, and that ids like `grp_vip` come
  from those definitions.
- The same doc used "provisioning" as a section title without ever defining
  it. It now opens: "Provisioning means making sure the customer record
  exists before it is needed."
- An auth passage used `JWT`, `sub`, and "minted" cold. It now reads "a JWT —
  the signed login token your backend creates for them" and "the customer id
  in the token (its `sub` claim)".
- A section used "observation" as though the reader knew it. It now carries a
  one-line definition plus a link to Policy Design.

Coining a term is fine; using a coined term as though it were standard is not.
"Ambient accessors" was our own invention, never defined anywhere, and it
appeared in eight places across the docs. We replaced it with the thing
itself: "the `.current` accessors", `p.Session.current`.

## Code examples

This is where the bar is highest, because agents copy these.

**Simulate the turn before you publish.** Ask: when this policy matches or
this tool runs, what exactly happens next? If the agent would have to
improvise, or the conversation dead-ends, the example is broken — no matter
how good it reads.

```python
# Don't: an instruction written for a human employee handbook. The agent has
# no approval channel and no concrete move, so the turn dead-ends.
await agent.create_policy(
    content="When the customer requests a refund above $500, require manager approval.",
)

# Do: name the agent's actual in-conversation move.
await agent.create_policy(
    content=(
        "When the customer requests a refund above $500, don't process it. "
        "Tell them it needs a manager's review and they'll hear back within "
        "one business day."
    ),
    labels=["manager-review"],  # the managers' queue: sessions awaiting review
)
```

**A tool's `data` is what the model reads.** Return what the model should act
on or say — never an internal state dump you added to demonstrate an API.

```python
# Don't: the label list is internal bookkeeping; the model gains nothing.
return p.ToolResult(data={"labels": sorted(session.labels)})

# Do: tell the model what happened, so it can tell the customer.
return p.ToolResult(
    data=(
        "Transfer requested. Let the customer know a human agent "
        "will pick the conversation up shortly."
    ),
    control={"mode": "manual"},
)
```

**Close every loop.** If something waits on a human, show the mechanism (a
label as the queue, metadata as the reason, `mode="manual"` to park the
session) and how it resolves, even when the resolution happens out of band.

**Every field must earn its place.** A reviewer should not be able to ask
"why is that there?" and get no answer. We deleted a `note=` from an
observation example because the note only restated the condition. If a field
is in the example, the prose should be able to justify it.

**Prefer one coherent scenario per doc** over scattered toy cases.
`docs/sessions.md` uses human handoff throughout: the tool requests the
transfer, the labels form the operator queue, metadata records why, and the
HTTP section shows the same handoff from the remote side. Each example
reinforces the last.

**Don't inherit examples from a doc's previous version unquestioned.**
That is how filler survives rewrites.

**Handle the unhappy path the way real code should.** A tool that raises a
bare exception teaches the wrong pattern:

```python
# Don't: the model just sees an error and has to improvise.
raise Exception("No customer message yet")

# Do: tell the model what to do next.
return p.ToolResult(
    data="No photo found on the customer's message. Ask them to attach a photo of the problem."
)
```

**Examples must run against the shipped API.** Check the signature before you
publish. Two real failures: an example called
`client.sessions.create_event(attachments=[...])` when no shipped client had
that parameter (it would `TypeError` for every reader), and several examples
declared a `context: p.ToolContext` first parameter that had become optional,
teaching an obsolete idiom.

## Accuracy: verify against the source

Every factual claim in a doc is a claim about code. Check it. Reviews of our
own docs turned up:

- Pins described as *not* feeding the classifier, when they train it like any
  other signal *and* act as an override.
- `allow_greeting` documented as unused, when it starts a turn on session
  creation.
- "Caption is the only field the agent model can read", when the model also
  sees the filename and an image/file marker.
- A `tool_id` example showing `generate_invoice` when tool events carry the
  qualified `built-in:generate_invoice`, so any frontend matching the doc
  would match nothing.
- An undocumented permission (`create_customer_event_with_attachments`) whose
  absence returns 403.

When a doc states a threshold, a default, or an ordering, cite it from the
code — not from memory or from the previous version of the doc. When you
describe a rename or a removal, grep for stragglers instead of assuming.

## Cross-references and filenames

- **File names are kebab-case**: `adaptive-effort.md`, not
  `adaptive_effort.md`.
- **Link with titled markdown links**, not bare backticked paths:
  `[Sessions](sessions.md)`, not `` `docs/sessions.md` ``. Use paths relative
  to the linking file (`../sessions.md` from `docs/internal/`).
- **Anchors must resolve.** A link to `#managing-custom-signals` has to match
  a real heading.
- **Point at the right doc.** A claim about tool-result metadata should link
  to the tools doc, not to an internal doc that happens to mention it.

## Markdown mechanics

**Close every emphasis span on the line it opens.** Markdown does not render
`**bold` on one line and `text**` on the next. This bites constantly when you
reflow paragraphs:

```
Don't:
    Note the consequence: **the classifier is trained on your signals versus
    your anti-signals.**

Do:
    Note the consequence: **the classifier is trained on your signals versus your anti-signals.**
```

The same applies to `*italics*`, `__underline__`, and `` `inline code` ``.
When a span makes the line too long, move the whole span to the next line
rather than splitting it.

Also: wide tables and long code lines need a container that scrolls, so keep
table cells short and put the explanation in the surrounding prose.

## Before you publish

- [ ] Every claim checked against the code, not remembered.
- [ ] Every code example simulated: what happens on the turn it runs?
- [ ] Every field in every example justified by the prose.
- [ ] Every term defined before it is used.
- [ ] Intro covers what the feature is for, when to use it, and the trade-offs.
- [ ] General case before special cases; recommended option first.
- [ ] Parlant is the actor; no internal component names in prose.
- [ ] Plain English, full sentences, no marketing fragments and no hype words.
- [ ] Confident claims kept — the pass should not flatten the doc's voice.
- [ ] All emphasis and code spans close on their own line.
- [ ] Links use titled markdown links, resolve, and point at the right doc.
- [ ] Nothing documented that does not ship in this tree.
