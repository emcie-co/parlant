# Prompt Customization

Parlant writes its own prompts — for responding, matching rules, reviewing,
compacting — and for most deployments you never need to think about them.
Prompt customization is for the rare case where a built-in prompt says
something you need said differently.

Every prompt is built from **named, templated sections**, and you customize
by intercepting sections as they are added: keep, rewrite, or drop each one,
and leave everything else untouched. You choose *which* prompt to intercept
with the `prompts` parameter, using the same route keys as the `models`
parameter — routes decide which model a component calls *and* which prompt
builder it uses.

**When to use it.** Everything an agent should *do* belongs in instructions,
policies, and canned responses. Those work per situation, and they are where
almost all behavior shaping should happen. Prompt customization is for the
layer beneath: when a *built-in* principle — how Parlant frames the work
itself — conflicts with your deployment (brand voice, formality, language
rules, regulatory wording), or when platform boilerplate needs to go.

**The trade-off.** What you gain is precise, consistent control: you replace
one section, everywhere it is used, through a stable API, without forking
Parlant. What you take on is ownership: Parlant's built-in wording is
tuned and tested against its own components. A replaced section stops
receiving those upstream improvements, and it can shift behavior in ways
only your own testing will catch. So customize carefully, and keep it
small: replace the one section that is actually wrong for you, not
everything. Re-test your conversations whenever you change a model or an
override.

## How it works

Two pieces make up the whole mechanism:

- **A prompt provider** — any zero-argument callable that returns a
  `PromptBuilder`. A subclass qualifies, and so does a `lambda`. You register
  it under a route via `prompts={...}`, and the routed component builds its
  prompt through it.
- **`adapt_section`** — the one supported override point on your builder.
  The component adds its sections as usual, and `adapt_section` is called
  once per section with its name, template, and props. It returns one of
  three verdicts:
  - `None` — keep the section unchanged (the default for everything you
    don't match).
  - a `str` — replace the section's template. Keep every `{placeholder}`
    that the original template uses; Parlant supplies their values through
    props. A dropped placeholder means dropped content, and a missing one
    fails loudly at build time.
  - `p.VETO` — drop the section entirely.

**NOTE:** Do **not** override `add_section` itself. Its signature is internal
and free to change; `adapt_section` is the stable extension point, and it
receives everything you need — including `props`, for replacements that
depend on a section's actual content.

## A worked example: your own general instructions

The most common customization in practice is replacing the **responder's general instructions** — the platform-level principles every response
follows. They encode opinionated defaults that fit most deployments but not
all: for example, *"be concise and conversational and avoid being overly polite"*
(a private-banking agent may be expected to be formal and
courteous), *"avoid em dashes"* (your brand voice may embrace them), or
*"always respond in the language of the customer's last message"* (a
regulated business may be required to answer in one language only).
Overriding the section replaces those house rules with yours:

```python
import parlant.sdk as p


class MyResponderPrompt(p.PromptBuilder):
    def adapt_section(self, name, template, props):
        if name == p.PromptSections.Responder.TASK_DESCRIPTION:
            return MY_GENERAL_INSTRUCTIONS  # replacement template
        return None  # keep everything else as-is


server = p.Server(
    models={"*": "openai"},
    prompts={"react": MyResponderPrompt},
)
```

`react` is the responder's route; the section names come from the
`p.PromptSections` registry (both covered below).

## Choosing which prompts to customize

**By route.** Keys use the exact same selector syntax as `models`: concrete
routes, namespace wildcards (`"matching.*"`), the global `"*"`, and effort
qualifiers (`"matching.rank[effort=high]"`, `"matching.*[effort>=medium]"`).
Within a scope, the most specific selector wins: a concrete route beats a
namespace wildcard, which beats `"*"`, and a tighter effort qualifier beats
a looser one.

The routes that carry prompts:

`react`
: The responder. Both its system instructions and its per-turn instructions
  build through this route's provider.

`matching.rank`
: Per-rule turn evaluation (the rule ranker) **and** discovered-rule pruning
  (the rule pruner). The two share a route; providers tell them apart by
  section names (`p.PromptSections.Ranker` vs. `p.PromptSections.Pruner`).

`review.message` / `review.tools`
: Message and tool-call reviews.

`compaction`
: Session compaction (see [Session Compaction](compaction.md)).

**NOTE:** Embedding and tokenizer routes carry no prompts;
configuring providers for them has no effect.

**By scope.** Providers register server-wide (as above) or per agent, and
agent scope wins (it overrides any server-level configuration):

```python
agent = await server.create_agent(
    id="my-agent",
    name="Support Agent",
    instructions="...",
    prompts={"matching.rank[effort=high]": MyCarefulRankerPrompt},
)
```

## Finding the section names

Every section name is registered, stable API. Each component exposes a
`PromptSections` enum next to its prompt, and the SDK collects them under
one namespace — so autocompleting `p.PromptSections` shows you the full
registry:

- `p.PromptSections.Responder` — system and per-step instructions
- `p.PromptSections.Ranker` — per-rule turn evaluation
- `p.PromptSections.Pruner` — discovered-rule pruning
- `p.PromptSections.Reviewer` — both review prompts (`.MESSAGE_*` / `.TOOLS_*`)
- `p.PromptSections.Compacter` — session compaction
- `p.PromptSections.MatchingCommon` — fragments used by several prompts (rule terms,
  agent reasoning)

Match on enum members, not raw strings. A test in the repo guarantees that
no prompt contains an unregistered name, so the enums are complete. Sections
added by the shared builder helpers (agent identity, glossary, interaction
history, and so on) carry `p.PromptBuilder`'s `BuiltInSection` names.
`adapt_section` sees those too, so you can reword or veto them the same way.

## Rules to follow

- **Be deterministic.** Prompts are split into a stable cached prefix and a
  live tail, and the prompt cache keys on exact bytes. A prompt provider
  whose output changes between calls (timestamps, random ordering, mutable
  closures) silently disables prompt caching and raises your costs.
  Customize *how the prompt reads*, not *per-call content* — Parlant
  supplies the per-call content through props.
- **Providers run per prompt construction.** Keep the code cheap and fast to run;
  do any expensive preparation at import time.
- **Trace logging is preserved automatically.** Parlant attaches its
  `on_build` listener to whatever your provider returns — you don't need to
  forward anything, and your own listeners are composed with, not replaced.
