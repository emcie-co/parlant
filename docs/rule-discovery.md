# Rule Discovery

Parlant decides two different things about every rule, and it helps to keep
them apart:

- **Turn evaluation** asks: *does this rule apply to the current turn?* This
  verdict is short-lived. Parlant judges it again every turn.
- **Discovery** asks: *has the conversation made this rule relevant to the session?*
  This verdict sticks. A discovered rule joins the session's
  working set and stays there until pruned.

This document is about discovery: how it works, how the classifiers behind it
are trained and retrained, and how to steer it with custom signals.

The data flow is:

```text
Rule inventory (content + signals + anti-signals)
  -> Training (per-agent, offline)
  -> Per-rule classifiers
  -> Recall at turn time (embed conversation, score each rule)
  -> Session working set
  -> Responder instructions
```

## Key terms

| Term | Meaning |
|---|---|
| **Discovery** | The decision that a rule has ongoing relevance to the session. A discovered rule is admitted into the session working set. |
| **Recall** | The embedding-based mechanism that implements discovery. No LLM call is involved. Recall is a dot product against trained classifiers. |
| **Session working set** | The rules admitted to the session so far. Parlant renders them into the agent's instructions every turn, marked "may or may not be relevant right now — use your judgment." |
| **Signal** | A short text on a rule expressing a way the conversation might trigger it, typically a customer utterance. Signals are the classifier's positive examples. |
| **Anti-signal** | A near-miss: a text that *resembles* a trigger but must not fire the rule. Anti-signals are the classifier's negative examples. |
| **Pin** | A signal prefixed with `[__pin__]`. It trains the classifier like any other signal, and it also acts as a must-fire override: a customer message very close to the pin text recalls the rule no matter what the classifier says. |
| **Frame** | An agent's full set of trained per-rule classifiers. Frames are strictly per-agent and never shared. A rule's classifier is only meaningful relative to its own agent's inventory. |
| **Training** | The offline pass that derives every classifier from the current inventory. Safe to re-run at any time; retraining an unchanged inventory changes nothing. |
| **Pruning** | The post-response pass that caps the session working set. It evicts members that are no longer relevant to where the conversation is going. |

## Which rules go through discovery

From the reader's side, every rule falls into one of three groups:

- **Always rendered, never discovered.** Two kinds of rules are simply always
  in the agent's instructions, so there is nothing to discover:
  - **LOW-weight rules** — with one exception: a LOW rule that raises effort,
    gates a dependent rule, or sits in a priority relationship needs a real
    per-turn verdict, so Parlant matches it like any other rule.
  - **Rules whose condition refers to the agent's other policies** (for
    example, "anything not covered by another policy") and that carry no
    tools or handlers. Embedding recall cannot judge such a condition, so
    Parlant includes the rule statically.
- **Code-evaluated rules.** A rule with a custom `evaluation_func=` (a Python
  function) is judged by your code and never goes through discovery. It also
  receives no signal proposition: signals exist for embedding recall, and
  these rules don't use it.
- **Everything else is discovered.** Whenever such a rule is not yet in the
  session working set, recall scores it against the conversation.

Discovery is not the whole story for higher-stakes rules. Any rule whose
match has mechanical consequences — tools to offer, a review to trigger, an
`on_matched` handler, labels, dependents, a priority relationship, or a
raised effort level — also gets a real LLM verdict for the current turn. That
is turn evaluation. It runs alongside discovery, not instead of it.

Two shortcuts are worth knowing when debugging:

- **Small inventories skip recall.** If the agent has five or fewer usable
  rules, the whole inventory fits in the agent's instructions anyway. Rules
  that would have used plain recall are admitted outright, with no embedding
  at all. Rules that need an LLM verdict keep their full treatment.
- **Already-admitted rules skip recall too.** Discovery only judges rules
  that are not yet in the working set, so recall cost shrinks as the session
  matures.

### Discovered is not matched

Discovery puts a rule in front of the agent with "use your judgment." It does
not declare that the rule applies right now. Consequences that hang on a
*match* — running `on_matched` handlers, applying the rule's labels — need a
real turn-match verdict, which discovery alone never produces. `on_discovered`
(below) is the event that fires from discovery itself.

One more admission gate: **dependencies**. A rule created with
`depend_on(...)` joins the working set only after its dependency has actually
matched a turn in this session. Until then, the rule can score as Relevant in
every discovery batch and still stay out. A fresh discovery verdict is judged
each turn instead, so the rule enters the moment its dependency matches.

A rule with several dependencies waits for all of them, unless they were
declared with `depend_on_any`, where one is enough.

## How recall works

Understanding the mechanism at this level is usually enough to predict what
will and won't be recalled, and why.

### Training

For each rule, training builds a small linear classifier that separates that
rule's *triggers* from its *near-misses*:

- The **positive examples** are the rule's signals, including any pins. If a
  rule lacks either signals or anti-signals, its own content text (title,
  "When *condition*, then *action*", content) joins the positive set as
  well.
- The **negative examples** are the rule's anti-signals. If it has none,
  Parlant samples *other* rules' texts instead, so the classifier at least
  learns what distinguishes this rule from the rest of the inventory.

Note the consequence: for a typical rule, which has both signals and
anti-signals, **the classifier is trained on your signals versus your anti-signals. The rule's condition text itself is not a training input.** The
quality of recall for a rule is the quality of its signals.

All texts are embedded once and cached, so retraining an unchanged inventory
is cheap.

### Scoring at turn time

At matching time, Parlant embeds a set of *queries* derived from the
conversation:

- one cumulative query — the whole conversation so far, prefixed with the
  session summary if the session has been compacted — and
- one query per individual customer message.

Each candidate rule's classifier scores every query, and the rule takes its
**maximum** score across them. The rule is discovered when that maximum
clears the classifier's margin.

Max-pooling is what lets recall look at the whole conversation rather than only
the newest message: a rule whose trigger appeared earlier keeps scoring on that
earlier message, so it can still be discovered several turns later.

Staying in play is a separate matter. Once a rule is discovered, Parlant records
it in the session's working set and stops judging it — that is what makes
discovery *sticky* across turns. See
[Stickiness, pruning, and readmission](#stickiness-pruning-and-readmission).

Scores you see in logs are squashed to `[0, 1]`, with `0.5` the borderline:
above it the rule was discovered, below it it wasn't.

### Pins

A signal that starts with `[__pin__]` does double duty. It trains the
classifier like any other signal. On top of that, Parlant keeps the pin text
as a raw reference: if any customer message is *very* close to it (cosine
similarity above 0.9), the rule is recalled no matter what the classifier
decided.

Use pins for utterances that must never be missed. Expect the override to be
strict: 0.9 cosine means the customer nearly said the pin text itself, not
something on the same topic. For broader coverage, the pin already helps as a
signal — and more distinct signals help more.

Pins are always authored by you; automatic signal proposition never generates
them.

### Stickiness, pruning, and readmission

Discovered rules accumulate. To keep instructions bounded, Parlant caps the
working set: when it grows past 30 members, a post-response pruning pass asks
which members are still relevant to where the conversation is going, and
evicts the least relevant down to 20. HIGH-weight members and rules matched
in the current turn are never evicted, and pruning runs at most once per 10
customer messages.

An evicted rule is not gone, but Parlant remembers the eviction. Only
conversation that arrives **after** the eviction can readmit the rule. The
customer message that originally triggered it can no longer bring it back.
Without this, the sticky max-pool would readmit every evicted rule on its old
trigger, and pruning would never hold. If an evicted rule seems to be
ignoring an old message, that is this rule working as intended; a fresh
trigger readmits the rule immediately.

## Training and retraining

A rule's classifier is meaningful only relative to the whole inventory: other
rules supply negatives, and the embedding space is centered on the
inventory's average. So any change to the inventory — adding, editing, or
removing rules or their signals — calls for retraining. Training is offline,
per-agent, safe to re-run, and does not touch the LLM.

### When training runs

Parlant trains every agent automatically as part of startup (the "Training
classifiers" progress bar). Rules are defined in code, so changing them means
restarting the server — and the restart retrains everything. With hot reload
on (see [Hot Reload](hot-reload.md)), saving your file restarts and retrains
for you.

Trained classifiers live in memory, not on disk. After a server restart, the
startup training pass rebuilds them.

### Rules created after startup

If your code creates or edits rules while the server is already running,
recall keeps working rather than failing, but two cases behave differently:

- **A new rule** (an id the trained frame doesn't cover) makes Parlant train
  a fallback frame for the batch at matching time. Correct, but slower on
  first use, and the fallback cache is small.
- **An edited rule** (an existing id whose signals or content changed) is the
  dangerous case: the trained frame still *covers* that id, so recall keeps
  using the **old** classifier. Your edits silently have no effect on
  discovery until the next restart.

Both cases resolve on restart, when startup training runs again.

## Managing custom signals

### Where signals come from

> **Not to be confused with** canned response `signals`
> ([Canned Responses](canned-responses.md)). Those are alternate retrieval
> phrasings for surfacing an example message — no training, no anti-signals.
> Rule signals, covered here, are training data for the discovery classifier,
> come paired with `anti_signals`, and take effect on retraining.

Both `signals` and `anti_signals` are plain lists of strings on a rule:

```python
await agent.create_policy(
    content=(
        "When the customer reports a product defect within the warranty "
        "period, open a warranty claim and give them the claim number"
    ),
    signals=[
        "My blender stopped working after a month",
        "This arrived broken — can I get it repaired?",
        "[__pin__] I want to file a warranty claim",
    ],
    anti_signals=[
        "What does the warranty cover?",       # asking about terms, not claiming
        "How long is the warranty period?",
    ],
)
```

For every rule that uses the default evaluator, Parlant also **proposes**
signals at startup: an LLM generates ten signals and ten anti-signals from
the rule's content, and they are merged into the rule — your authored entries
first, generated ones appended, duplicates removed case-insensitively. Your
signals are never dropped or rewritten. Proposition results are cached, so
this costs LLM calls only when a rule's content changes.

This means a typical rule already has a solid training set with no work from
you. Add custom signals when you have knowledge the generator lacks — the
phrases your real customers use.

### What to expect from custom signals

- **Write signals as customer utterances.** Queries are built from
  conversation messages. A signal phrased the way a customer actually talks
  ("my blender stopped working") lands closer in embedding space than an
  abstract description ("product defect scenario").
- **Anti-signals are for near-misses, not opposites.** The classifier learns
  the *boundary* between signals and anti-signals. "What does the warranty
  cover?" is a useful anti-signal for a claims rule because it is nearby
  (warranty-flavored) but must not fire it. A random off-topic sentence
  teaches the classifier nothing it didn't know.
- **An over-recalled rule needs anti-signals, not fewer signals.** Take the
  messages that wrongly triggered it and add them (or paraphrases) as
  anti-signals, then retrain.
- **An under-recalled rule needs more diverse signals.** Cover the distinct
  phrasings that should trigger it; ten paraphrases of one sentence are worth
  much less than five genuinely different triggers.
- **Signals compete within one agent only.** The same rule attached to two
  agents trains against two different inventories and can behave differently.
  That is by design.
- **Observations take signals too.** Observations are rules and are
  discovered the same way; `create_observation` accepts the same `signals`
  and `anti_signals` parameters, with the same proposition-merge behavior.

(For when to reach for signals at all — versus rewording the rule's
condition — see [Policy Design](policy-design.md#when-a-policy-doesnt-come-up--or-comes-up-too-much).)

### Reacting to discovery

Because discovery is the "this rule just became relevant to the session"
event, the SDK lets you hook it directly. A realistic use: tag the session
the moment a topic enters it, so your dashboards and queues see it without
waiting for a confirmed match:

```python
async def on_warranty_discovered(
    ctx: p.EngineContext,
    evaluation: p.RuleEvaluation,
) -> None:
    await p.Session.current.labels.add("warranty-claim")

await agent.create_observation(
    condition="the customer is reporting a product defect",
    on_discovered=on_warranty_discovered,
)
```

`on_discovered` fires when the rule **newly enters** the session working set:
once per admission, not once per turn. If the rule is later evicted by
pruning and readmitted by a fresh trigger, the handler fires again.

Choose the hook by cost. `on_discovered` rides on recall, which is cheap.
`on_matched` needs a confirmed per-turn verdict, so attaching it makes
Parlant judge the rule with the LLM evaluator every turn. Use `on_matched`
when the reaction must only happen if the rule truly applies right now; use
`on_discovered` when "the topic has entered the conversation" is enough.

## Debugging recall

Parlant logs every discovery batch at debug level — grep the logs for
`Rule discovery results` to see each candidate's score and verdict:

```text
... Rule discovery results:
### 1 Open a warranty claim [Score: 0.82 (Relevant)]
    Condition: the customer reports a product defect
    ...
### 2 Explain warranty terms [Score: 0.31 (Not Relevant)]
    ...
```

Common symptoms and their usual causes:

| Symptom | Likely cause |
|---|---|
| A rule is always included, never scored | The agent has ≤ 5 usable rules (small inventories skip recall), the rule is LOW weight, or its condition refers to the agent's other policies (both are always rendered). |
| A rule never appears in discovery results | It has a custom `evaluation_func=` (code-evaluated rules skip discovery), or it is already in the session working set. |
| A rule scores as Relevant but never joins the working set | It has a `depend_on(...)` target that has never actually matched a turn in this session. |
| A rule shows `Score: 0.00 (Relevant)` | The discovery batch held only that one rule and it has no anti-signals, so Parlant admitted it without scoring. Normal. |
| Signal edits seem to have no effect | The server hasn't restarted since the edit — recall still uses the classifiers trained at startup. Restart (or save the file under hot reload). |
| A rule fires on the wrong messages | Signals too generic. Add the offending messages as anti-signals and retrain. |
| A rule misses phrasings it should catch | Signal set too narrow. Add genuinely different trigger phrasings; pin the one that must never miss. |
| A pin doesn't fire | The 0.9 cosine threshold is strict — the customer's message must nearly repeat the pin text. |
| An evicted rule won't come back on an old topic | By design: only conversation newer than the eviction can readmit it. |
| A rule lingers after its topic has passed | Stickiness is by design; the working set is only pruned above 30 members, and at most once per 10 customer messages. |
| Tools or `on_matched` didn't fire although the rule was discovered | Discovery is not a match. Those consequences need a per-turn match verdict (see "Discovered is not matched"). |
