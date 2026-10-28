# Matching (Compass) — Internal Engineering Guide

This document explains how the Compass engine decides which rules participate
in a turn: why matching exists, what makes it challenging, the vocabulary the code
uses, and how the evaluation phase works end to end. Code anchors are given as
module references; the terminology sections in `matching/common.py` and
`matching/evaluation_scheduler.py` are the canonical in-code glossaries.

## 1. Motivation for matching

An agent's rule inventory grows far beyond what can — or should — be pushed
into a single prompt:

- **Instruction-following degrades with volume.** Models follow ten focused
  instructions far better than two hundred ambient ones. Matching keeps each
  turn's prompt down to the rules that actually govern it, rendered with
  per-turn rationale, so the responder's attention lands where it matters.
- **Cost and latency are proportional to prompt size.** Rendering the whole
  inventory every turn multiplies token spend for no behavioral gain.
- **Prompt-cache stability.** The responder's system prompt is prefix-cached.
  A carefully managed, slowly changing rule set keeps that prefix stable
  across turns; naive per-turn rule churn would defeat the cache entirely.
- **Attachments need a trigger.** Tools, canned responses, handlers, review
  requests and triggered effort all fire *when a rule matches*. Matching is the
  event that makes a rule's machinery actionable, not just its text visible.

The matcher's job, in one sentence: **put the right rules in front of the model at the right moment, with costs (AI spend) proportional to the stakes.**

### The shape of a turn

Where matching sits in the engine's processing of one customer turn:

```text
preload            load inventory + discovered set + statically-included set
  │                (before warm-up, so cached prompt prefixes stay stable)
warm_up            prefill the evaluators/responder prompt prefixes
  │
match_early        deterministic EARLY-phase code evaluators (e.g. budget
  │                predicates) — before any chargeable work
fill               ┌ rule matching (this document)
  │                ├ tool recall          (matching/tool_recaller.py)
  │                └ glossary recall      (matching/glossary_recaller.py)
  │                all three in parallel, then tool selection
respond loop       the responder generates; after each tool-running step,
  │                update() re-evaluates rules gated on the tools that just
  │                ran (REEVALUATION relationships) and re-selects tools
finalize           prune the discovered rule set and session glossary
                   (post-response, so cache misses land between turns)
```

This document covers the **rule matching** box; tool recall and glossary
recall are structural siblings with their own recall/stickiness designs.

## 2. Challenges of matching (broadly)

- **Judgment quality vs. cost.** LLM verdicts are accurate but cost a call per
  rule per turn; embedding classifiers are nearly free but blind to nuance —
  and structurally blind to some condition types (see cross-referential rules
  below). The engine constantly arbitrates between the two.
- **Conversations are stateful.** A rule that mattered three turns ago usually
  still matters ("I want a refund" doesn't expire when the customer answers a
  follow-up question). Relevance must be sticky — but unbounded stickiness
  bloats the prompt, and careless eviction causes flapping (a rule leaving and
  re-entering on its original trigger).
- **Rules are interdependent.** Dependencies gate rules on other rules;
  priorities arbitrate between simultaneous matches; cross-referential rules
  *mean* different things depending on what the rest of the inventory says.
  Judging each rule in isolation is therefore sometimes simply wrong.
- **Everything must degrade gracefully.** Matching sits on the critical path
  of every turn. Optimizations (scheduling, caching, batching) must fail open
  into simpler behavior, never into a broken turn.

## 3. Terminology of matching

**Inventory / usable rules**
: Every enabled rule available to the agent this turn (`state.usable_rules`).

**Discovered rules**
: The session working set: rules admitted on this or an earlier turn,
  persisted in session metadata, rendered every turn with "may or may not be
  relevant — use your judgment."

**Pending admission**
: Rules fresh discovery has judged relevant this turn but that have not yet
  passed final resolution (`MatcherState.pending_discovered_rules`).

**Turn-matched rules**
: Rules with a positive per-turn verdict from an evaluator. Rendered
  prominently with rationale; all attachments key off this.

**Evaluation**
: The act of judging a rule (by any evaluator). Produces a verdict, not
  necessarily a match.

**Evaluator**
: A component that judges rules for the current turn: the ranker, a
  distiller, or a code (function) evaluator.

**Discovery / recall**
: The embedding-based mechanism that admits rules into the discovered set
  (see [Rule Discovery](../rule-discovery.md) for the full treatment).

**Weight**
: A rule's importance tier (LOW/MEDIUM/HIGH). Decides evaluation strategy
  together with effort, exempts from pruning, scales reviewer reasoning.

**Candidate**
: A rule that will actually be evaluated this matcher call. Scheduling
  thresholds count candidates, not the inventory.

**Cross-referential rule**
: A rule whose meaning depends on the agent's *other* rules, flagged via
  `rule.metadata["references_other_policies"]` (classified at build time by
  the `RuleRelativityProposer`; author metadata overrides).

**Attachment**
: Anything mechanical that fires only on a turn match: tools, dependencies,
  priority relationships (either side), labels, triggered effort, a
  non-default priority, review requests, match/discovery handlers, code
  evaluators, and an explicit internal `evaluation_method` metadata entry
  (`"rank"` or `"distill"` — stamped by the SDK for rule-attached
  retrievers, and the author-facing way to force per-turn verdicts).

**Static inclusion**
: The treatment of *bare* cross-referential rules (no attachments): always
  rendered in the domain instructions with their declared weight (a
  LOW-weight rule carries the suggestion note, a HIGH-weight rule keeps the
  high-weight note), never discovered or ranked.

**Policy context**
: The set of other rules rendered into a cross-referential rule's evaluation
  prompt so its references can be resolved.

**Policy landscape**
: The inventory-size regime, split by `MAX_FULL_POLICY_CONTEXT_RULES`
  (`matching/common.py`): small → full-inventory policy context, no ordering
  needed; large → bounded context, evaluation deferred until discovery
  reports.

**Precondition / evidence / round**
: Scheduling vocabulary — see the glossary at the top of
  `matching/evaluation_scheduler.py`.

## 4. Evaluation phase

### i. Overview of evaluation

Each turn, the matcher (`compass/matcher.py`):

1. **Preloads**: the inventory, the discovered set (from session metadata),
   and the statically-included set (bare cross-referential rules — computed
   before prompt warm-up so cached prefixes stay stable).
2. **Selects candidates**: LOW-weight and statically-included rules are
   excluded (they render unconditionally), except that LOW rules which can
   raise effort or which serve as dependency targets are evaluated;
   code-evaluated rules route to the function evaluator; the rest are bucketed
   by strategy — RECALL (discovery only) or RANK (discovery *plus* an LLM
   verdict). Rules evaluated in `match_early` (deterministic EARLY-phase code evaluators)
   are excluded here — their verdicts were already recorded before any
   chargeable work began.
3. **Schedules**: the evaluation scheduler builds a plan (see §v). For most
   turns the plan is one pass; with preconditions in play it becomes bounded
   rounds.
4. **Runs rounds**: within a round, code evaluation, discovery, and LLM
   evaluators all fan out concurrently. Between rounds, discovery's fresh
   admissions land in `pending_discovered_rules` and the plan observes each
   evidence source's results.
5. **Resolves** (see §vi) and **records**: entailment expansion, dependency
   filtering, and priority arbitration produce the final turn-matched and
   session-admitted sets, which are written to state and session metadata.

### ii. "Evaluated" vs. "matched"

*Evaluation* is the judgment; *matching* is one possible outcome. Every match
comes from an evaluation, but most evaluations end in "not this turn" — and
that is normal operation, not failure. There is also a third outcome that is
neither: **session admission**, where discovery deems a rule relevant to the
conversation without any claim about the current turn. Keeping these three
apart (judged / turn-matched / admitted) is the single most useful mental
distinction when reading matcher code or logs.

### iii. Discovered rules vs. turn-matched rules

#### a. Purpose of discovery

Discovery answers "has the conversation made this rule relevant to the
*session*?" — a sticky, cheap, embedding-based judgment (per-rule trained
classifiers over signals/anti-signals, with pin overrides; full mechanics in
[Rule Discovery](../rule-discovery.md)). Discovered rules render every turn as judgment
guidance for the responder. This gives the engine ambient awareness at near
zero marginal cost, and its stickiness is what keeps the responder's cached
prompt prefix stable across turns.

#### b. Pruning

Unbounded stickiness would bloat the prompt, so the discovered set is capped:
above `_DISCOVERED_RULES_HIGH_WATER_MARK` (30) the pruner asks which members
still matter going forward and evicts the stalest down to
`_DISCOVERED_RULES_TARGET` (20). HIGH-weight members and this turn's matches
are exempt; pruning runs at most once per 10 customer messages so cache
misses are batched. Evicted rules enter an **eviction ledger** with the
offset at eviction — they may only be *readmitted* by conversation newer
than that offset, so the sticky max-pool can't resurrect them off their
original trigger (anti-flapping).

#### c. Turn-matched rules

A turn match is an explicit verdict: *this rule governs this turn*. Matched
rules render prominently with the evaluator's rationale, and everything
mechanical fires from them: tools join the offered catalog, canned responses
become style references and review guidance, `on_matched` handlers run,
review requests arm, effort is raised, and priorities arbitrate.

One rendering edge case: the step reminders normally reference a titled
matched rule by title, trusting its full content to sit in the system-wide
instructions. But those are built once, at turn start — and a rule can match
for the *first* time mid-turn (its `reevaluate_after` tool just ran) and be
admitted only then. For the remaining steps of that one turn, the reminder
renders the rule's full content inline instead of a by-title reference the
model couldn't resolve (`rendered_rule_ids` snapshot in response state);
from the next turn on, the durable admission puts the content in the
system-wide instructions and the reminder reverts to by-reference.

#### d. Rule weight and the strategy matrix

Weight is the author's stakes declaration. Together with the turn's dynamic
effort level it selects the evaluation strategy (`_get_strategy`):

- **MIN / LOW effort**: MEDIUM and HIGH weight both use RECALL.
- **MEDIUM effort**: HIGH weight earns RANK; MEDIUM weight stays at RECALL.
- **HIGH / MAX effort**: MEDIUM and HIGH weight both use RANK.
- **LOW weight, at every effort**: excluded from matching entirely and
  rendered in the system instructions unconditionally — nothing to decide,
  so nothing is spent deciding it. Triggered effort on a LOW rule forces real
  evaluation, since its match changes the turn's economics; serving as a
  dependency target does likewise, since its verdict gates another rule, as
  does participating in a priority relationship, since suppression operates
  on verdicts.

Three families of overrides sit on top of the matrix:

- **The small-inventory fast path** (`_SMALL_INVENTORY_THRESHOLD`, 5): at or
  below five usable rules, would-be-RECALL rules skip discovery entirely and
  are admitted wholesale (strategy NONE) — the whole inventory fits in the
  responder's instructions, so a rule the recaller rejected would be
  rendered anyway, and the embedding round trip buys nothing but latency.
- **Escalations to RANK, regardless of the matrix**: rules with labels,
  triggered effort, review requests, match handlers, the
  cross-referential flag, which serve as **dependency targets**, or which sit
  on **either side of a priority relationship** are always ranked — their
  match verdicts carry consequences (analytics, gated follow-ups, armed
  reviews, priority suppression) that embedding recall cannot be trusted
  with. For priority, both sides need verdicts: the source's decides whether
  suppression fires, the target's whether there is anything to suppress.
  Merely depending on another rule does not itself earn RANK: the source rule
  keeps its ordinary matrix strategy unless another reason escalates it, while
  the target is ranked because its verdict gates the source.
- **Effort-sensitive tool escalation**: a tool attachment guarantees at least
  RANK from MEDIUM effort upward. At MIN and LOW, a tool alone does not promote
  its rule beyond the ordinary matrix (or the small-inventory fast path), but
  any independent escalation above still applies.

Weight is also consumed outside strategy selection: HIGH exempts from
pruning, and the reviewer scales its reasoning depth by whether high-weight
rules matched the turn.

#### e. Highlights and instruction reminders

Evaluators return **highlights** with a positive verdict: the specific points
of a (typically long) rule that are relevant *right now*. For complex or
content-only rules, the matched-rule rendering becomes an **instruction reminder** — `### Review the instructions under "<title>"` plus the
highlight bullets — pointing the model back at the rule's full text in the
(cached) system instructions instead of duplicating it in the turn section.
This preserves the prefix cache, keeps the turn prompt focused, and directs
attention to the currently-relevant fragment of a large rule rather than the
whole wall of text.

### iv. Evaluation methods

Verdicts come from three kinds of sources. **Code (function) evaluators** —
Python callables registered via the SDK's `evaluation_func=` and resolved
through the `RuleEvaluatorRegistry` — are authoritative: a rule with one is
never discovered or LLM-evaluated, its author's code decides. **Discovery**
produces admissions, not verdicts (§iii.a). The **LLM evaluators** below
produce the turn verdicts for RANK-tier rules; the matcher is natively
multi-evaluator (`_select_evaluator` is the routing seam subclasses extend).

#### a. Custom evaluation functions

Pass an async evaluation function when creating a policy or observation. It
receives the current `RuleEvaluationContext` and the rule being
evaluated, and must return a `RuleEvaluation` with that rule's ID:

```python
import parlant.sdk as p


async def evaluate_refund_request(
    ctx: p.RuleEvaluationContext,
    policy: p.Policy,
) -> p.RuleEvaluation:
    message = ctx.session.interaction.last_customer_message
    is_relevant = bool(message and "refund" in message.content.lower())

    return p.RuleEvaluation(
        id=policy.id,
        is_relevant=is_relevant,
        reasoning=(
            "The customer requested a refund."
            if is_relevant
            else "The customer did not request a refund."
        ),
        highlights=(
            "Refunds require proof of purchase.",
            "Refund requests must be made within 30 days.",
        ) if is_relevant else (),
    )


await agent.create_policy(
    title="Refund policy",
    content=(
        "Refunds require proof of purchase. Refund requests must be made "
        "within 30 days. Opened consumable products cannot be refunded."
    ),
    evaluation_func=evaluate_refund_request,
)
```

`highlights` is optional. On a positive verdict for a MEDIUM- or HIGH-weight
rule, each returned string becomes a bullet in the matched-rule instruction
reminder described in §iii.e. Use it to select the parts of a long rule that
matter to the current turn; it should reinforce the rule's content rather than
introduce unrelated instructions. A negative verdict is not rendered, so it
normally returns no highlights.

The context also exposes the session, agent, customer, loaded variables,
staged events, and `staged_tool_calls`, allowing deterministic evaluation from
application state or tool results. Because a custom evaluator is authoritative,
its rule skips discovery and LLM evaluation.

#### b. Ranking

`RuleRanker` (`matching/rule_ranker.py`) scores one rule per prompt on a 1–5
relevance scale, framed as a *first-pass screen* ("a human judge reviews
everything that isn't clearly irrelevant") — which biases it away from false
negatives. Its prompt is split into a cross-turn-stable shared prefix
(instructions, shots, identities — prefill-warmed and cached) and a live
tail (current turn, agent reasoning, staged tool events, the rule's glossary
terms, dependency context, and — for cross-referential rules — the policy
context). Rules judged relevant become turn matches; the ranker's per-rule
isolation is exactly why cross-referential rules need the policy-context
section.

#### c. Distillation

A distiller is an optional `TurnEvaluator` that judges a whole batch of rules
in one request, instead of fanning out a request per rule the way the ranker
does. The default composition binds none, so this is a seam rather than a
component you will find below.

An evaluator that answers the batch at once says so through
`evaluates_batch_in_one_request`, and that changes the cost arithmetic the
strategy matrix is built on: per-rule fan-out is what makes a verdict
expensive, so rules that would otherwise be left to embedding recall become
affordable to judge.

### v. Evaluation scheduling

#### a. Motivation

Some rules cannot be judged correctly until *other* judgments have landed:

- A **dependent** rule's verdict is moot until its dependency shows evidence
  of being in play — evaluating both simultaneously wastes the dependent's
  call or judges it against incomplete premises.
- A **cross-referential** rule's prompt needs the discovered set
  *including this turn's fresh admissions* — which only exist after discovery reports.

The scheduler exists to order such evaluations — and to do nothing at all
when no ordering is needed.

#### b. Constraints

- **Bounded**: at most `max_rounds` (default 2) rounds; the final round
  deliberately includes candidates whose state is still unknown, so cycles
  and deep chains degrade to "evaluate anyway" rather than silent drops.
- **No cost below the threshold**: phasing engages only when the candidate
  count reaches `EvaluationSchedulingPolicy.threshold` (default 8) — with one
  exception: **forcing preconditions**, which mark correctness requirements
  (not optimizations) and enable phasing regardless, logged when the forcing
  changed the outcome. Providers emitting them must self-gate so forcing is
  never pointless (the cross-reference provider skips small landscapes
  entirely, where the ranker uses the full inventory and waiting buys
  nothing).
- **Fail-open**: provider errors degrade to one-pass evaluation; skipped
  candidates are logged, never silently dropped.
- **Generic**: the scheduler knows nothing about rankers, distillers, or
  recall. It sees candidates, preconditions, and evidence. New ordering
  reasons are added as `EvaluationPreconditionProvider`s at the composition
  root — dependencies and cross-references are the two current providers.

#### c. Mechanism

The matcher counts the evidence sources expected per candidate and supplies
initial evidence (durably discovered rules, early deterministic matches).
`EvaluationScheduler.create_plan` gathers preconditions from all providers
concurrently and returns an `EvaluationPlan`. Each round: the plan selects
candidates whose preconditions are satisfied (the final round also takes
unknowns), then the matcher snapshots the targets' effective states for each
dependent model evaluation. A determinate prompt includes only satisfied
targets; if any target is still unknown, the matcher warns and includes all
configured targets as explicitly uncertain fallback context. This prompt
preparation does not promote a recall-only source to RANK or DISTILL. The
matcher then runs that round's evaluators and (round 1) discovery concurrently,
fresh admissions land in the pending-discovered state, and each source's
results are fed back via `plan.observe`. Positive evidence is sticky; negative
evidence is conclusive only once every expected source for a rule has reported.
The full state machine is documented at the top of
`matching/evaluation_scheduler.py`.

### vi. Rule resolution

After all rounds, the `RelationalResolver` (`compass/relational_resolver.py`)
has the authoritative last word over what the evaluators produced:

- **Entailment expansion** (`expand_entailment`): before anything is
  filtered, ENTAILMENT relationships activate additional rules — a matched
  (or admitted) rule pulls its entailment targets into the same set, with
  the same usage its strongest entailing source had (turn match begets turn
  match; admission begets admission). The closure is indirect and traverses
  group targets into their members. Expansion runs *before* the dependency
  gate, so an entailed rule is itself dependency-gated and can satisfy the
  dependencies of other evaluations in the same pass.
- **Dependency filtering** (`filter_unmet_dependencies`): an evaluation
  survives only if its dependencies are met — by another *match* this same
  turn (same-turn chains resolve in order) or by the session's
  **matched-rule ledger** (`compass.rules_ever_matched_in_session`): the set of rules
  that have turn-matched at least once this session. The ledger is a
  verdict-level latch — recorded *before* priority resolution (a
  deprioritized match still counts), never erased by working-set eviction.
  Discovery admission satisfies **nothing**: recall is deliberately
  generous, and letting its embedding-level false positives unlock
  dependents would make `depend_on()` decorative. A rule whose dependency
  was never met stays out *entirely* — including out of the discovered set —
  so it keeps being judged fresh each turn until its dependency arrives.
  Multiple dependencies are AND-ed (`UNMET_DEPENDENCY_ALL`); those declared
  through `depend_on_any` share a `group_id` and form an OR group, satisfied
  by any one member (`UNMET_DEPENDENCY_ANY`). A group target resolves the
  same way: `GROUP_ANY` (`p.AnyOf`) needs one relevant member, while a bare
  group needs *every* member relevant, and an empty group is unmet.
  Dependencies declared on a group are inherited by its members, since the
  source ids are the rule plus its groups.
  Note the deliberate two-layer design: the scheduler uses dependencies
  merely to *order* evaluations; the resolver *enforces* them.
  (Both steps above share one choke point in the matcher,
  `_resolve_admission_relationships`.)
- **Relational priority** (`resolve_priority_relationships`): among turn
  matches, a match is dropped when a *direct* PRIORITY relationship targets
  it (or one of its groups) and the source is another turn match (or a
  group with such a member). Priority does not propagate through inactive
  intermediaries (the reinstatement principle), but sources are judged
  against the full pre-filter set, so a dropped rule still suppresses its
  own targets and chains collapse in one pass. Deprioritization does *not*
  cascade into dependencies: a deprioritized rule stays admitted and keeps
  satisfying its dependents.
- **Numeric priority arbitration** (`keep_highest_priority`): among the
  relational survivors, only the top numeric priority remains turn-matched.
  Entailed matches are exempt — they owe their activation to their entailer
  and must not suppress it. Outranked rules are demoted to session
  admission (they were relevant — they just lost this turn). Mid-turn
  updates re-arbitrate, accepting the prompt-prefix break that a stronger
  mid-turn match causes.

Because a deprioritized rule remains in the discovered working set, its text
still appears in the responder's full instruction set. Turn-set removal
alone therefore doesn't suppress behavior; the responder renders this
turn's priority losers in an explicit "OUTPRIORITIZED THIS TURN — do not
act on these" live-tail section (`turn_deprioritized_rules` in response
state), keeping the cacheable system-wide instructions untouched. The same
note also covers *standing* instructions outranked by numeric priority:
when a turn match wins at priority > 0, every rendered rule with a lower
priority is listed — including recall-admitted rules that never received a
turn verdict, since numeric priority is declared data and needs none.

Tools participate in this via **transient rules**: a `ToolResult` may carry
`rules=[{action, condition?, priority?, weight?}]`, which the matcher's
post-step refresh injects as turn matches ("Returned by tool 'x'"). They
are per-turn only — never stored, admitted, or ledgered (the `transient`
evaluation-metadata flag filters them out of durable recording) — but they
compete in priority resolution like any other match and render in full in
the step reminders. A high-priority tool rule ("the account is locked —
say only X") therefore suppresses the standing instructions for exactly
one turn. Note for MEDIUM-effort agents: the per-step instruction cache is
invalidated when new tool events are staged, so the refresh that performs
this injection (and tool-gated reevaluation) is never skipped by the
caching optimization.

Only after resolution are results recorded: turn matches into response
state, admissions into the discovered set and session metadata — at which
point pending discoveries become durable and the cycle is ready for the next
turn.

## 5. Putting it together: cross-referential rules end-to-end

Cross-referential rules exercise nearly every mechanism above, so they make
a good closing walkthrough. A rule like
*"the customer asks about a capability that is **not supported**"*
cannot be judged in isolation —
"supported" is defined by the *other* rules — and embedding recall is
structurally weakest on exactly this shape (a complement set has no stable
embedding neighborhood). The build-time `RuleRelativityProposer` flags such
rules; matching then routes them by one question —
*does anything mechanical hang off the match?* — and, when ranked, by the
policy-landscape size:

- **Not flagged**: normal matching, per the strategy matrix.
- **Flagged, no attachments ("bare")**: statically included — always
  rendered in the domain instructions with its declared weight, never
  discovered or ranked. The responder judges it with the surrounding rules
  in the same prompt — free, and immune to recall's blind spot.
- **Flagged + attachments, small landscape** (inventory ≤
  `MAX_FULL_POLICY_CONTEXT_RULES`): escalated to RANK; evaluated in round 1
  with the **full inventory** as its policy context. No deferral — the
  complete landscape is already in the prompt, so waiting would buy nothing.
- **Flagged + attachments, large landscape**: escalated to RANK and
  **deferred to the final scheduling round** (forcing phasing if the
  candidate count wouldn't have enabled it, and clamping a single-round
  policy to two rounds), so its policy context — the discovered set, this
  turn's pending discovered rules (recall), AND pending matched rules from earlier
  evaluation rounds — is populated when the prompt renders. The round
  verdicts matter because recall's misses are exactly what ranking catches:
  a rule that only the ranker caught must still be visible to the deferred
  rule.

Two caveats. A **code evaluator** on an attached flagged rule bypasses the
LLM policy context entirely — code evaluators build no prompt, so resolving
the rule's references is the evaluator author's job. And the policy context
renders wherever the rule is evaluated.

The design invariants worth internalizing: deferral can never be pointless
(it only exists where the bounded context is used), and the bounded context
can never be stale (it is only used where deferral is guaranteed). Both fall
out of the ranker and the scheduler's cross-reference provider sharing one
boundary — `MAX_FULL_POLICY_CONTEXT_RULES` in `matching/common.py` — rather
than each keying off its own measure.
