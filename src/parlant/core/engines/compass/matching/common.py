# Copyright 2026 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers shared by the per-rule matching components (turn evaluators),
which evaluate one rule per prompt and fan out concurrently.

Terminology used here and in the modules that import from here:

- **Inventory** (or "usable rules"): every enabled rule the agent could use
  this turn (``state.usable_rules``) — as opposed to the *discovered* working
  set, which is the subset admitted to the session so far.
- **Cross-referential rule**: a rule whose meaning depends on the agent's
  OTHER rules, betrayed by dangling references in its own text — implicit
  complements ("a capability that is not supported"), quantified references
  ("any of our policies"), or definite references ("unless the return policy
  applies"). Classified at build time by the RuleRelativityProposer and
  carried in ``rule.metadata["references_other_policies"]`` (author-set
  metadata overrides the proposition). Such a rule cannot be judged in
  isolation: a blind verdict is wrong on exactly the dimension that defines
  the rule.
- **Policy context**: the set of other rules rendered into a
  cross-referential rule's evaluation prompt so its references can actually
  be resolved.
- **Policy landscape**: the size regime of the inventory, split by
  ``MAX_FULL_POLICY_CONTEXT_RULES``. In a *small* landscape the entire
  inventory fits in a prompt, so the policy context is complete and no
  evaluation ordering is needed. In a *large* landscape the policy context is
  the bounded discovered set plus this turn's provisional evidence (pending
  recall admissions and positive round verdicts), which is only accurate
  AFTER discovery has run — so cross-referential evaluation is
  deferred to the scheduler's final round, forcing phased evaluation if
  needed (see CrossReferenceEvaluationPreconditionProvider).
- **Attachment**: anything mechanical that fires only when a rule officially
  matches a turn — tools, dependencies, labels, triggered effort, a
  non-default priority, message/tool review requests, match/discovery
  handlers, or a code evaluator. A cross-referential rule WITH attachments must be ranked (so
  its attachments can trigger); one with NO attachments ("bare") is instead
  statically included in the always-rendered instructions, where the
  responder judges it with the surrounding rules in the same prompt — free,
  and immune to embedding recall's structural weakness on complement-type
  conditions. (Known v1 limitation: canned-response associations are not
  counted as attachments by the automatic check; set the metadata flag
  explicitly on such rules if they must be ranked.)
"""

from enum import Enum
from typing import Callable, Sequence, cast

from parlant.core.engines.compass.prompt_builder import PromptBuilder, SectionStatus
from parlant.core.engines.compass.context import EngineContext
from parlant.core.glossary import Term
from parlant.core.rules import Rule
from parlant.core.sessions import (
    Event,
    EventKind,
    EventSource,
    MessageEventData,
    delivered_message_text,
)
from parlant.core.nlp.common import (
    REPORTED_COST_EXTRA_KEY,
    UsageInfo,
    add_usage_extra_value,
    format_reported_cost,
)
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core import metadata


def render_message_query(event: Event) -> str:
    """Render one session MESSAGE event as a recall-query line.

    This is the ONE canonical form every recaller (rules, tools, glossary,
    canned responses) uses to turn a conversation message into a similarity
    query, so the same turn produces the same string and is embedded once
    (shared across all same-model recall routes via the embedder cache).

    The text is ``delivered_message_text`` — the censor- and voice-aware view
    of what was actually said — tagged with the speaker's source. Using the
    delivered text everywhere means a moderation-flagged message never drives
    any recall query (previously only canned responses censored it).
    """
    data = cast(MessageEventData, event.data)
    return f"{event.source.value}: {delivered_message_text(data, event.metadata)}"


def message_events(context: EngineContext) -> list[Event]:
    """Every MESSAGE event in the interaction, in order (no status/tool/custom)."""
    return [e for e in context.interaction.events if e.kind == EventKind.MESSAGE]


def customer_message_events(context: EngineContext) -> list[Event]:
    """The interaction's customer MESSAGE events, in order."""
    return [e for e in message_events(context) if e.source == EventSource.CUSTOMER]


def build_cumulative_query(context: EngineContext) -> str:
    """The whole conversation as one query: an optional session summary followed
    by every message rendered via :func:`render_message_query`. Empty when there
    is nothing to say."""
    events = message_events(context)
    if not events and not context.state.session_summary:
        return ""

    lines: list[str] = []
    if context.state.session_summary:
        lines.append(f"Session summary: {context.state.session_summary}")
    lines.extend(render_message_query(e) for e in events)
    return "\n".join(lines)


def latest_customer_query(context: EngineContext) -> str:
    """The most recent customer message as a query line, or "" if there is none."""
    event = context.interaction.last_customer_message_event
    return render_message_query(event) if event is not None else ""


class PromptSections(str, Enum):
    """Section names for the prompt fragments shared across turn evaluators.

    Every module that names prompt sections exposes a ``PromptSections`` enum
    beside the ``add_section`` calls it names; the registry test in
    tests/core/stable/engines/compass/test_prompt_sections.py enforces that
    no compass prompt contains an unregistered name.
    """

    AGENT_REASONING = "shared.agent-reasoning"
    RULE_TERMS = "shared.rule-terms"


def add_agent_reasoning(builder: PromptBuilder, reasoning_steps: Sequence[str]) -> None:
    """Append the agent's reasoning from earlier steps of the current turn, so the
    per-rule evaluation is aware of what the agent has already concluded (e.g.
    that it needs to run a tool, or which facts it has established).

    Added to the per-call tail by callers - NOT the shared prompt - so it stays out
    of the cached prefix: the reasoning grows with every step, whereas the cached
    prefix must stay byte-stable across the prefill/load pair and across steps."""
    if not reasoning_steps:
        return

    reasoning_text = "\n\n".join(
        f"Step {i}: {step.strip()}" for i, step in enumerate(reasoning_steps, start=1)
    )
    builder.add_section(
        name=PromptSections.AGENT_REASONING,
        template="""
AGENT'S REASONING SO FAR THIS TURN
-----------------
While preparing the current response, you (the agent) have already reasoned through the steps below, in order. Take this into account when evaluating the rule - it reflects what you have concluded and what you intend to do next:
{reasoning_text}
""",
        props={"reasoning_text": reasoning_text},
    )


def add_rule_terms(builder: PromptBuilder, terms: Sequence[Term]) -> None:
    """Render the glossary terms a rule depends on for correct interpretation,
    next to the rule itself.

    Added to the per-call tail by callers - NOT the shared prompt - both because it
    varies per rule (the cached prefix must stay byte-stable across the
    fan-out) and because callers skip terms already present in the shared-prefix
    glossary section."""
    if not terms:
        return

    terms_text = "\n\n".join(
        f"### {term.name}\n"
        + (f"Synonyms: {', '.join(term.synonyms)}\n" if term.synonyms else "")
        + term.description
        for term in sorted(terms, key=lambda t: (t.name, t.id))
    )
    builder.add_section(
        name=PromptSections.RULE_TERMS,
        template="""
TERMS USED BY THIS RULE
-----------------
The rule below may rely on the following domain terms. Interpret it according to these definitions, prioritizing them over any general meaning you may know:
{terms_text}
""",
        props={"terms_text": terms_text},
    )


def aggregate_generation_info(
    infos: Sequence[GenerationInfo],
    total_duration: float | None = None,
) -> GenerationInfo:
    """Aggregate usage across the per-rule requests of a fan-out: tokens are
    summed, duration is the max (the requests run concurrently, so it reflects
    wall-clock, not total work), and token breakdowns in ``extra`` (possibly
    absent) are summed with a 0 default."""
    return GenerationInfo(
        schema_name=infos[0].schema_name,
        model=infos[0].model,
        duration=total_duration
        if total_duration is not None
        else max(info.duration for info in infos),
        usage=UsageInfo(
            input_tokens=sum(info.usage.input_tokens for info in infos),
            output_tokens=sum(info.usage.output_tokens for info in infos),
            cached_input_tokens=sum(info.usage.cached_input_tokens for info in infos),
            # Sum every numeric meter key-wise — billable meters (cache writes,
            # storage, reported cost) must survive aggregation, or their costs
            # silently vanish between the adapter and the ledger.
            extra=_sum_extra_meters(infos),
            requests=sum(info.usage.requests for info in infos),
        ),
    )


def _sum_extra_meters(infos: Sequence[GenerationInfo]) -> dict[str, int | float | str]:
    merged: dict[str, int | float | str] = {}
    for info in infos:
        for key, value in info.usage.extra.items():
            if key == REPORTED_COST_EXTRA_KEY:
                if (formatted_cost := format_reported_cost(value)) is None:
                    continue
                meter_value: int | float | str = formatted_cost
            elif isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            else:
                meter_value = value
            existing = merged.get(key)
            merged[key] = add_usage_extra_value(key, existing, meter_value)
    return merged


# The policy-landscape boundary: at or below this many usable rules, a
# cross-referential rule's evaluation prompt carries the FULL inventory as its
# policy context (complete landscape, no ordering dependency); above it, the
# bounded discovered-set context is used and evaluation is deferred until
# discovery has reported. Shared by the rule ranker and the evaluation
# scheduler's cross-reference provider — a single boundary, so the two can
# never disagree about which regime applies. Aligned with the
# discovered-rules pruning target (see Matcher._DISCOVERED_RULES_TARGET).
MAX_FULL_POLICY_CONTEXT_RULES = 20


def is_small_policy_landscape(inventory_size: int) -> bool:
    """Whether the whole inventory fits in a cross-referential rule's prompt."""
    return inventory_size <= MAX_FULL_POLICY_CONTEXT_RULES


def add_dependency_context(
    builder: PromptBuilder,
    context: EngineContext,
    rule: Rule,
    format_rule: Callable[[Rule], str],
    *,
    section_name: str,
) -> None:
    """Render dependency targets prepared for this rule's evaluation prompt.

    The matcher normally narrows ``ResponseState.dependency_targets_by_rule``
    to targets with effectively satisfied evidence. When any target remains
    indeterminate, it preserves best-effort evaluation by preparing every
    configured target and marking the section as fallback context. Direct
    prompt builds without a scheduler snapshot use the same all-target fallback.

    A dependent rule only applies where its targets apply, and its condition
    is often written to be read inside their content — step names, defined
    scopes, "the procedure" — so the verdict needs the targets in view to
    interpret it. Per-rule and turn-varying, so callers must add it to the
    live tail, never the cached prefix. Renders nothing for rules without
    dependency targets.
    """
    configured_targets = context.state.dependency_targets_by_rule.get(rule.id, [])
    has_prepared_snapshot = rule.id in context.state.dependency_prompt_targets_by_rule
    prompt_targets = (
        context.state.dependency_prompt_targets_by_rule[rule.id]
        if has_prepared_snapshot
        else configured_targets
    )
    dependency_context = sorted(
        (target for target in prompt_targets if target.id != rule.id),
        key=lambda target: str(target.id),
    )

    if not dependency_context:
        return

    if (
        not has_prepared_snapshot
        or rule.id in context.state.indeterminate_dependency_context_rule_ids
    ):
        template = """
# CONTEXT THIS RULE MAY OPERATE WITHIN

One or more dependency targets are still indeterminate, so all configured targets are included below as fallback context. Do not assume that every target applies. Use the conversation to determine which context is actually in play, while interpreting the rule under evaluation against any applicable procedures, steps, definitions, and scope.

{dependency_context_text}
"""
    else:
        template = """
# CONTEXT THIS RULE OPERATES WITHIN

The following dependency target(s) are satisfied for this evaluation. The rule under evaluation can only apply when its dependency requirements are met, and its condition is meant to be interpreted inside this active context. Judge its relevance against the content below — its procedures, steps, definitions, and scope — not in isolation. For example, a condition like "the procedure has reached the payment step" refers to a step defined in the depended-on content below.

{dependency_context_text}
"""

    builder.add_section(
        name=section_name,
        template=template,
        props={
            "dependency_context_text": "\n".join(
                f"- ###\n{format_rule(target)}\n###" for target in dependency_context
            ),
        },
        status=SectionStatus.ACTIVE,
    )


def add_policy_context(
    builder: PromptBuilder,
    context: EngineContext,
    rule: Rule,
    format_rule: Callable[[Rule], str],
    *,
    section_name: str,
) -> None:
    """Render a cross-referential rule's policy context (see the module
    docstring) into its evaluation prompt, whichever evaluator builds it.

    Small landscape: the full inventory — complete by construction, no
    dependency on evaluation ordering. Large landscape: the discovered
    working set plus this turn's provisional evidence — pending recall
    admissions AND positive verdicts from earlier evaluation rounds (recall's
    misses are exactly what ranking catches, and a deferred rule exists to
    absorb that evidence).

    Turn-varying, so callers must add it to the live tail, never the cached
    prefix. Renders nothing for non-flagged rules or an empty context: an
    empty pool in a large landscape only means recall surfaced nothing — an
    explicit "no other policies apply" claim would launder that blindness
    into the verdict.
    """
    if not (
        rule.metadata.get("references_other_policies")
        or metadata.internal_value(rule.metadata, ("references_other_policies",), False)
    ):
        return

    if is_small_policy_landscape(len(context.state.usable_rules)):
        pool: set[Rule] = set(context.state.usable_rules)
    else:
        pool = {
            *context.state.discovered_rules,
            *context.state.matcher.pending_discovered_rules,
            *context.state.matcher.pending_matched_rules,
        }

    policy_context = sorted(
        (other for other in pool if other.id != rule.id),
        key=lambda other: str(other.id),
    )

    if not policy_context:
        return

    builder.add_section(
        name=section_name,
        template="""
# THE AGENT'S OTHER ACTIVE POLICIES

The rule under evaluation refers, implicitly or explicitly, to the agent's other policies. Judge its relevance IN LIGHT of the following policies currently in play. For example: a condition about requests "not supported" or "not covered elsewhere" holds only if none of these policies covers the request, and a condition deferring to another or a more specific policy holds only per what these policies actually say.

{policy_context_text}
""",
        props={
            "policy_context_text": "\n".join(
                f"- ###\n{format_rule(other)}\n###" for other in policy_context
            ),
        },
        status=SectionStatus.ACTIVE,
    )
