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
which evaluate one rule per prompt and fan out concurrently."""

from typing import Sequence

from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.glossary import Term
from parlant.core.nlp.common import UsageInfo
from parlant.core.nlp.generation_info import GenerationInfo


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
        name="agent-reasoning-so-far",
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
        name="rule-terms",
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
            extra={
                "reasoning_tokens": sum(
                    int(info.usage.extra.get("reasoning_tokens", 0)) for info in infos
                ),
            },
        ),
    )
