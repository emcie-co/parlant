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

from dataclasses import dataclass, field
from functools import cached_property
from itertools import chain
from typing import Any, Optional, TypeAlias

from parlant.core.agents import Effort
from parlant.core.capabilities import Capability
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.rule_match import RuleMatch
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
from parlant.core.engines.engine_context import EngineContext as _EngineContext
from parlant.core.glossary import Term
from parlant.core.rules import Rule, RuleId
from parlant.core.journeys import Journey, JourneyId
from parlant.core.common import Weight
from parlant.core.tools import Tool, ToolId


@dataclass(frozen=True)
class IterationState:
    """State of a single iteration in the response process"""

    matched_rules: list[RuleMatch]
    ruled_out: list[RuleMatch]
    resolved_rules: list[RuleMatch]
    tool_insights: ToolInsights
    executed_tools: list[ToolId]


@dataclass
class ResponseState:
    agent_effort: Effort = Effort.MEDIUM
    ordinary_rule_matches: list[RuleMatch] = field(default_factory=list)
    tool_enabled_rule_matches: dict[RuleMatch, list[ToolId]] = field(default_factory=dict)
    # tools the matched rules enabled this turn (described in the prompt)
    matched_tools: list[Tool] = field(default_factory=list)
    # per-turn relevance scores for candidate tools, computed before final selection
    tool_relevance_scores: dict[ToolId, float] = field(default_factory=dict)
    # final tool catalog offered to the model, capped and emitted by name for cache stability
    available_tools: list[Tool] = field(default_factory=list)
    tool_ids_by_name: dict[str, ToolId] = field(default_factory=dict)  # to run a tool by its name
    # initial tool catalog selected during fill(); update() logs only deltas from this baseline
    fill_available_tool_ids: set[ToolId] = field(default_factory=set)

    # The agent's reasoning from each step of the response loop so far this turn,
    # in order. The loop appends to it after every step; the matching components
    # (turn evaluators) feed it into their per-rule prompts so each step's
    # evaluation is aware of what the agent has already concluded. Empty on the
    # initial match (no steps have run yet).
    reasoning_steps: list[str] = field(default_factory=list)

    # Durable summary of session events before the latest compaction marker.
    # Empty means no compaction summary is active for this loaded interaction.
    session_summary: str = ""

    # Reviewer-provided replacement reasoning when pending tool calls would breach
    # policy. Empty means no breach was found, or the reviewer has not run yet.
    step_notes: str = ""

    # Reviewer-provided summary of what the agent still needs to do before
    # responding to the user. Empty means the reviewer has not run yet.
    todo: str = ""

    # Set when the loop has exhausted its attempts and is forcing a final message.
    # While true, the step prompt is stripped of the tool catalog, matched rules,
    # and reviewer TODO so the model stops attempting the task and simply explains to
    # the user why it couldn't help. Reset once the give-up step completes.
    giving_up: bool = False

    # Per-turn signals the matcher precomputes (once) so its per-rule strategy
    # selection can stay synchronous: rules that carry tools, and rules
    # that participate in a dependency relationship.
    rule_ids_with_tools: set[RuleId] = field(default_factory=set)
    rule_ids_with_dependencies: set[RuleId] = field(default_factory=set)

    # Tools attached to a rule's action (e.g. a journey condensed into a
    # rule carries the tools of its tool-using steps). Read by turn evaluators
    # to surface what each tool does; empty for rules without attached tools.
    tools_by_rule: dict[RuleId, set[tuple[ToolId, Tool]]] = field(default_factory=dict)

    # Glossary terms each evaluated rule depends on to be interpreted
    # correctly (top-k by the rule's own query; see Matcher._load_rule_terms).
    # Rendered with the rule in the per-rule prompt tails.
    terms_by_rule: dict[RuleId, list[Term]] = field(default_factory=dict)

    # TODO: Remove what isn't needed
    context_variables: list[tuple[ContextVariable, ContextVariableValue]] = field(
        default_factory=list
    )
    glossary_terms: set[Term] = field(default_factory=set)
    capabilities: list[Capability] = field(default_factory=list)
    journeys: list[Journey] = field(default_factory=list)
    journey_paths: dict[JourneyId, list[Optional[str]]] = field(default_factory=dict)
    tool_events: list[EmittedEvent] = field(default_factory=list)
    tool_insights: ToolInsights = field(default_factory=ToolInsights)
    prepared_to_respond: bool = False
    message_events: list[EmittedEvent] = field(default_factory=list)
    usable_rules: list[Rule] = field(default_factory=list)
    session_rules: set[Rule] = field(default_factory=set)
    # Eviction ledger for session rules: {id: last event offset at eviction}.
    # A ledgered rule may only be readmitted by conversation that arrived
    # after its eviction (see Matcher.prune_session_rules).
    evicted_session_rules: dict[RuleId, int] = field(default_factory=dict)
    additional_canned_response_fields: dict[str, Any] = field(default_factory=dict)
    iterations: list[IterationState] = field(default_factory=list)

    @property
    def ordinary_rules(self) -> list[Rule]:
        return [gp.rule for gp in self.ordinary_rule_matches]

    @property
    def tool_enabled_rules(self) -> list[Rule]:
        return [gp.rule for gp in self.tool_enabled_rule_matches.keys()]

    @property
    def rules(self) -> list[Rule]:
        return self.ordinary_rules + self.tool_enabled_rules

    @cached_property
    def dynamic_effort_level(self) -> Effort:
        """Resolve effective effort from the agent default and matched rule effort levels."""
        efforts = [
            self.agent_effort,
            *(
                match.rule.effort_lift
                for match in chain(
                    self.ordinary_rule_matches,
                    self.tool_enabled_rule_matches.keys(),
                )
                if match.rule.effort_lift is not None
            ),
        ]

        return max(efforts)

    @cached_property
    def has_matched_high_criticality_rules(self) -> bool:
        return any(
            match.rule.weight == Weight.HIGH
            for match in chain(
                self.ordinary_rule_matches,
                self.tool_enabled_rule_matches.keys(),
            )
        )

    def invalidate_cached_properties(self) -> None:
        self.__dict__.pop("dynamic_effort_level", None)
        self.__dict__.pop("has_matched_high_criticality_rules", None)


# The compass engine sees its own ResponseState typed through context.state.
EngineContext: TypeAlias = _EngineContext[ResponseState]
