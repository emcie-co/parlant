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

# mypy: ignore-errors

"""Matcher behavior tests, in the classical style.

Almost every test here builds a REAL ``Matcher`` through the shared
:mod:`harness` (real stores, real recaller, real scheduler, real resolver)
and scripts only the model boundaries. Assertions target observable
outcomes: response state, session persistence, and the scripted boundaries'
recorded traffic. See ``harness.py`` for the boundary contract.

The one deliberate exception is the "orchestration and logging seams"
section: tests of fill's concurrency shape and of log rendering, where the
observable IS the seam (a log line, a delegation) and no model boundary is
involved. Those keep small hand-rolled fakes.
"""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import asyncio
from typing import Any, cast
import pytest

from parlant.core.container import Container
from parlant.core.loggers import StdoutLogger
from parlant.core.tracer import LocalTracer

from parlant.core.agents import Effort
from parlant.core.common import IdGenerator, Weight
from parlant.core.engines.rule_evaluation import RuleEvaluation
from parlant.core.engines.compass.prompt_builder import PromptBuilder
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.engines.compass.matching.rule_evaluation import (
    RuleEvaluationResult,
    TurnEvaluator,
)
from parlant.core.engines.compass.matching.tool_selection import ToolSelector
from parlant.core.engines.compass.matcher import (
    Matcher,
    _ContextUsage,
    _EVICTED_DISCOVERED_RULES_PATH,
    _LAST_DISCOVERED_RULE_PRUNING_OFFSET_PATH,
    _DISCOVERED_RULE_IDS_PATH,
    _RULES_EVER_MATCHED_IN_SESSION_PATH,
)
from parlant.core import metadata as metadata_utils
from parlant.core.engines.compass.context import (
    EngineContext,
    ResponseState,
)
from parlant.core.groups import GroupId, GroupStore
from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.core.relationships import (
    RelationshipDocumentStore,
    RelationshipEntity,
    RelationshipEntityKind,
    RelationshipKind,
)
from parlant.core.rules import Rule
from parlant.core.sessions import EventSource
from parlant.core.tools import Tool, ToolId, ToolOverlap

from tests.core.common.utils import create_event_message
from tests.core.stable.engines.compass.matching.harness import (
    MatchingHarness,
    make_rule,
    matching_harness,
)
from tests.core.stable.engines.compass.matching.utils import (
    create_engine_context,
    create_rule,
    create_term,
)


def _relevant(rule: Rule) -> RuleEvaluation:
    return RuleEvaluation(rule=rule, reasoning="applies", is_relevant=True)


def _match_usage(rule: Rule) -> tuple[RuleEvaluation, _ContextUsage]:
    return (
        RuleEvaluation(rule=rule, reasoning="matched"),
        _ContextUsage.MATCH_CURRENT_TURN,
    )


# --- Prompt rendering (pure, no components involved) -----------------------------


def test_that_content_only_highlighted_matches_are_rendered_as_instruction_reminders() -> None:
    rule = replace(
        create_rule(
            condition="booking a flight",
            action=None,
            content="Collect booking details in order.",
        ),
        weight=Weight.HIGH,
        title="Book flight",
    )
    match = RuleEvaluation(
        rule=rule,
        reasoning="Relevant.",
        metadata={"highlights": ["Ask the user for the trip type."]},
    )

    prompt = PromptBuilder().add_matched_rules([match], {}, {rule.id: rule}).build()

    assert '### Review the instructions under "Book flight"\n' in prompt
    assert "Highlights:\n- Ask the user for the trip type." in prompt
    assert "IMPORTANT: Please go back and reason" in prompt


def test_that_glossary_terms_are_rendered_in_deterministic_order() -> None:
    # Callers pass set-derived lists; rendering must not depend on iteration
    # order, or the cached prefixes it appears in churn across restarts.
    terms = [create_term(f"term {i}", f"description {i}") for i in range(5)]

    forward = PromptBuilder().add_glossary(terms).build()
    backward = PromptBuilder().add_glossary(terms[::-1]).build()

    assert forward == backward


# --- Orchestration and logging seams ---------------------------------------------
#
# Deliberately non-classical: these test fill's concurrency shape and log
# rendering. The observable IS the seam — a log line, a delegation, an event
# ordering — and no model boundary is involved, so a scripted boundary can't
# observe it. Everything else in this file goes through the real harness.


class _FakeLogger:
    def __init__(self) -> None:
        self.debug_messages: list[str] = []
        self.trace_messages: list[str] = []
        self.warning_messages: list[str] = []

    def debug(self, *args, **kwargs):
        self.debug_messages.append(str(args[0]))

    def trace(self, *args, **kwargs):
        self.trace_messages.append(str(args[0]))

    def warning(self, *args, **kwargs):
        self.warning_messages.append(str(args[0]))


class _RecordingToolSelector(ToolSelector):
    def __init__(self) -> None:
        self.prepare_started = asyncio.Event()
        self.prepare_calls: list[EngineContext] = []
        self.select_calls: list[EngineContext] = []

    async def prepare(self, context: EngineContext) -> None:
        self.prepare_calls.append(context)
        self.prepare_started.set()

    async def select(self, context: EngineContext) -> None:
        self.select_calls.append(context)


class _RecordingGlossaryRecaller:
    def __init__(self) -> None:
        self.prune_calls: list[EngineContext] = []

    async def prune(self, context: EngineContext) -> None:
        self.prune_calls.append(context)


class _FillSeamMatcher(Matcher):
    """Overrides fill's collaborator methods to observe its concurrency shape.

    Deliberately skips ``Matcher.__init__``: ``fill`` touches only the
    overridden methods and the tool selector.
    """

    def __init__(self, tool_selector: ToolSelector) -> None:
        self._tool_selector = tool_selector
        # fill() wraps its body in the component logging scope, so the seam
        # needs a real logger despite skipping Matcher.__init__.
        self._logger = StdoutLogger(LocalTracer())
        self.match_started = asyncio.Event()
        self.release_match = asyncio.Event()
        self.load_glossary_calls: list[EngineContext] = []
        self.select_tools_calls: list[tuple[EngineContext, bool]] = []

    async def _match(self, context: EngineContext) -> None:
        self.match_started.set()
        await self.release_match.wait()

    async def _load_glossary(self, context: EngineContext) -> None:
        self.load_glossary_calls.append(context)

    async def _select_tools(
        self,
        context: EngineContext,
        *,
        log_delta_from_fill: bool,
    ) -> None:
        self.select_tools_calls.append((context, log_delta_from_fill))


def _seam_context(*rules: Rule, effort: Effort = Effort.MEDIUM) -> EngineContext:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hello")])
    context.state = ResponseState(
        agent_effort=effort,
        usable_rules=list(rules),
        glossary_terms={create_term("known term", "already loaded")},
    )
    return context


def _tool(name: str, description: str | None = None) -> Tool:
    return Tool(
        name=name,
        creation_utc=datetime.now(timezone.utc),
        description=description or f"{name} tool",
        metadata={},
        parameters={},
        required=[],
        consequential=False,
        overlap=ToolOverlap.NONE,
    )


@pytest.mark.asyncio
async def test_that_tool_selector_prepare_runs_in_parallel_with_rule_matching() -> None:
    tool_selector = _RecordingToolSelector()
    matcher = _FillSeamMatcher(tool_selector)
    context = _seam_context()

    fill_task = asyncio.create_task(matcher.fill(context))
    await matcher.match_started.wait()
    await asyncio.wait_for(tool_selector.prepare_started.wait(), timeout=1.0)

    matcher.release_match.set()
    await fill_task

    assert tool_selector.prepare_calls == [context]
    assert matcher.select_tools_calls == [(context, False)]


@pytest.mark.asyncio
async def test_that_select_tools_delegates_to_tool_selector() -> None:
    matcher = object.__new__(Matcher)
    matcher._logger = _FakeLogger()
    tool_selector = _RecordingToolSelector()
    matcher._tool_selector = tool_selector
    context = _seam_context()

    await matcher._select_tools(context, log_delta_from_fill=False)

    assert tool_selector.select_calls == [context]


@pytest.mark.asyncio
async def test_that_select_tools_logs_no_delta_when_update_keeps_fill_catalog() -> None:
    logger = _FakeLogger()
    matcher = object.__new__(Matcher)
    matcher._logger = logger
    matcher._tool_selector = _RecordingToolSelector()
    context = _seam_context()
    tool_id = ToolId("local", "lookup")
    context.state.available_tools = [_tool("lookup")]
    context.state.tool_ids_by_name = {"lookup": tool_id}
    context.state.fill_available_tool_ids = {tool_id}

    await matcher._select_tools(context, log_delta_from_fill=True)

    assert logger.debug_messages == []


@pytest.mark.asyncio
async def test_that_select_tools_logs_only_delta_when_update_changes_fill_catalog() -> None:
    logger = _FakeLogger()
    matcher = object.__new__(Matcher)
    matcher._logger = logger
    matcher._tool_selector = _RecordingToolSelector()
    context = _seam_context()
    old_tool_id = ToolId("local", "old_lookup")
    new_tool_id = ToolId("local", "new_lookup")
    context.state.available_tools = [_tool("new_lookup", "New lookup tool.")]
    context.state.tool_ids_by_name = {"new_lookup": new_tool_id}
    context.state.tool_relevance_scores = {new_tool_id: 0.8}
    context.state.fill_available_tool_ids = {old_tool_id}

    await matcher._select_tools(context, log_delta_from_fill=True)

    assert len(logger.debug_messages) == 1
    assert "Tool recall:" in logger.debug_messages[0]
    assert "Added:" in logger.debug_messages[0]
    assert "new_lookup [Score: 0.80]" in logger.debug_messages[0]
    assert "Removed:" in logger.debug_messages[0]
    assert "old_lookup" in logger.debug_messages[0]


def test_that_available_tool_log_is_sorted_by_bucket_then_score() -> None:
    logger = _FakeLogger()
    matcher = object.__new__(Matcher)
    matcher._logger = logger
    turn_rule = create_rule("turn tool applies")
    session_rule = create_rule("session tool applies")
    turn_low_id = ToolId("local", "turn_low")
    turn_high_id = ToolId("local", "turn_high")
    session_id = ToolId("local", "session_tool")
    complementary_id = ToolId("local", "complementary_tool")
    context = _seam_context(turn_rule, session_rule)
    context.state.available_tools = [
        _tool("complementary_tool"),
        _tool("session_tool"),
        _tool("turn_low"),
        _tool("turn_high"),
    ]
    context.state.matched_tools = [_tool("turn_low"), _tool("turn_high")]
    context.state.discovered_rules = {session_rule}
    context.state.tools_by_rule = {session_rule.id: {(session_id, _tool("session_tool"))}}
    context.state.tool_ids_by_name = {
        "turn_low": turn_low_id,
        "turn_high": turn_high_id,
        "session_tool": session_id,
        "complementary_tool": complementary_id,
    }
    context.state.tool_relevance_scores = {
        turn_low_id: 0.2,
        turn_high_id: 0.9,
        session_id: 0.8,
        complementary_id: 1.0,
    }

    matcher._log_available_tools(context)

    log = logger.trace_messages[0]
    assert log.index("### 1 turn_high") < log.index("### 2 turn_low")
    assert log.index("### 2 turn_low") < log.index("### 3 session_tool")
    assert log.index("### 3 session_tool") < log.index("### 4 complementary_tool")


@pytest.mark.asyncio
async def test_that_the_matcher_delegates_glossary_pruning_to_the_recaller() -> None:
    matcher = object.__new__(Matcher)
    glossary_recaller = _RecordingGlossaryRecaller()
    matcher._glossary_recaller = glossary_recaller
    context = _seam_context()

    await matcher.prune_session_glossary(context)

    assert glossary_recaller.prune_calls == [context]


# --- Interaction queries ----------------------------------------------------------


# --- Discovered rule persistence (real session store) -----------------------------


async def test_that_discovered_rules_are_loaded_from_session_metadata(
    container: Container,
) -> None:
    rule_1 = make_rule("HELP")
    rule_2 = make_rule("REFUND")
    harness = await matching_harness(
        container,
        [rule_1, rule_2],
        rank_relevant=set(),
        session_metadata={
            _DISCOVERED_RULE_IDS_PATH: [
                str(rule_2.id),
                "missing-rule",
                str(rule_2.id),
                str(rule_1.id),
                17,
            ]
        },
    )

    await harness.matcher._load_discovered_rules(harness.context)

    assert harness.context.state.discovered_rules == {rule_2, rule_1}


async def test_that_matched_rules_are_stored_in_session_metadata(
    container: Container,
) -> None:
    rule_1 = make_rule("HELP")
    rule_2 = make_rule("REFUND")
    rule_3 = make_rule("BILLING")
    harness = await matching_harness(
        container,
        [rule_1, rule_2, rule_3],
        rank_relevant=set(),
        discovered_rules={rule_1},
    )
    harness.context.state.ordinary_rule_matches = [
        RuleEvaluation(rule=rule_2, reasoning="relevant"),
        RuleEvaluation(rule=rule_1, reasoning="still relevant"),
    ]
    matches = [
        (RuleEvaluation(rule=rule_2, reasoning="relevant"), _ContextUsage.MATCH_CURRENT_TURN),
        (RuleEvaluation(rule=rule_1, reasoning="still relevant"), _ContextUsage.MATCH_CURRENT_TURN),
    ]

    await harness.matcher._store_discovered_rules(harness.context, matches)

    assert harness.context.state.discovered_rules == {rule_1, rule_2}
    assert await harness.staged_discovered_rule_ids() == {str(rule_1.id), str(rule_2.id)}
    assert set(
        metadata_utils.internal_value(
            harness.context.session.metadata, _DISCOVERED_RULE_IDS_PATH, []
        )
    ) == {
        str(rule_1.id),
        str(rule_2.id),
    }
    assert harness.context.state.discovered_rule_evaluations == [matches[0][0]]


async def test_that_storing_discovered_rules_preserves_existing_rules(
    container: Container,
) -> None:
    rule_1 = make_rule("HELP")
    rule_2 = make_rule("REFUND")
    harness = await matching_harness(
        container,
        [rule_1, rule_2],
        rank_relevant=set(),
        discovered_rules={rule_1},
        session_metadata={_DISCOVERED_RULE_IDS_PATH: [str(rule_1.id)]},
    )
    matches = [
        (
            RuleEvaluation(rule=rule_2, reasoning="newly relevant"),
            _ContextUsage.ADMIT_TO_DISCOVERED,
        )
    ]

    await harness.matcher._store_discovered_rules(harness.context, matches)

    assert harness.context.state.discovered_rules == {rule_1, rule_2}
    assert await harness.staged_discovered_rule_ids() == {str(rule_1.id), str(rule_2.id)}
    assert harness.context.state.discovered_rule_evaluations == [matches[0][0]]


async def test_that_storing_no_new_discovered_rules_does_not_clear_existing_rules(
    container: Container,
) -> None:
    rule = make_rule("HELP")
    harness = await matching_harness(
        container,
        [rule],
        rank_relevant=set(),
        discovered_rules={rule},
        session_metadata={_DISCOVERED_RULE_IDS_PATH: [str(rule.id)]},
    )

    await harness.matcher._store_discovered_rules(harness.context, [])

    assert harness.context.state.discovered_rules == {rule}
    assert harness.context.state.discovered_rule_evaluations == []
    assert await harness.staged_discovered_rule_ids() == {str(rule.id)}


async def test_that_discovered_rules_are_not_stored_when_metadata_is_unchanged(
    container: Container,
) -> None:
    # The contract is "no needless staged mutation"; a clean workspace makes
    # the engine skip the final metadata write.
    rule = make_rule("HELP")
    harness = await matching_harness(
        container,
        [rule],
        rank_relevant=set(),
        discovered_rules={rule},
        session_metadata={_DISCOVERED_RULE_IDS_PATH: [str(rule.id)]},
    )
    harness.context.state.ordinary_rule_matches = [RuleEvaluation(rule=rule, reasoning="")]

    await harness.matcher._store_discovered_rules(harness.context, [_match_usage(rule)])

    assert not harness.context.session_metadata.is_dirty


async def test_that_session_only_matches_do_not_enter_turn_matched_rules(
    container: Container,
) -> None:
    recalled_rule = make_rule("HELP")
    turn_rule = make_rule("REFUND")
    harness = await matching_harness(
        container,
        [recalled_rule, turn_rule],
        rank_relevant=set(),
    )
    matches = [
        (
            RuleEvaluation(rule=recalled_rule, reasoning="recalled"),
            _ContextUsage.ADMIT_TO_DISCOVERED,
        ),
        (
            RuleEvaluation(rule=turn_rule, reasoning="matched"),
            _ContextUsage.MATCH_CURRENT_TURN,
        ),
    ]

    await harness.matcher._record(harness.context, matches, append=False)

    assert harness.context.state.ordinary_rule_matches == [
        RuleEvaluation(rule=turn_rule, reasoning="matched")
    ]
    assert harness.context.state.tool_enabled_rule_matches == {}
    assert harness.context.state.discovered_rules == {recalled_rule, turn_rule}
    assert harness.context.state.discovered_rule_evaluations == [
        matches[0][0],
        matches[1][0],
    ]


# --- Eviction ledger --------------------------------------------------------------


async def test_that_sessions_without_an_eviction_ledger_load_with_an_empty_ledger(
    container: Container,
) -> None:
    rule = make_rule("HELP")
    harness = await matching_harness(
        container,
        [rule],
        rank_relevant=set(),
        session_metadata={_DISCOVERED_RULE_IDS_PATH: [str(rule.id)]},
    )

    await harness.matcher._load_discovered_rules(harness.context)

    assert harness.context.state.discovered_rules == {rule}
    assert harness.context.state.evicted_discovered_rules == {}


async def test_that_the_eviction_ledger_is_loaded_from_session_metadata(
    container: Container,
) -> None:
    member = make_rule("HELP")
    evicted = make_rule("REFUND")
    harness = await matching_harness(
        container,
        [member, evicted],
        rank_relevant=set(),
        session_metadata={
            _DISCOVERED_RULE_IDS_PATH: [str(member.id)],
            _EVICTED_DISCOVERED_RULES_PATH: {str(evicted.id): 7},
        },
    )

    await harness.matcher._load_discovered_rules(harness.context)

    assert harness.context.state.discovered_rules == {member}
    assert harness.context.state.evicted_discovered_rules == {evicted.id: 7}


async def test_that_a_readmitted_rule_is_removed_from_the_eviction_ledger(
    container: Container,
) -> None:
    rule = make_rule("REFUND")
    harness = await matching_harness(
        container,
        [rule],
        rank_relevant=set(),
        session_metadata={
            _DISCOVERED_RULE_IDS_PATH: [],
            _EVICTED_DISCOVERED_RULES_PATH: {str(rule.id): 5},
        },
    )
    harness.context.state.evicted_discovered_rules = {rule.id: 5}

    await harness.matcher._store_discovered_rules(
        harness.context,
        [
            (
                RuleEvaluation(rule=rule, reasoning="fresh trigger"),
                _ContextUsage.MATCH_CURRENT_TURN,
            )
        ],
    )

    metadata = await harness.staged_metadata()
    assert str(rule.id) in metadata_utils.internal_value(metadata, _DISCOVERED_RULE_IDS_PATH, [])
    assert str(rule.id) not in metadata_utils.internal_value(
        metadata, _EVICTED_DISCOVERED_RULES_PATH, {}
    )
    assert rule.id not in harness.context.state.evicted_discovered_rules


# --- Glossary working set ---------------------------------------------------------


async def test_that_load_glossary_loads_the_session_glossary_working_set(
    container: Container,
) -> None:
    harness = await matching_harness(
        container,
        [make_rule("TOPIC")],
        rank_relevant=set(),
    )
    term = await harness.glossary_store.create_term(
        name="PRS",
        description="Pinewood Rash Syndrome - an allergy to pinewood.",
    )

    await harness.matcher._load_glossary(harness.context)

    assert term in harness.context.state.glossary_terms


# --- Warm-up routing --------------------------------------------------------------
#
# Warm-up must agree with matching: the evaluators it prefixes are exactly
# those `_select_evaluator` would pick. `_get_prefill_targets` on a real
# matcher is the routing observable.


async def test_that_warm_up_prefills_the_selected_evaluator(container: Container) -> None:
    harness = await matching_harness(
        container,
        [make_rule("HELP")],
        rank_relevant=set(),
        effort=Effort.HIGH,
    )

    targets = await harness.matcher._get_prefill_targets(harness.context)

    assert len(targets) == 1


async def test_that_warm_up_prefills_the_evaluator_for_complex_rules_too(
    container: Container,
) -> None:
    rule = replace(
        make_rule("HELP"),
        content="Follow this detailed process. " * 20,
    )
    harness = await matching_harness(
        container,
        [rule],
        rank_relevant=set(),
        effort=Effort.HIGH,
    )

    targets = await harness.matcher._get_prefill_targets(harness.context)

    assert len(targets) == 1


async def test_that_warm_up_skips_the_evaluator_when_strategy_needs_none(
    container: Container,
) -> None:
    harness = await matching_harness(
        container,
        [make_rule("HELP")],
        rank_relevant=set(),
        effort=Effort.LOW,
    )

    targets = await harness.matcher._get_prefill_targets(harness.context)

    assert targets == []


async def test_that_warm_up_ranks_rule_that_can_raise_effort(container: Container) -> None:
    rule = replace(make_rule("REGULATED"), effort=Effort.HIGH)
    harness = await matching_harness(
        container,
        [rule],
        rank_relevant=set(),
        effort=Effort.LOW,
    )

    targets = await harness.matcher._get_prefill_targets(harness.context)

    assert len(targets) == 1


async def test_that_warm_up_includes_a_low_weight_dependency_target(
    container: Container,
) -> None:
    dependent = make_rule("DEPENDENT")
    dependency = replace(make_rule("LOW_DEPENDENCY"), weight=Weight.LOW)
    harness = await matching_harness(
        container,
        [dependent, dependency],
        rank_relevant=set(),
        effort=Effort.MEDIUM,
    )
    await harness.depend(dependent, dependency)

    targets = await harness.matcher._get_prefill_targets(harness.context)

    assert targets == [harness.matcher._rule_evaluator]


async def test_that_warm_up_is_skipped_when_the_fan_out_is_too_small(
    container: Container,
) -> None:
    # Prefill pays for one extra request to make the repeats cheap, and a cache
    # write costs more than an ordinary one — so warming to serve two requests
    # loses money.
    harness = await matching_harness(
        container,
        [make_rule("HELP"), make_rule("BILLING")],
        rank_relevant=set(),
        effort=Effort.HIGH,
        min_prefill_fan_out=3,
    )

    targets = await harness.matcher._get_prefill_targets(harness.context)

    assert targets == []


async def test_that_warm_up_happens_once_the_fan_out_clears_the_threshold(
    container: Container,
) -> None:
    harness = await matching_harness(
        container,
        [make_rule("HELP"), make_rule("BILLING"), make_rule("RETURNS")],
        rank_relevant=set(),
        effort=Effort.HIGH,
        min_prefill_fan_out=3,
    )

    targets = await harness.matcher._get_prefill_targets(harness.context)

    assert targets == [harness.matcher._rule_evaluator]


async def test_that_a_zero_threshold_turns_prefill_off_entirely(
    container: Container,
) -> None:
    harness = await matching_harness(
        container,
        [make_rule("HELP"), make_rule("BILLING"), make_rule("RETURNS")],
        rank_relevant=set(),
        effort=Effort.HIGH,
        min_prefill_fan_out=0,
    )

    targets = await harness.matcher._get_prefill_targets(harness.context)

    assert targets == []


async def test_that_warm_up_targets_exclude_statically_included_rules(
    container: Container,
) -> None:
    # A bare cross-referential rule selects no evaluator, so it must not
    # drag an evaluator prefix warm-up either.
    bare = replace(make_rule("BARE_NET"), metadata={"references_other_policies": True})
    harness = await matching_harness(
        container,
        [bare],
        rank_relevant=set(),
        effort=Effort.HIGH,
    )
    await harness.matcher._load_static_inclusion(harness.context)

    targets = await harness.matcher._get_prefill_targets(harness.context)

    assert targets == []


# --- Turn matching vs. session discovery (full real _match) -----------------------


async def test_that_low_weight_rule_that_can_raise_effort_is_matched_this_turn(
    container: Container,
) -> None:
    rule = replace(make_rule("REGULATED"), weight=Weight.LOW, effort=Effort.HIGH)
    harness = await matching_harness(
        container,
        [rule],
        rank_relevant={"REGULATED"},
        effort=Effort.LOW,
    )

    await harness.match()

    assert [m.rule.id for m in harness.context.state.ordinary_rule_matches] == [rule.id]
    assert rule in harness.context.state.discovered_rules


async def test_that_ranked_rule_can_still_be_discovered_into_session_by_recall(
    container: Container,
) -> None:
    rule = replace(make_rule("REGULATED"), weight=Weight.HIGH)
    harness = await matching_harness(
        container,
        [rule],
        rank_relevant=set(),  # The ranker judges it not relevant this turn...
        conversation="RECALL_REGULATED",  # ...but recall still finds it.
    )

    await harness.match()

    assert harness.context.state.ordinary_rule_matches == []
    assert await harness.staged_discovered_rule_ids() == {str(rule.id)}


async def test_that_an_evaluated_rule_can_still_be_discovered_into_session_by_recall(
    container: Container,
) -> None:
    rule = replace(
        make_rule("WORKFLOW"),
        weight=Weight.HIGH,
        content="This workflow has many details. " * 20,
    )
    harness = await matching_harness(
        container,
        [rule],
        rank_relevant=set(),
        conversation="RECALL_WORKFLOW",
    )

    await harness.match()

    assert harness.context.state.ordinary_rule_matches == []
    assert await harness.staged_discovered_rule_ids() == {str(rule.id)}


async def test_that_turn_match_takes_precedence_over_session_recall_discovery(
    container: Container,
) -> None:
    rule = replace(make_rule("REGULATED"), weight=Weight.HIGH)
    harness = await matching_harness(
        container,
        [rule],
        rank_relevant={"REGULATED"},
        conversation="RECALL_REGULATED",
    )

    await harness.match()

    # Both the ranker and recall fired; the turn verdict wins the state.
    assert [m.rule.id for m in harness.context.state.ordinary_rule_matches] == [rule.id]
    assert await harness.staged_discovered_rule_ids() == {str(rule.id)}


# --- Per-rule glossary terms (interpretation dependencies) ------------------------
#
# Turn evaluators judge one rule per prompt; a rule may depend on glossary
# terms to be interpreted correctly, so the matcher resolves each evaluated
# rule's relevant terms into state.terms_by_rule before the batches run.
# The lookup seam is observed via the harness's recording EntityQueries proxy
# (real behavior, recorded calls) — the embedding cache makes the embedder
# boundary blind to repeated identical queries.


async def test_that_rule_terms_are_loaded_for_evaluated_rules(container: Container) -> None:
    rule = make_rule("PRS_CASE")  # MEDIUM weight at HIGH effort -> RANK
    harness = await matching_harness(container, [rule], rank_relevant=set())
    term = await harness.glossary_store.create_term(
        name="PRS",
        description="Pinewood Rash Syndrome - an allergy to pinewood.",
    )

    await harness.match()

    assert harness.context.state.terms_by_rule.get(rule.id) == [term]
    assert len(harness.entity_queries.find_glossary_calls) == 1
    query, k = harness.entity_queries.find_glossary_calls[0]
    assert "RANK_PRS_CASE" in query
    assert k == Matcher._TERMS_PER_RULE


async def test_that_recall_strategy_rules_do_not_fetch_terms(container: Container) -> None:
    rules = [make_rule(f"TOPIC_{i}") for i in range(6)]  # MEDIUM weight/effort -> RECALL
    harness = await matching_harness(
        container,
        rules,
        rank_relevant=set(),
        effort=Effort.MEDIUM,
    )
    await harness.glossary_store.create_term(name="PRS", description="Pinewood Rash Syndrome.")

    await harness.match()

    assert harness.entity_queries.find_glossary_calls == []


async def test_that_rule_term_lookups_are_memoized_across_turns(container: Container) -> None:
    rule = make_rule("PRS_CASE")
    harness = await matching_harness(container, [rule], rank_relevant=set())
    term = await harness.glossary_store.create_term(
        name="PRS", description="Pinewood Rash Syndrome."
    )

    await harness.match()
    await harness.match()

    # The lookup ran once; the second turn was served from the memo — but the
    # state still carries the terms.
    assert len(harness.entity_queries.find_glossary_calls) == 1
    assert harness.context.state.terms_by_rule.get(rule.id) == [term]


async def test_that_editing_a_rule_invalidates_its_term_lookup(container: Container) -> None:
    rule = make_rule("PRS_CASE")
    harness = await matching_harness(container, [rule], rank_relevant=set())
    await harness.glossary_store.create_term(name="PRS", description="Pinewood Rash Syndrome.")

    await harness.match()

    edited = replace(rule, modified_utc=rule.modified_utc + timedelta(seconds=1))
    harness.context.state.usable_rules = [edited]
    await harness.match()

    assert len(harness.entity_queries.find_glossary_calls) == 2


async def test_that_a_glossary_change_invalidates_rule_term_lookups(
    container: Container,
) -> None:
    rule = make_rule("PRS_CASE")
    harness = await matching_harness(container, [rule], rank_relevant=set())
    await harness.glossary_store.create_term(name="PRS", description="Pinewood Rash Syndrome.")

    await harness.match()

    # A new term joins the glossary at runtime: the mapping must be recomputed.
    await harness.glossary_store.create_term(
        name="RMA-3", description="The escalation procedure for medical cases."
    )
    await harness.match()

    assert len(harness.entity_queries.find_glossary_calls) == 2


async def test_that_an_empty_glossary_skips_rule_term_lookups(container: Container) -> None:
    rule = make_rule("PRS_CASE")
    harness = await matching_harness(container, [rule], rank_relevant=set())

    await harness.match()

    assert harness.entity_queries.find_glossary_calls == []
    assert harness.context.state.terms_by_rule == {}


# --- The evaluator-selection seam -------------------------------------------------
#
# The matcher selects a turn evaluator per rule (None = discovery-only) and
# groups rules by evaluator instance — natively multi-evaluator, so a subclass
# can route specific rules to its own evaluator by overriding _select_evaluator.
# The routed evaluator is a plug-in port (an LLM component in production), so a
# recording script at that port is the boundary fake here.


class _RecordingTurnEvaluator(TurnEvaluator):
    def __init__(self, results: list[RuleEvaluation] | None = None) -> None:
        self.batches: list[list[Rule]] = []
        self.warm_ups = 0
        self._results = results or []

    async def evaluate(self, context: EngineContext, rules: Any) -> RuleEvaluationResult:
        self.batches.append(list(rules))
        return RuleEvaluationResult(self._results, None)

    async def warm_up(self, context: EngineContext) -> GenerationInfo | None:
        self.warm_ups += 1
        return None


class _RoutingMatcher(Matcher):
    """A test subclass routing rules whose condition mentions 'special' to a
    second evaluator — the extension seam a plug-in matcher uses."""

    special_evaluator: _RecordingTurnEvaluator

    def _select_evaluator(self, context, rule):
        if "special" in rule.condition:
            return self.special_evaluator
        return super()._select_evaluator(context, rule)


async def _routing_harness(
    container: Container,
    rules: list[Rule],
    *,
    rank_relevant: set[str] = set(),
    effort: Effort = Effort.MEDIUM,
    special_results: list[RuleEvaluation] | None = None,
) -> MatchingHarness:
    harness = await matching_harness(
        container,
        rules,
        rank_relevant=rank_relevant,
        effort=effort,
        matcher_type=_RoutingMatcher,
    )
    cast(_RoutingMatcher, harness.matcher).special_evaluator = _RecordingTurnEvaluator(
        special_results
    )
    return harness


async def test_that_select_evaluator_returns_the_bound_evaluator_for_rank_tier_rules(
    container: Container,
) -> None:
    rank_tier = replace(make_rule("URGENT"), weight=Weight.HIGH)
    recall_tier = make_rule("BACKGROUND")  # Weight.MEDIUM
    fillers = [make_rule(f"FILLER_{i}") for i in range(5)]
    harness = await matching_harness(
        container,
        [rank_tier, recall_tier, *fillers],
        rank_relevant=set(),
        effort=Effort.MEDIUM,
    )
    await harness.matcher._load_strategy_choice_signals(
        harness.context, [rank_tier, recall_tier, *fillers]
    )

    assert (
        harness.matcher._select_evaluator(harness.context, rank_tier)
        is harness.matcher._rule_evaluator
    )
    assert harness.matcher._select_evaluator(harness.context, recall_tier) is None


@pytest.mark.parametrize(
    ("effort", "expects_evaluator"),
    [
        (Effort.MIN, False),
        (Effort.LOW, False),
        (Effort.MEDIUM, True),
    ],
)
async def test_that_tool_attachments_require_rank_only_from_medium_effort(
    container: Container,
    effort: Effort,
    expects_evaluator: bool,
) -> None:
    tool_rule = make_rule("TOOL_RULE")  # MEDIUM weight normally uses RECALL here.
    fillers = [make_rule(f"FILLER_{index}") for index in range(5)]
    harness = await matching_harness(
        container,
        [tool_rule, *fillers],
        rank_relevant=set(),
        effort=effort,
    )
    harness.context.state.rule_ids_with_tools = {tool_rule.id}

    evaluator = harness.matcher._select_evaluator(harness.context, tool_rule)

    assert (evaluator is harness.matcher._rule_evaluator) is expects_evaluator


async def test_that_only_dependency_targets_are_escalated_to_rank(
    container: Container,
) -> None:
    dependent = make_rule("DEPENDENT")
    dependency = make_rule("DEPENDENCY")
    fillers = [make_rule(f"FILLER_{index}") for index in range(5)]
    rules = [dependent, dependency, *fillers]
    harness = await matching_harness(
        container,
        rules,
        rank_relevant=set(),
        effort=Effort.MEDIUM,
    )
    await harness.depend(dependent, dependency)

    await harness.matcher._load_strategy_choice_signals(harness.context, rules)

    assert harness.matcher._select_evaluator(harness.context, dependent) is None
    assert (
        harness.matcher._select_evaluator(harness.context, dependency)
        is harness.matcher._rule_evaluator
    )


async def test_that_a_low_weight_dependency_target_is_still_ranked(
    container: Container,
) -> None:
    dependent = make_rule("DEPENDENT")
    dependency = replace(make_rule("LOW_DEPENDENCY"), weight=Weight.LOW)
    fillers = [make_rule(f"FILLER_{index}") for index in range(5)]
    harness = await matching_harness(
        container,
        [dependent, dependency, *fillers],
        rank_relevant={"LOW_DEPENDENCY"},
        conversation="RECALL_DEPENDENT",
        effort=Effort.MEDIUM,
    )
    await harness.depend(dependent, dependency)

    await harness.match()

    assert harness.rank_generator.calls == ["LOW_DEPENDENCY"]
    assert await harness.staged_discovered_rule_ids() == {
        str(dependent.id),
        str(dependency.id),
    }


async def test_that_a_rule_with_a_match_handler_is_escalated_to_rank(
    container: Container,
) -> None:
    handled = make_rule("HANDLED")
    fillers = [make_rule(f"FILLER_{index}") for index in range(5)]
    rules = [handled, *fillers]
    harness = await matching_harness(
        container,
        rules,
        rank_relevant=set(),
        effort=Effort.MEDIUM,
    )

    async def on_matched(ctx: EngineContext, evaluation: object) -> None:
        pass

    harness.hooks.on_rule_matched_handlers[handled.id].append(on_matched)  # type: ignore[arg-type]

    await harness.matcher._load_strategy_choice_signals(harness.context, rules)

    # Code attached to a match verdict declares the match consequential; the
    # rule gets a real per-turn evaluation rather than recall alone.
    assert (
        harness.matcher._select_evaluator(harness.context, handled)
        is harness.matcher._rule_evaluator
    )
    assert harness.matcher._select_evaluator(harness.context, fillers[0]) is None


async def test_that_a_discovery_handler_alone_does_not_escalate(
    container: Container,
) -> None:
    discovered = make_rule("DISCOVERED")
    fillers = [make_rule(f"FILLER_{index}") for index in range(5)]
    rules = [discovered, *fillers]
    harness = await matching_harness(
        container,
        rules,
        rank_relevant=set(),
        effort=Effort.MEDIUM,
    )

    async def on_discovered(ctx: EngineContext, evaluation: object) -> None:
        pass

    harness.hooks.on_rule_discovered_handlers[discovered.id].append(on_discovered)  # type: ignore[arg-type]

    await harness.matcher._load_strategy_choice_signals(harness.context, rules)

    # on_discovered is recall's native, cheap-tier event — it must not
    # silently buy a per-turn evaluation.
    assert harness.matcher._select_evaluator(harness.context, discovered) is None


async def test_that_a_dependency_source_with_attachments_still_ranks(
    container: Container,
) -> None:
    source = replace(make_rule("SOURCE"), effort=Effort.HIGH)
    target = make_rule("TARGET")
    fillers = [make_rule(f"FILLER_{index}") for index in range(5)]
    rules = [source, target, *fillers]
    harness = await matching_harness(
        container,
        rules,
        rank_relevant=set(),
        effort=Effort.MEDIUM,
    )
    await harness.depend(source, target)

    await harness.matcher._load_strategy_choice_signals(harness.context, rules)

    # Being a dependency source never suppresses escalation: the effort raise
    # earns the evaluator on its own merits, dependency edge or not.
    assert (
        harness.matcher._select_evaluator(harness.context, source)
        is harness.matcher._rule_evaluator
    )


async def test_that_dependency_targets_are_mapped_per_source_rule(
    container: Container,
) -> None:
    dependent = make_rule("DEPENDENT")
    direct_target = make_rule("DIRECT_TARGET")
    group_member = replace(make_rule("GROUP_MEMBER"), groups=[GroupId("scope-group")])
    group_dependent = make_rule("GROUP_DEPENDENT")
    unrelated = make_rule("UNRELATED")
    rules = [dependent, direct_target, group_member, group_dependent, unrelated]
    harness = await matching_harness(
        container,
        rules,
        rank_relevant=set(),
        effort=Effort.MEDIUM,
    )
    await harness.depend(dependent, direct_target)
    await harness.depend_on_group(
        group_dependent,
        GroupId("scope-group"),
        target_kind=RelationshipEntityKind.GROUP_ALL,
    )

    await harness.matcher._load_strategy_choice_signals(harness.context, rules)

    targets = harness.context.state.dependency_targets_by_rule
    assert [rule.id for rule in targets[dependent.id]] == [direct_target.id]
    assert [rule.id for rule in targets[group_dependent.id]] == [group_member.id]
    # Only sources carry context; targets and bystanders map to nothing.
    assert direct_target.id not in targets
    assert unrelated.id not in targets


async def test_that_dependency_targets_are_resolved_from_the_full_inventory(
    container: Container,
) -> None:
    dependent = make_rule("DEPENDENT")
    dependency = make_rule("DEPENDENCY")
    unrelated = make_rule("UNRELATED")
    rules = [dependent, dependency, unrelated]
    harness = await matching_harness(
        container,
        rules,
        rank_relevant=set(),
        effort=Effort.MEDIUM,
    )
    await harness.depend(dependent, dependency)

    # Mid-turn reevaluation passes only its gated subset, but prompt context
    # and target strategy signals must still resolve against the full inventory.
    await harness.matcher._load_strategy_choice_signals(harness.context, [dependent])

    assert harness.context.state.dependency_targets_by_rule[dependent.id] == [dependency]
    assert dependency.id in harness.context.state.rule_ids_with_dependents


async def test_that_a_recalled_dependent_survives_when_its_ranked_dependency_matches(
    container: Container,
) -> None:
    dependent = make_rule("DEPENDENT")
    dependency = make_rule("DEPENDENCY")
    fillers = [make_rule(f"FILLER_{index}") for index in range(5)]
    harness = await matching_harness(
        container,
        [dependent, dependency, *fillers],
        rank_relevant={"DEPENDENCY"},
        conversation="RECALL_DEPENDENT",
        effort=Effort.MEDIUM,
    )
    await harness.depend(dependent, dependency)

    await harness.match()

    assert harness.rank_generator.calls == ["DEPENDENCY"]
    assert await harness.staged_discovered_rule_ids() == {
        str(dependent.id),
        str(dependency.id),
    }


async def test_that_members_of_a_dependency_target_group_are_escalated_to_rank(
    container: Container,
) -> None:
    target_group = await container[GroupStore].create_group(name="dependency-targets")
    dependent = make_rule("DEPENDENT")
    first_dependency = replace(make_rule("FIRST_DEPENDENCY"), groups=[target_group.id])
    second_dependency = replace(make_rule("SECOND_DEPENDENCY"), groups=[target_group.id])
    fillers = [make_rule(f"FILLER_{index}") for index in range(4)]
    rules = [dependent, first_dependency, second_dependency, *fillers]
    harness = await matching_harness(
        container,
        rules,
        rank_relevant=set(),
        effort=Effort.MEDIUM,
    )
    await harness.depend_on_group(
        dependent,
        target_group.id,
        target_kind=RelationshipEntityKind.GROUP_ANY,
    )

    await harness.matcher._load_strategy_choice_signals(harness.context, rules)

    assert harness.matcher._select_evaluator(harness.context, dependent) is None
    assert (
        harness.matcher._select_evaluator(harness.context, first_dependency)
        is harness.matcher._rule_evaluator
    )
    assert (
        harness.matcher._select_evaluator(harness.context, second_dependency)
        is harness.matcher._rule_evaluator
    )


async def test_that_rules_are_grouped_by_selected_evaluator(container: Container) -> None:
    special_1 = replace(
        make_rule("SPECIAL_ONE", extra_condition="a special topic"), weight=Weight.HIGH
    )
    special_2 = replace(
        make_rule("SPECIAL_TWO", extra_condition="another special topic"), weight=Weight.HIGH
    )
    normal = replace(make_rule("NORMAL"), weight=Weight.HIGH)
    discovery_only = make_rule("BACKGROUND")
    harness = await _routing_harness(
        container,
        [special_1, special_2, normal, discovery_only],
        effort=Effort.MEDIUM,
    )

    await harness.match()

    special = cast(_RoutingMatcher, harness.matcher).special_evaluator
    assert len(special.batches) == 1
    assert set(special.batches[0]) == {special_1, special_2}
    # The base ranker saw only the normal rule; the discovery-only rule was
    # never evaluated by anyone.
    assert harness.rank_generator.calls == ["NORMAL"]


async def test_that_matches_from_a_routed_evaluator_carry_its_highlights(
    container: Container,
) -> None:
    rule = replace(
        make_rule("SPECIAL_BILLING", extra_condition="special billing"), weight=Weight.HIGH
    )
    harness = await _routing_harness(
        container,
        [rule],
        effort=Effort.MEDIUM,
        special_results=[
            RuleEvaluation(
                rule=rule,
                reasoning="applies now",
                is_relevant=True,
                highlights=["Charge exactly $5."],
            )
        ],
    )

    await harness.match()

    assert [m.highlights for m in harness.context.state.ordinary_rule_matches] == [
        ["Charge exactly $5."]
    ]


async def test_that_warm_up_prefills_each_distinct_selected_evaluator_once(
    container: Container,
) -> None:
    special = replace(
        make_rule("SPECIAL_TOPIC", extra_condition="a special topic"), weight=Weight.HIGH
    )
    normal = replace(make_rule("NORMAL"), weight=Weight.HIGH)
    harness = await _routing_harness(container, [special, normal], effort=Effort.MEDIUM)

    targets = await harness.matcher._get_prefill_targets(harness.context)

    assert len(targets) == 2
    assert cast(_RoutingMatcher, harness.matcher).special_evaluator in targets


async def test_that_a_subclass_can_route_a_small_inventory_rule_to_its_own_evaluator(
    container: Container,
) -> None:
    # The fast path skips discovery, not the `_select_evaluator` seam: a
    # subclass routing rules of its own accord must still be consulted.
    rule = make_rule("SPECIAL_TOPIC", extra_condition="a special topic")
    harness = await _routing_harness(
        container,
        [rule],
        effort=Effort.MEDIUM,
        special_results=[_relevant(rule)],
    )

    await harness.match()

    assert harness.embedder.calls == []  # no discovery round trip
    special = cast(_RoutingMatcher, harness.matcher).special_evaluator
    assert special.batches == [[rule]]
    # Its evaluator turn-matched it — the turn verdict wins the state.
    assert [m.rule.id for m in harness.context.state.ordinary_rule_matches] == [rule.id]


# --- Relational resolution: dependency gating and turn priority -------------------
#
# Dependency gating applies to EVERYTHING the matcher emits; priority applies
# ONLY to the turn-matched state (outranked rules still enter the session
# set). Deeper dependency semantics live in test_multi_phase_evaluation.py.


async def test_that_a_rule_with_an_unmet_dependency_is_kept_out_of_turn_and_session_state(
    container: Container,
) -> None:
    dependent = make_rule("DEPENDENT")
    dependency = make_rule("DEPENDENCY")
    harness = await matching_harness(
        container,
        [dependent, dependency],
        rank_relevant={"DEPENDENT"},
    )
    await harness.depend(dependent, dependency)

    await harness.match()

    assert harness.context.state.ordinary_rule_matches == []
    assert harness.context.state.discovered_rules == set()
    assert await harness.staged_discovered_rule_ids() == set()


async def test_that_a_dependent_rule_matches_when_its_dependency_matches_the_same_turn(
    container: Container,
) -> None:
    dependent = make_rule("DEPENDENT")
    dependency = make_rule("DEPENDENCY")
    harness = await matching_harness(
        container,
        [dependent, dependency],
        rank_relevant={"DEPENDENT", "DEPENDENCY"},
    )
    await harness.depend(dependent, dependency)

    await harness.match()

    assert {m.rule.id for m in harness.context.state.ordinary_rule_matches} == {
        dependent.id,
        dependency.id,
    }
    assert harness.context.state.discovered_rules == {dependent, dependency}


async def test_that_only_top_priority_turn_matches_stay_in_the_turn_state(
    container: Container,
) -> None:
    low = replace(make_rule("GENERAL"), priority=0)
    high = replace(make_rule("URGENT"), priority=10)
    harness = await matching_harness(
        container,
        [low, high],
        rank_relevant={"GENERAL", "URGENT"},
    )

    await harness.match()

    assert [m.rule.id for m in harness.context.state.ordinary_rule_matches] == [high.id]
    # The outranked rule still enters the session working set.
    assert harness.context.state.discovered_rules == {low, high}


async def test_that_discovery_admissions_are_not_priority_filtered(
    container: Container,
) -> None:
    high = replace(make_rule("URGENT"), priority=10)
    discovered = replace(make_rule("BACKGROUND"), priority=0)
    harness = await matching_harness(
        container,
        [high, discovered],
        rank_relevant={"URGENT"},
        conversation="RECALL_BACKGROUND",
    )

    await harness.match()

    assert [m.rule.id for m in harness.context.state.ordinary_rule_matches] == [high.id]
    assert discovered in harness.context.state.discovered_rules


async def test_that_a_higher_priority_mid_turn_match_drops_recorded_lower_priority_matches(
    container: Container,
) -> None:
    low = replace(make_rule("GENERAL"), priority=0)
    high = replace(make_rule("URGENT"), priority=10)
    harness = await matching_harness(container, [low, high], rank_relevant=set())
    harness.context.state.ordinary_rule_matches = [
        RuleEvaluation(rule=low, reasoning="matched earlier")
    ]
    harness.context.state.discovered_rules = {low}

    await harness.matcher._record(
        harness.context,
        [
            (
                RuleEvaluation(rule=high, reasoning="matched mid-turn"),
                _ContextUsage.MATCH_CURRENT_TURN,
            )
        ],
        append=True,
    )

    # An accepted responder-prefix break: the stronger rule displaces the
    # already-recorded one mid-turn.
    assert [m.rule.id for m in harness.context.state.ordinary_rule_matches] == [high.id]
    assert harness.context.state.discovered_rules == {low, high}


async def test_that_lower_priority_mid_turn_matches_are_demoted_to_the_session_set(
    container: Container,
) -> None:
    low = replace(make_rule("GENERAL"), priority=0)
    high = replace(make_rule("URGENT"), priority=10)
    harness = await matching_harness(container, [low, high], rank_relevant=set())
    harness.context.state.ordinary_rule_matches = [
        RuleEvaluation(rule=high, reasoning="matched earlier")
    ]
    harness.context.state.discovered_rules = {high}

    await harness.matcher._record(
        harness.context,
        [
            (
                RuleEvaluation(rule=low, reasoning="matched mid-turn"),
                _ContextUsage.MATCH_CURRENT_TURN,
            )
        ],
        append=True,
    )

    assert [m.rule.id for m in harness.context.state.ordinary_rule_matches] == [high.id]
    assert harness.context.state.discovered_rules == {low, high}


# --- The small-inventory fast path ------------------------------------------------
#
# An agent whose entire inventory fits in the responder's instructions gains
# nothing from embedding-based discovery: every rule would be rendered anyway.
# Below the threshold the matcher admits them directly and never touches the
# embedding boundary.


async def test_that_rules_are_admitted_without_recall_when_the_inventory_is_small(
    container: Container,
) -> None:
    rules = [make_rule(f"TOPIC_{i}") for i in range(Matcher._SMALL_INVENTORY_THRESHOLD)]
    harness = await matching_harness(
        container,
        rules,
        rank_relevant=set(),
        effort=Effort.MEDIUM,
    )

    await harness.match()

    assert harness.embedder.calls == []
    assert harness.rank_generator.calls == []
    assert await harness.staged_discovered_rule_ids() == {str(g.id) for g in rules}


async def test_that_an_inventory_above_the_threshold_still_recalls(
    container: Container,
) -> None:
    rules = [make_rule(f"TOPIC_{i}") for i in range(Matcher._SMALL_INVENTORY_THRESHOLD + 1)]
    harness = await matching_harness(
        container,
        rules,
        rank_relevant=set(),
        effort=Effort.MEDIUM,
    )

    await harness.match()

    assert harness.embedder.calls != []  # the embedding boundary was consulted
    assert await harness.staged_discovered_rule_ids() == set()  # no marker -> no hit


async def test_that_a_large_inventory_with_a_small_discovery_batch_still_recalls(
    container: Container,
) -> None:
    # The batch shrinks as rules join the session set, so it is not a proxy
    # for inventory size: a big agent must not fall into the fast path on a
    # late turn just because most of its rules have already been admitted.
    rules = [make_rule(f"TOPIC_{i}") for i in range(Matcher._SMALL_INVENTORY_THRESHOLD + 4)]
    harness = await matching_harness(
        container,
        rules,
        rank_relevant=set(),
        effort=Effort.MEDIUM,
        discovered_rules=set(rules[:-2]),
    )

    await harness.match()

    assert harness.embedder.calls != []


async def test_that_a_small_inventory_still_ranks_rules_that_escalate_to_rank(
    container: Container,
) -> None:
    ranked = replace(make_rule("URGENT"), weight=Weight.HIGH)
    recalled = make_rule("BACKGROUND")
    harness = await matching_harness(
        container,
        [ranked, recalled],
        rank_relevant={"URGENT"},
        effort=Effort.MEDIUM,
    )

    await harness.match()

    assert harness.rank_generator.calls == ["URGENT"]
    assert [m.rule.id for m in harness.context.state.ordinary_rule_matches] == [ranked.id]
    # The recall-tier rule was admitted by the fast path alongside the match.
    assert await harness.staged_discovered_rule_ids() == {str(ranked.id), str(recalled.id)}


# --- Cross-referential rules: bare vs attached routing ----------------------------
#
# A rule flagged as referencing the agent's other policies is routed by
# whether anything mechanical hangs off its match. Bare: statically included,
# never discovered, never ranked (end-to-end coverage lives in
# test_multi_phase_evaluation.py). Attached: escalated to RANK so its
# attachments can actually fire. See matching/common.py for the motivation.


def _flagged_rule(name: str, **overrides: Any) -> Rule:
    return replace(
        make_rule(name),
        metadata={"references_other_policies": True},
        **overrides,
    )


async def test_that_an_attached_cross_referential_rule_is_escalated_to_rank(
    container: Container,
) -> None:
    attached = _flagged_rule("COVERAGE_NET", message_review=True)
    harness = await matching_harness(
        container,
        [attached],
        rank_relevant=set(),
        effort=Effort.MIN,  # MIN never ranks by the strategy matrix alone
    )
    await harness.matcher._load_strategy_choice_signals(harness.context, [attached])
    await harness.matcher._load_static_inclusion(harness.context)

    assert attached not in harness.context.state.statically_included_rules
    assert harness.matcher._select_evaluator(harness.context, attached) is not None


async def test_that_a_dependency_source_still_counts_as_a_match_attachment(
    container: Container,
) -> None:
    dependent = _flagged_rule("COVERAGE_NET")
    dependency = make_rule("FOUNDATION")
    harness = await matching_harness(
        container,
        [dependent, dependency],
        rank_relevant=set(),
        effort=Effort.MEDIUM,
    )
    await harness.depend(dependent, dependency)

    await harness.matcher._load_static_inclusion(harness.context)

    assert dependent not in harness.context.state.statically_included_rules


async def test_that_a_review_requesting_rule_is_escalated_to_rank(
    container: Container,
) -> None:
    # A rule carrying message_review/tool_review needs a real match verdict
    # for its review request to ever trigger — flagged or not.
    rule = replace(make_rule("RETENTION"), message_review=True)
    harness = await matching_harness(
        container,
        [rule],
        rank_relevant=set(),
        effort=Effort.MEDIUM,  # MEDIUM weight -> RECALL by the matrix
    )
    await harness.matcher._load_strategy_choice_signals(harness.context, [rule])

    assert harness.matcher._select_evaluator(harness.context, rule) is not None


async def test_that_a_matched_handler_counts_as_an_attachment(container: Container) -> None:
    flagged = _flagged_rule("COVERAGE_NET")
    harness = await matching_harness(
        container,
        [flagged],
        rank_relevant=set(),
        effort=Effort.MEDIUM,
    )

    async def handler(*args: Any) -> None: ...

    harness.hooks.on_rule_matched_handlers[flagged.id].append(handler)
    await harness.matcher._load_strategy_choice_signals(harness.context, [flagged])
    await harness.matcher._load_static_inclusion(harness.context)

    assert flagged not in harness.context.state.statically_included_rules


async def test_that_static_inclusion_is_empty_when_no_rules_are_flagged(
    container: Container,
) -> None:
    harness = await matching_harness(
        container,
        [make_rule("PRICING")],
        rank_relevant=set(),
        effort=Effort.MEDIUM,
    )

    await harness.matcher._load_static_inclusion(harness.context)

    assert harness.context.state.statically_included_rules == []


async def test_that_the_cross_referential_escalation_beats_the_small_inventory_fast_path(
    container: Container,
) -> None:
    # In a tiny inventory, would-be-RECALL rules take the NONE fast path — but
    # an attached cross-referential rule must still be RANKED (its attachments
    # need a verdict, and its context is the full inventory anyway).
    attached = _flagged_rule("COVERAGE_NET", message_review=True)
    plain = make_rule("PRICING")
    harness = await matching_harness(
        container,
        [attached, plain],
        rank_relevant=set(),
        effort=Effort.MEDIUM,
    )
    await harness.matcher._load_strategy_choice_signals(harness.context, [attached, plain])
    await harness.matcher._load_static_inclusion(harness.context)

    assert harness.matcher._select_evaluator(harness.context, plain) is None
    assert harness.matcher._select_evaluator(harness.context, attached) is not None


# --- Attachment predicate: MC/DC-style coverage -----------------------------------
#
# _check_if_has_match_attachments is a pure disjunction over nine attachment
# kinds. For a disjunction, MC/DC reduces to: each condition true ALONE flips
# the outcome, plus the all-false case. Real registry, real hooks, real state.


@pytest.mark.parametrize(
    "attachment",
    [
        "labels",
        "effort",
        "message_review",
        "tool_review",
        "tools_signal",
        "dependencies_signal",
        "code_evaluator",
        "matched_handler",
        "discovered_handler",
        "priority",
        "evaluation_method",
    ],
)
async def test_that_each_attachment_alone_marks_a_rule_attached(
    container: Container,
    attachment: str,
) -> None:
    rule = _flagged_rule("COVERAGE_NET")
    harness = await matching_harness(
        container,
        [rule],
        rank_relevant=set(),
        effort=Effort.MEDIUM,
    )
    context = harness.context

    async def handler(*args: Any) -> None: ...

    async def code_evaluator(context: EngineContext, rule: Rule) -> RuleEvaluation:
        return _relevant(rule)

    match attachment:
        case "labels":
            rule = replace(rule, labels={"compliance"})
        case "effort":
            rule = replace(rule, effort=Effort.HIGH)
        case "message_review":
            rule = replace(rule, message_review=True)
        case "tool_review":
            rule = replace(rule, tool_review=True)
        case "tools_signal":
            context.state.rule_ids_with_tools = {rule.id}
        case "dependencies_signal":
            context.state.rule_ids_with_dependencies = {rule.id}
        case "code_evaluator":
            harness.evaluator_registry.register(rule.id, code_evaluator)
        case "matched_handler":
            harness.hooks.on_rule_matched_handlers[rule.id].append(handler)
        case "discovered_handler":
            harness.hooks.on_rule_discovered_handlers[rule.id].append(handler)
        case "priority":
            # Priority fires only through turn-match arbitration
            # (_resolve_turn_priority) — statically including a prioritized
            # rule would mean its priority never suppresses anything.
            rule = replace(rule, priority=10)
        case "evaluation_method":
            # An explicit evaluation request directly contradicts static
            # inclusion ("never evaluated") — the explicit knob must win.
            rule = replace(
                rule,
                metadata=metadata_utils.set_internal_value(
                    rule.metadata, ("evaluation_method",), "rank"
                ),
            )

    assert harness.matcher._check_if_has_match_attachments(context, rule)


async def test_that_a_rule_with_no_attachments_is_not_attached(container: Container) -> None:
    rule = _flagged_rule("COVERAGE_NET")
    harness = await matching_harness(
        container,
        [rule],
        rank_relevant=set(),
        effort=Effort.MEDIUM,
    )

    assert not harness.matcher._check_if_has_match_attachments(harness.context, rule)


# --- Discovered rule pruning (cap + eviction) -------------------------------------
#
# All classical: real pruner, real session store; the pruner's per-rule
# verdicts are scripted at the model boundary (same matching.rank route and
# schema as ranking — see harness.ScriptedRankGenerator.scores_by_name).


def _members(count: int, *, weight: Weight = Weight.MEDIUM, prefix: str = "MEMBER") -> list[Rule]:
    return [replace(make_rule(f"{prefix}_{i:03d}"), weight=weight) for i in range(count)]


def _member_names(members: list[Rule]) -> list[str]:
    return [g.condition.split()[0].removeprefix("RANK_") for g in members]


def _conversation_events(count: int = 3) -> list[Any]:
    sources = [EventSource.CUSTOMER, EventSource.AI_AGENT, EventSource.CUSTOMER]
    return [
        create_event_message(
            offset=i,
            source=sources[i % len(sources)],
            message=f"message {i}",
        )
        for i in range(count)
    ]


async def _pruning_harness(
    container: Container,
    members: list[Rule],
    *,
    scores_by_name: dict[str, int],
    conversation_events: list[Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> MatchingHarness:
    return await matching_harness(
        container,
        members,
        rank_relevant=set(),
        scores_by_name=scores_by_name,
        conversation_events=conversation_events
        if conversation_events is not None
        else _conversation_events(),
        discovered_rules=set(members),
        session_metadata={
            _DISCOVERED_RULE_IDS_PATH: [str(g.id) for g in members],
            **(extra_metadata or {}),
        },
    )


async def test_that_session_rule_pruning_is_skipped_at_or_below_the_high_water_mark(
    container: Container,
) -> None:
    members = _members(Matcher._DISCOVERED_RULES_HIGH_WATER_MARK)
    harness = await _pruning_harness(container, members, scores_by_name={})

    await harness.matcher.prune_discovered_rules(harness.context)

    assert harness.rank_generator.calls == []  # no pruning verdicts requested
    assert harness.context.state.discovered_rules == set(members)
    metadata = await harness.staged_metadata()
    assert metadata_utils.internal_value(metadata, _EVICTED_DISCOVERED_RULES_PATH) is None


async def test_that_session_rule_pruning_evicts_stale_rules_down_to_the_target(
    container: Container,
) -> None:
    members = _members(Matcher._DISCOVERED_RULES_HIGH_WATER_MARK + 1)
    names = _member_names(members)
    excess = len(members) - Matcher._DISCOVERED_RULES_TARGET
    # The first `excess` members are judged stale; the rest confidently fresh.
    scores = {name: (1 if i < excess else 5) for i, name in enumerate(names)}
    harness = await _pruning_harness(container, members, scores_by_name=scores)

    await harness.matcher.prune_discovered_rules(harness.context)

    survivors = members[excess:]
    evicted = members[:excess]
    assert harness.context.state.discovered_rules == set(survivors)

    # Evictions are ledgered at the interaction's last event offset, and the
    # pruning itself is stamped for the rate limiter.
    last_offset = max(e.offset for e in harness.context.interaction.events)
    metadata = await harness.staged_metadata()
    assert set(metadata_utils.internal_value(metadata, _DISCOVERED_RULE_IDS_PATH, [])) == {
        str(g.id) for g in survivors
    }
    ledger = metadata_utils.internal_value(metadata, _EVICTED_DISCOVERED_RULES_PATH, {})
    assert set(ledger) == {str(g.id) for g in evicted}
    assert all(offset == last_offset for offset in ledger.values())
    assert (
        metadata_utils.internal_value(metadata, _LAST_DISCOVERED_RULE_PRUNING_OFFSET_PATH)
        == last_offset
    )
    assert harness.context.state.evicted_discovered_rules == {g.id: last_offset for g in evicted}


async def test_that_pruning_keeps_the_highest_scoring_rules_when_too_many_are_judged_stale(
    container: Container,
) -> None:
    members = _members(Matcher._DISCOVERED_RULES_HIGH_WATER_MARK + 1)
    names = _member_names(members)
    excess = len(members) - Matcher._DISCOVERED_RULES_TARGET
    # EVERY member is judged stale (score < 3) — but pruning must still land
    # at exactly the target, keeping the (relatively) highest-scoring ones.
    scores = {name: (1 if i < excess else 2) for i, name in enumerate(names)}
    harness = await _pruning_harness(container, members, scores_by_name=scores)

    await harness.matcher.prune_discovered_rules(harness.context)

    metadata = await harness.staged_metadata()
    assert set(metadata_utils.internal_value(metadata, _DISCOVERED_RULE_IDS_PATH, [])) == {
        str(g.id) for g in members[excess:]
    }


async def test_that_pruning_evicts_lowest_scoring_then_lowest_weight_rules_when_too_few_are_stale(
    container: Container,
) -> None:
    # All members are judged still-relevant with IDENTICAL scores; ties break
    # by weight (LOW evicted before MEDIUM).
    needed = Matcher._DISCOVERED_RULES_HIGH_WATER_MARK + 1 - Matcher._DISCOVERED_RULES_TARGET
    low_members = _members(needed, weight=Weight.LOW, prefix="LOWMEMBER")
    medium_members = _members(
        Matcher._DISCOVERED_RULES_HIGH_WATER_MARK + 1 - needed,
        weight=Weight.MEDIUM,
        prefix="MIDMEMBER",
    )
    members = low_members + medium_members
    scores = {name: 5 for name in _member_names(members)}
    harness = await _pruning_harness(container, members, scores_by_name=scores)

    await harness.matcher.prune_discovered_rules(harness.context)

    metadata = await harness.staged_metadata()
    assert set(metadata_utils.internal_value(metadata, _EVICTED_DISCOVERED_RULES_PATH, {})) == {
        str(g.id) for g in low_members
    }


async def test_that_high_weight_discovered_rules_are_never_evicted(
    container: Container,
) -> None:
    high_members = _members(
        Matcher._DISCOVERED_RULES_HIGH_WATER_MARK - 5, weight=Weight.HIGH, prefix="HIGHMEMBER"
    )
    medium_members = _members(6, weight=Weight.MEDIUM, prefix="MIDMEMBER")
    members = high_members + medium_members
    scores = {name: 1 for name in _member_names(members)}
    harness = await _pruning_harness(container, members, scores_by_name=scores)

    await harness.matcher.prune_discovered_rules(harness.context)

    # Only the non-exempt members were even judged at the model boundary...
    assert set(harness.rank_generator.calls) == set(_member_names(medium_members))

    # ...and only they were evicted: every HIGH member survives, even though
    # the set stays above the target.
    metadata = await harness.staged_metadata()
    surviving_ids = set(metadata_utils.internal_value(metadata, _DISCOVERED_RULE_IDS_PATH, []))
    assert {str(g.id) for g in high_members} <= surviving_ids


async def test_that_rules_matched_in_the_current_turn_are_never_evicted(
    container: Container,
) -> None:
    members = _members(Matcher._DISCOVERED_RULES_HIGH_WATER_MARK + 1)
    scores = {name: 1 for name in _member_names(members)}
    harness = await _pruning_harness(container, members, scores_by_name=scores)

    turn_matched = members[0]
    harness.context.state.ordinary_rule_matches = [
        RuleEvaluation(rule=turn_matched, reasoning="matched this turn")
    ]

    await harness.matcher.prune_discovered_rules(harness.context)

    assert "MEMBER_000" not in harness.rank_generator.calls
    metadata = await harness.staged_metadata()
    assert str(turn_matched.id) in metadata_utils.internal_value(
        metadata, _DISCOVERED_RULE_IDS_PATH, []
    )


async def test_that_pruning_failure_leaves_discovered_rules_and_metadata_unchanged(
    container: Container,
) -> None:
    # Fault injection at the pruner port: whatever blows up inside pruning,
    # the turn must survive with the state and session untouched.
    members = _members(Matcher._DISCOVERED_RULES_HIGH_WATER_MARK + 1)
    harness = await _pruning_harness(container, members, scores_by_name={})

    class _ExplodingPruner:
        async def prune(self, context: EngineContext, rules: Any) -> Any:
            raise RuntimeError("boom")

    harness.matcher._rule_pruner = cast(Any, _ExplodingPruner())

    await harness.matcher.prune_discovered_rules(harness.context)  # must not raise

    assert harness.context.state.discovered_rules == set(members)
    metadata = await harness.staged_metadata()
    assert metadata_utils.internal_value(metadata, _EVICTED_DISCOVERED_RULES_PATH) is None


async def test_that_pruning_is_rate_limited_to_once_per_ten_customer_messages(
    container: Container,
) -> None:
    members = _members(Matcher._DISCOVERED_RULES_HIGH_WATER_MARK + 1)
    # A pruning already ran at offset 0; only 9 customer messages arrived since.
    events = [
        create_event_message(offset=i, source=EventSource.CUSTOMER, message=f"message {i}")
        for i in range(10)
    ]
    harness = await _pruning_harness(
        container,
        members,
        scores_by_name={},
        conversation_events=events,
        extra_metadata={_LAST_DISCOVERED_RULE_PRUNING_OFFSET_PATH: 0},
    )

    await harness.matcher.prune_discovered_rules(harness.context)

    assert harness.rank_generator.calls == []
    assert harness.context.state.discovered_rules == set(members)


async def test_that_pruning_runs_again_after_ten_customer_messages(
    container: Container,
) -> None:
    members = _members(Matcher._DISCOVERED_RULES_HIGH_WATER_MARK + 1)
    events = [
        create_event_message(offset=i, source=EventSource.CUSTOMER, message=f"message {i}")
        for i in range(11)
    ]
    scores = {name: 1 for name in _member_names(members)}
    harness = await _pruning_harness(
        container,
        members,
        scores_by_name=scores,
        conversation_events=events,
        extra_metadata={_LAST_DISCOVERED_RULE_PRUNING_OFFSET_PATH: 0},
    )

    await harness.matcher.prune_discovered_rules(harness.context)

    assert harness.rank_generator.calls != []
    metadata = await harness.staged_metadata()
    assert metadata_utils.internal_value(metadata, _EVICTED_DISCOVERED_RULES_PATH) is not None


def test_that_low_weight_rules_render_with_a_suggestion_note_in_domain_instructions() -> None:
    # Low-weight rules render like any other instruction — no separate
    # "principles" section — but carry an inverse of the high-weight note:
    # treat as a suggestion, not a strict rule.
    low = replace(
        create_rule(condition="the customer makes smalltalk", action="keep answers brief"),
        weight=Weight.LOW,
    )

    prompt = PromptBuilder().add_system_wide_rules([low], {}).build()

    assert "keep answers brief" in prompt
    assert "low-weight instruction" in prompt


def test_that_matched_low_weight_rules_render_like_other_matches_with_a_suggestion_note() -> None:
    low = replace(
        create_rule(condition="the customer makes smalltalk", action="keep answers brief"),
        weight=Weight.LOW,
    )
    match = RuleEvaluation(rule=low, reasoning="Relevant.")

    prompt = PromptBuilder().add_matched_rules([match], {}, {low.id: low}).build()

    assert "keep answers brief" in prompt
    assert "low-weight instruction" in prompt


async def test_that_an_internal_rank_evaluation_method_escalates_to_rank(
    container: Container,
) -> None:
    # The internal evaluation_method entry is the explicit "this rule needs a
    # turn verdict" knob — the SDK stamps it for rule-attached retrievers.
    rule = replace(
        make_rule("RETRIEVING"),
        metadata=metadata_utils.set_internal_value({}, ("evaluation_method",), "rank"),
    )
    harness = await matching_harness(
        container,
        [rule],
        rank_relevant=set(),
        effort=Effort.MEDIUM,  # MEDIUM weight -> RECALL/fast path by the matrix
    )
    await harness.matcher._load_strategy_choice_signals(harness.context, [rule])

    assert harness.matcher._select_evaluator(harness.context, rule) is not None


async def test_that_a_distill_evaluation_method_also_ranks_in_the_core_engine(
    container: Container,
) -> None:
    # Core has no distiller; "distill" still means "evaluator tier" here and
    # degrades gracefully to ranking.
    rule = replace(
        make_rule("DISTILLING"),
        metadata=metadata_utils.set_internal_value({}, ("evaluation_method",), "distill"),
    )
    harness = await matching_harness(
        container,
        [rule],
        rank_relevant=set(),
        effort=Effort.MEDIUM,
    )
    await harness.matcher._load_strategy_choice_signals(harness.context, [rule])

    assert harness.matcher._select_evaluator(harness.context, rule) is not None


# ---------------------------------------------------------------------------
# The matched-rule ledger: dependencies are satisfied by "ever matched this
# session" (a verdict-level latch), NOT by discovery admission — otherwise
# dependency gating inherits embedding recall's false positives and
# depend_on() becomes decorative.
# ---------------------------------------------------------------------------


async def test_that_a_dependency_is_satisfied_by_a_previously_matched_target(
    container: Container,
) -> None:
    dependent = make_rule("DEPENDENT")
    target = make_rule("TARGET")
    harness = await matching_harness(
        container,
        [dependent, target],
        rank_relevant={"DEPENDENT"},
        session_metadata={_RULES_EVER_MATCHED_IN_SESSION_PATH: [str(target.id)]},
    )
    await harness.depend(dependent, target)

    await harness.matcher._load_discovered_rules(harness.context)
    await harness.match()

    matched = {evaluation.rule for evaluation in harness.context.state.ordinary_rule_matches}
    assert dependent in matched


async def test_that_discovery_admission_alone_does_not_satisfy_a_dependency(
    container: Container,
) -> None:
    # The target is already in the session working set (admitted by discovery
    # in some earlier turn) but has never actually matched. The dependent must
    # stay locked.
    dependent = make_rule("DEPENDENT")
    target = make_rule("TARGET")
    harness = await matching_harness(
        container,
        [dependent, target],
        rank_relevant={"DEPENDENT"},
        discovered_rules={target},
        session_metadata={_DISCOVERED_RULE_IDS_PATH: [str(target.id)]},
    )
    await harness.depend(dependent, target)

    await harness.match()

    matched = {evaluation.rule for evaluation in harness.context.state.ordinary_rule_matches}
    assert dependent not in matched


async def test_that_a_same_turn_match_satisfies_a_dependency(
    container: Container,
) -> None:
    dependent = make_rule("DEPENDENT")
    target = make_rule("TARGET")
    harness = await matching_harness(
        container,
        [dependent, target],
        rank_relevant={"DEPENDENT", "TARGET"},
    )
    await harness.depend(dependent, target)

    await harness.match()

    matched = {evaluation.rule for evaluation in harness.context.state.ordinary_rule_matches}
    assert dependent in matched
    assert target in matched


async def test_that_a_same_turn_discovery_admission_does_not_satisfy_a_dependency(
    container: Container,
) -> None:
    # The target is discovered THIS turn (its recall marker is in the
    # conversation) but the ranker judges it not relevant — an admission, not
    # a match. The dependent must stay locked.
    dependent = make_rule("DEPENDENT")
    target = make_rule("TARGET")
    harness = await matching_harness(
        container,
        [dependent, target],
        rank_relevant={"DEPENDENT"},
        conversation="RECALL_TARGET",
    )
    await harness.depend(dependent, target)

    await harness.match()

    matched = {evaluation.rule for evaluation in harness.context.state.ordinary_rule_matches}
    assert dependent not in matched


async def test_that_turn_matches_are_recorded_in_the_matched_rule_ledger(
    container: Container,
) -> None:
    matching = make_rule("MATCHING")
    admitted_only = make_rule("ADMITTED")
    harness = await matching_harness(
        container,
        [matching, admitted_only],
        rank_relevant={"MATCHING"},
        conversation="RECALL_ADMITTED",
    )

    await harness.match()

    assert await harness.staged_rules_ever_matched() == {str(matching.id)}


async def test_that_deprioritized_matches_are_recorded_in_the_matched_ledger(
    container: Container,
) -> None:
    # A match that loses priority resolution was still judged relevant — it
    # must enter the ledger (deprioritization never cascades into
    # dependencies, so it must keep satisfying dependents in later turns).
    winner = make_rule("WINNER")
    loser = make_rule("LOSER")
    harness = await matching_harness(
        container,
        [winner, loser],
        rank_relevant={"WINNER", "LOSER"},
    )
    await harness.depend(winner, loser, kind=RelationshipKind.PRIORITY)

    await harness.match()

    matched = {evaluation.rule for evaluation in harness.context.state.ordinary_rule_matches}
    assert loser not in matched
    assert await harness.staged_rules_ever_matched() == {str(winner.id), str(loser.id)}


# --- tool overlap collection ---


async def _overlap_seam_matcher(overlaps: list[tuple[ToolId, ToolId]]) -> Matcher:
    matcher = object.__new__(Matcher)
    matcher._logger = _FakeLogger()
    matcher._tool_selector = _RecordingToolSelector()
    relationship_store = await RelationshipDocumentStore(
        IdGenerator(), TransientDocumentDatabase()
    ).__aenter__()
    for source, target in overlaps:
        await relationship_store.create_relationship(
            source=RelationshipEntity(id=source, kind=RelationshipEntityKind.TOOL),
            target=RelationshipEntity(id=target, kind=RelationshipEntityKind.TOOL),
            kind=RelationshipKind.OVERLAP,
        )
    matcher._relationship_store = relationship_store
    return matcher


@pytest.mark.asyncio
async def test_that_overlap_relationships_among_available_tools_are_collected() -> None:
    search_all = ToolId("local", "search_all")
    search_narrow = ToolId("local", "search_narrow")
    unavailable = ToolId("local", "unavailable")
    matcher = await _overlap_seam_matcher(
        [
            (search_narrow, search_all),
            (search_all, unavailable),
        ]
    )
    context = _seam_context()
    context.state.available_tools = [
        replace(_tool("search_all"), overlap=ToolOverlap.AUTO),
        replace(_tool("search_narrow"), overlap=ToolOverlap.AUTO),
    ]
    context.state.tool_ids_by_name = {
        "search_all": search_all,
        "search_narrow": search_narrow,
    }
    context.state.fill_available_tool_ids = {search_all, search_narrow}

    await matcher._select_tools(context, log_delta_from_fill=True)

    assert context.state.overlapping_tool_ids == [(search_all, search_narrow)]


@pytest.mark.asyncio
async def test_that_an_always_overlap_tool_pairs_with_every_available_tool() -> None:
    broad = ToolId("local", "broad")
    a = ToolId("local", "a")
    b = ToolId("local", "b")
    matcher = await _overlap_seam_matcher([])
    context = _seam_context()
    context.state.available_tools = [
        replace(_tool("broad"), overlap=ToolOverlap.ALWAYS),
        replace(_tool("a"), overlap=ToolOverlap.AUTO),
        replace(_tool("b"), overlap=ToolOverlap.AUTO),
    ]
    context.state.tool_ids_by_name = {"broad": broad, "a": a, "b": b}
    context.state.fill_available_tool_ids = {broad, a, b}

    await matcher._select_tools(context, log_delta_from_fill=True)

    assert context.state.overlapping_tool_ids == [(a, broad), (b, broad)]


@pytest.mark.asyncio
async def test_that_a_none_overlap_tool_is_excluded_even_with_a_declared_relationship() -> None:
    solo = ToolId("local", "solo")
    other = ToolId("local", "other")
    matcher = await _overlap_seam_matcher([(solo, other)])
    context = _seam_context()
    context.state.available_tools = [
        _tool("solo"),  # overlap=NONE
        replace(_tool("other"), overlap=ToolOverlap.AUTO),
    ]
    context.state.tool_ids_by_name = {"solo": solo, "other": other}
    context.state.fill_available_tool_ids = {solo, other}

    await matcher._select_tools(context, log_delta_from_fill=True)

    assert context.state.overlapping_tool_ids == []


###############################################################################
# Authoritative rule decisions (Dashboard telemetry staging)
###############################################################################


async def test_that_matching_stages_selected_and_ruled_out_decisions(
    container: Container,
) -> None:
    chosen = make_rule("CHOSEN")
    rejected = make_rule("REJECTED")
    harness = await matching_harness(
        container,
        [chosen, rejected],
        rank_relevant={"CHOSEN"},
    )

    await harness.match()

    decisions = harness.context.state.matcher.pending_rule_decisions
    by_id = {decision.rule.id: decision for decision in decisions}

    assert by_id[chosen.id].selected is True
    assert by_id[chosen.id].rationale

    assert by_id[rejected.id].selected is False
    assert by_id[rejected.id].rationale


async def test_that_dependency_filtered_rules_are_staged_as_ruled_out(
    container: Container,
) -> None:
    dependent = make_rule("DEPENDENT")
    dependency = make_rule("DEPENDENCY")
    harness = await matching_harness(
        container,
        [dependent, dependency],
        # The dependency itself is judged irrelevant, so the dependent is
        # dropped by dependency filtering despite its own positive verdict.
        rank_relevant={"DEPENDENT"},
    )
    await harness.depend(dependent, dependency)

    await harness.match()

    decisions = harness.context.state.matcher.pending_rule_decisions
    by_id = {decision.rule.id: decision for decision in decisions}

    assert by_id[dependent.id].selected is False
    assert "dependen" in by_id[dependent.id].rationale.lower()


async def test_that_a_priority_demoted_rule_still_stages_a_selected_decision(
    container: Container,
) -> None:
    # The matched-rule ledger records verdict-level matches (pre-priority), and
    # decisions mirror exactly what persists: the outranked rule is still
    # "selected" even though the turn state demoted it.
    low = replace(make_rule("GENERAL"), priority=0)
    high = replace(make_rule("URGENT"), priority=10)
    harness = await matching_harness(
        container,
        [low, high],
        rank_relevant={"GENERAL", "URGENT"},
    )

    await harness.match()

    decisions = harness.context.state.matcher.pending_rule_decisions
    by_id = {decision.rule.id: decision for decision in decisions}

    assert by_id[low.id].selected is True
    assert by_id[high.id].selected is True


async def test_that_an_effort_lifting_match_emits_a_raise_effort_event(
    container: Container,
) -> None:
    lifting = replace(make_rule("LIFTER"), effort=Effort.MAX)
    harness = await matching_harness(
        container,
        [lifting],
        rank_relevant={"LIFTER"},
        effort=Effort.LOW,
    )

    recorded_events: list[tuple[str, dict[str, Any]]] = []

    class _EventRecordingTracer(LocalTracer):
        def add_event(self, name: str, attributes: Any = {}) -> None:
            recorded_events.append((name, dict(attributes)))

    harness.context.tracer = _EventRecordingTracer()

    await harness.match()

    raises_ = [a for n, a in recorded_events if n == "action.raise_effort"]
    assert len(raises_) == 1
    assert raises_[0]["from_effort"] == "low"
    assert raises_[0]["to_effort"] == "max"
    assert raises_[0]["rule_ids"] == [str(lifting.id)]
