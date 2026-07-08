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

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import asyncio
from typing import Iterator, Mapping, cast
from unittest.mock import AsyncMock
import pytest

from parlant.core.agents import Effort
from parlant.core.common import Weight
from parlant.core.engines.rule_match import RuleMatch
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.engines.compass.matching.rule_evaluation import (
    RuleEvaluation,
    RuleEvaluationResult,
)
from parlant.core.engines.compass.matching.rule_discovery import (
    DiscoveredRule,
    RuleDiscoveryResult,
)
from parlant.core.engines.compass.matching.rule_pruner import (
    PrunedRule,
    RulePruningResult,
)
from parlant.core.engines.compass.matcher import (
    Matcher,
    _ContextUsage,
    _EVICTED_SESSION_RULES_METADATA_KEY,
    _LAST_SESSION_PRUNING_OFFSET_METADATA_KEY,
    _SESSION_RULE_IDS_METADATA_KEY,
)
from parlant.core.engines.compass.response_state import EngineContext, ResponseState
from parlant.core.glossary import Term
from parlant.core.sessions import EventSource
from parlant.core.tools import Tool, ToolId, ToolOverlap
from parlant.core.tracer import AttributeValue, LocalTracer

from tests.core.stable.engines.compass.matching.utils import (
    create_engine_context,
    create_rule,
    create_term,
)


class _FakeEntityQueries:
    def __init__(self) -> None:
        self.glossary_inventory: list = []
        # keyword -> terms; a query containing the keyword "finds" those terms
        self.relevant_terms_by_keyword: dict = {}
        self.find_glossary_calls: list[tuple[str, int]] = []

    async def find_rule_tool_associations(self):
        return []

    async def list_glossary_terms_for_context(self, agent_id):
        return list(self.glossary_inventory)

    async def find_glossary_terms_for_context(self, agent_id, query, max_terms=20):
        self.find_glossary_calls.append((query, max_terms))
        hits = [
            term
            for keyword, terms in self.relevant_terms_by_keyword.items()
            if keyword.lower() in query.lower()
            for term in terms
        ]
        return hits[:max_terms]


class _FakeEntityCommands:
    def __init__(self) -> None:
        self.update_session = AsyncMock()
        self.upsert_session_labels = AsyncMock()


class _RecordingTracer(LocalTracer):
    def __init__(self) -> None:
        super().__init__()
        self.started_spans: list[str] = []
        self.events: list[tuple[str, Mapping[str, AttributeValue]]] = []

    @contextmanager
    def span(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> Iterator[None]:
        self.started_spans.append(span_id)
        with super().span(span_id, attributes):
            yield

    def add_event(
        self,
        name: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> None:
        self.events.append((name, dict(attributes)))


class _FakeRelationshipStore:
    async def list_relationships(self, *args, **kwargs):
        return []


class _FakeMatcherRegistry:
    def get(self, rule_id):
        return None


class _FakeLogger:
    def __init__(self) -> None:
        self.debug_messages: list[str] = []
        self.warning_messages: list[str] = []

    def debug(self, *args, **kwargs):
        self.debug_messages.append(str(args[0]))

    def warning(self, *args, **kwargs):
        self.warning_messages.append(str(args[0]))


def _make_warm_up_matcher(matcher_cls: type[Matcher] = Matcher) -> Matcher:
    matcher = object.__new__(matcher_cls)
    matcher._rule_evaluator = AsyncMock()
    matcher._matcher_registry = _FakeMatcherRegistry()
    matcher._relationship_store = _FakeRelationshipStore()
    matcher._entity_queries = _FakeEntityQueries()
    matcher._entity_commands = _FakeEntityCommands()
    return matcher


def _make_batch_matcher(matcher_cls: type[Matcher] = Matcher) -> Matcher:
    matcher = _make_warm_up_matcher(matcher_cls)
    matcher._logger = _FakeLogger()
    matcher._rule_function_matcher = AsyncMock()
    matcher._rule_function_matcher.match = AsyncMock(return_value=[])
    matcher._rule_discoverer = AsyncMock()
    matcher._rule_discoverer.discover = AsyncMock(return_value=RuleDiscoveryResult([], 0.0))
    matcher._rule_evaluator.evaluate = AsyncMock(return_value=RuleEvaluationResult([], None))
    return matcher


def _make_session_rules_matcher(
    entity_commands: _FakeEntityCommands | None = None,
) -> Matcher:
    matcher = object.__new__(Matcher)
    matcher._entity_queries = _FakeEntityQueries()
    matcher._entity_commands = entity_commands or _FakeEntityCommands()
    return matcher


def _context_with_rules(*rules, effort: Effort) -> EngineContext:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hello")])
    context.state = ResponseState(
        agent_effort=effort,
        usable_rules=list(rules),
        glossary_terms={create_term("known term", "already loaded")},
    )
    return context


def _with_recording_tracer(context: EngineContext) -> tuple[EngineContext, _RecordingTracer]:
    tracer = _RecordingTracer()
    return replace(context, tracer=tracer), tracer


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
async def test_that_preload_emits_span_and_loaded_rule_events() -> None:
    matcher = object.__new__(Matcher)
    rule = replace(
        create_rule(condition="customer asks for regulated help", action="follow the policy"),
        weight=Weight.HIGH,
        effort_lift=Effort.HIGH,
        title="Regulated help",
    )
    context, tracer = _with_recording_tracer(_context_with_rules(rule, effort=Effort.MEDIUM))
    matcher._variable_loader = AsyncMock()
    matcher._variable_loader.load = AsyncMock(return_value=[])
    matcher._entity_queries = AsyncMock()
    matcher._entity_queries.find_rules_for_context = AsyncMock(return_value=[rule])
    matcher._load_session_rules = AsyncMock()

    await matcher.preload(context)

    assert "match.preload" in tracer.started_spans
    assert context.state.usable_rules == [rule]
    assert (
        "loaded.rule",
        {
            "rule_id": str(rule.id),
            "title": "Regulated help",
            "last_modified": rule.modified_utc.isoformat(),
            "criticality": "high",
            "effort": "high",
        },
    ) in tracer.events


@pytest.mark.asyncio
async def test_that_fill_and_update_emit_spans() -> None:
    matcher = object.__new__(Matcher)
    context, tracer = _with_recording_tracer(_context_with_rules(effort=Effort.MEDIUM))
    matcher._match = AsyncMock()
    matcher._tool_recaller = AsyncMock()
    matcher._tool_recaller.prepare = AsyncMock()
    matcher._load_glossary = AsyncMock()
    matcher._select_tools = AsyncMock()
    matcher._reevaluate = AsyncMock()

    await matcher.fill(context)
    await matcher.update(context)

    assert "match.fill" in tracer.started_spans
    assert "match.update" in tracer.started_spans
    matcher._select_tools.assert_any_await(context, log_delta_from_fill=False)
    matcher._select_tools.assert_any_await(context, log_delta_from_fill=True)


@pytest.mark.asyncio
async def test_that_tool_recaller_prepare_runs_in_parallel_with_rule_matching() -> None:
    matcher = object.__new__(Matcher)
    context = _context_with_rules(effort=Effort.MEDIUM)
    match_started = asyncio.Event()
    release_match = asyncio.Event()
    tool_prepare_started = asyncio.Event()

    async def match(_context: EngineContext) -> None:
        match_started.set()
        await release_match.wait()

    async def prepare_tools(_context: EngineContext) -> None:
        tool_prepare_started.set()

    matcher._match = AsyncMock(side_effect=match)
    matcher._tool_recaller = AsyncMock()
    matcher._tool_recaller.prepare = AsyncMock(side_effect=prepare_tools)
    matcher._load_glossary = AsyncMock()
    matcher._select_tools = AsyncMock()

    fill_task = asyncio.create_task(matcher.fill(context))
    await match_started.wait()
    await asyncio.wait_for(tool_prepare_started.wait(), timeout=1.0)

    release_match.set()
    await fill_task

    matcher._tool_recaller.prepare.assert_awaited_once_with(context)
    matcher._select_tools.assert_awaited_once_with(context, log_delta_from_fill=False)


@pytest.mark.asyncio
async def test_that_select_tools_delegates_to_tool_recaller() -> None:
    matcher = object.__new__(Matcher)
    matcher._logger = _FakeLogger()
    matcher._tool_recaller = AsyncMock()
    matcher._tool_recaller.select = AsyncMock()
    context = _context_with_rules(effort=Effort.MEDIUM)

    await matcher._select_tools(context, log_delta_from_fill=False)

    matcher._tool_recaller.select.assert_awaited_once_with(context)


@pytest.mark.asyncio
async def test_that_select_tools_logs_no_delta_when_update_keeps_fill_catalog() -> None:
    logger = _FakeLogger()
    matcher = object.__new__(Matcher)
    matcher._logger = logger
    matcher._tool_recaller = AsyncMock()
    matcher._tool_recaller.select = AsyncMock()
    context = _context_with_rules(effort=Effort.MEDIUM)
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
    matcher._tool_recaller = AsyncMock()
    matcher._tool_recaller.select = AsyncMock()
    context = _context_with_rules(effort=Effort.MEDIUM)
    old_tool_id = ToolId("local", "old_lookup")
    new_tool_id = ToolId("local", "new_lookup")
    context.state.available_tools = [_tool("new_lookup", "New lookup tool.")]
    context.state.tool_ids_by_name = {"new_lookup": new_tool_id}
    context.state.tool_relevance_scores = {new_tool_id: 0.8}
    context.state.fill_available_tool_ids = {old_tool_id}

    await matcher._select_tools(context, log_delta_from_fill=True)

    assert len(logger.debug_messages) == 1
    assert "Matcher tool recall:" in logger.debug_messages[0]
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
    context = _context_with_rules(turn_rule, session_rule, effort=Effort.MEDIUM)
    context.state.available_tools = [
        _tool("complementary_tool"),
        _tool("session_tool"),
        _tool("turn_low"),
        _tool("turn_high"),
    ]
    context.state.matched_tools = [_tool("turn_low"), _tool("turn_high")]
    context.state.session_rules = {session_rule}
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

    log = logger.debug_messages[0]
    assert log.index("### 1 turn_high") < log.index("### 2 turn_low")
    assert log.index("### 2 turn_low") < log.index("### 3 session_tool")
    assert log.index("### 3 session_tool") < log.index("### 4 complementary_tool")


def test_that_description_only_highlighted_matches_are_rendered_as_instruction_reminders() -> None:
    rule = replace(
        create_rule(
            condition="booking a flight",
            action=None,
            description="Collect booking details in order.",
        ),
        criticality=Weight.HIGH,
        title="Book flight",
    )
    match = RuleMatch(
        rule=rule,
        rationale="Relevant.",
        metadata={"highlights": ["Ask the user for the trip type."]},
    )

    prompt = PromptBuilder().add_matched_rules([match], {}, {rule.id: rule}).build()

    assert '### Review the instructions under "Book flight"\n' in prompt
    assert "Highlights:\n- Ask the user for the trip type." in prompt
    assert "IMPORTANT: Please go back and reason" in prompt


def test_that_matcher_queries_start_with_session_summary_when_available() -> None:
    matcher = _make_session_rules_matcher()
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "current request")])
    context.state = ResponseState(session_summary="Earlier booking details were collected.")

    lines = matcher._build_interaction_query_lines(context)

    assert lines == [
        "Session summary: Earlier booking details were collected.",
        "EventSource.CUSTOMER: current request",
    ]
    assert matcher._build_tool_query(context).endswith(str(lines))


@pytest.mark.asyncio
async def test_that_session_rules_are_loaded_from_session_metadata() -> None:
    rule_1 = create_rule(condition="customer asks for help", action="ask what they need")
    rule_2 = create_rule(condition="customer asks for refund", action="explain refunds")
    context = _context_with_rules(rule_1, rule_2, effort=Effort.MEDIUM)
    context.session = replace(
        context.session,
        metadata={
            _SESSION_RULE_IDS_METADATA_KEY: [
                str(rule_2.id),
                "missing-rule",
                str(rule_2.id),
                str(rule_1.id),
                17,
            ]
        },
    )
    matcher = _make_session_rules_matcher()

    await matcher._load_session_rules(context)

    assert context.state.session_rules == {rule_2, rule_1}


@pytest.mark.asyncio
async def test_that_matched_rules_are_stored_in_session_metadata() -> None:
    rule_1 = create_rule(condition="customer asks for help", action="ask what they need")
    rule_2 = create_rule(condition="customer asks for refund", action="explain refunds")
    rule_3 = create_rule(condition="customer asks for billing", action="explain billing")
    entity_commands = _FakeEntityCommands()
    matcher = _make_session_rules_matcher(entity_commands)
    context = _context_with_rules(rule_1, rule_2, rule_3, effort=Effort.MEDIUM)
    context.state.session_rules = [rule_1]
    context.state.ordinary_rule_matches = [
        RuleMatch(rule=rule_2, rationale="relevant"),
        RuleMatch(rule=rule_1, rationale="still relevant"),
    ]
    matches = [
        (
            RuleMatch(rule=rule_2, rationale="relevant"),
            _ContextUsage.MATCH_CURRENT_TURN,
        ),
        (
            RuleMatch(rule=rule_1, rationale="still relevant"),
            _ContextUsage.MATCH_CURRENT_TURN,
        ),
    ]

    await matcher._store_session_rules(context, matches)

    assert context.state.session_rules == {rule_1, rule_2}
    entity_commands.update_session.assert_awaited_once()
    updated_session_id, params = entity_commands.update_session.await_args.args
    assert updated_session_id == context.session.id
    assert set(params["metadata"][_SESSION_RULE_IDS_METADATA_KEY]) == {
        str(rule_1.id),
        str(rule_2.id),
    }
    assert set(context.session.metadata[_SESSION_RULE_IDS_METADATA_KEY]) == {
        str(rule_1.id),
        str(rule_2.id),
    }


@pytest.mark.asyncio
async def test_that_storing_session_rules_preserves_existing_rules() -> None:
    rule_1 = create_rule(condition="customer asks for help", action="ask what they need")
    rule_2 = create_rule(condition="customer asks for refund", action="explain refunds")
    entity_commands = _FakeEntityCommands()
    matcher = _make_session_rules_matcher(entity_commands)
    context = _context_with_rules(rule_1, rule_2, effort=Effort.MEDIUM)
    context.state.session_rules = {rule_1}
    context.session = replace(
        context.session,
        metadata={_SESSION_RULE_IDS_METADATA_KEY: [str(rule_1.id)]},
    )
    matches = [
        (
            RuleMatch(rule=rule_2, rationale="newly relevant"),
            _ContextUsage.INCLUDE_IN_SESSION,
        )
    ]

    await matcher._store_session_rules(context, matches)

    assert context.state.session_rules == {rule_1, rule_2}
    entity_commands.update_session.assert_awaited_once()
    _, params = entity_commands.update_session.await_args.args
    assert set(params["metadata"][_SESSION_RULE_IDS_METADATA_KEY]) == {
        str(rule_1.id),
        str(rule_2.id),
    }


@pytest.mark.asyncio
async def test_that_storing_no_new_session_rules_does_not_clear_existing_rules() -> None:
    rule = create_rule(condition="customer asks for help", action="ask what they need")
    entity_commands = _FakeEntityCommands()
    matcher = _make_session_rules_matcher(entity_commands)
    context = _context_with_rules(rule, effort=Effort.MEDIUM)
    context.state.session_rules = {rule}
    context.session = replace(
        context.session,
        metadata={_SESSION_RULE_IDS_METADATA_KEY: [str(rule.id)]},
    )

    await matcher._store_session_rules(context, [])

    assert context.state.session_rules == {rule}
    entity_commands.update_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_that_session_only_matches_do_not_enter_turn_matched_rules() -> None:
    recalled_rule = create_rule(
        condition="customer asks for help",
        action="ask what they need",
    )
    turn_rule = create_rule(
        condition="customer asks for refund",
        action="explain refunds",
    )
    matcher = _make_session_rules_matcher()
    context = _context_with_rules(recalled_rule, turn_rule, effort=Effort.MEDIUM)
    matches = [
        (
            RuleMatch(rule=recalled_rule, rationale="recalled"),
            _ContextUsage.INCLUDE_IN_SESSION,
        ),
        (
            RuleMatch(rule=turn_rule, rationale="matched"),
            _ContextUsage.MATCH_CURRENT_TURN,
        ),
    ]

    await matcher._record(context, matches, append=False)

    assert context.state.ordinary_rule_matches == [RuleMatch(rule=turn_rule, rationale="matched")]
    assert context.state.tool_enabled_rule_matches == {}
    assert context.state.session_rules == {recalled_rule, turn_rule}


@pytest.mark.asyncio
async def test_that_record_emits_effort_raise_event_for_current_turn_rule() -> None:
    entity_commands = _FakeEntityCommands()
    matcher = _make_session_rules_matcher(entity_commands)
    rule = replace(
        create_rule(condition="customer asks for regulated help", action="follow the policy"),
        effort_lift=Effort.HIGH,
    )
    context, tracer = _with_recording_tracer(_context_with_rules(rule, effort=Effort.LOW))
    matches = [(RuleMatch(rule=rule, rationale="matched"), _ContextUsage.MATCH_CURRENT_TURN)]

    await matcher._record(context, matches, append=False)

    assert (
        "action.raise_effort",
        {
            "from_effort": "low",
            "to_effort": "high",
            "rule_ids": [str(rule.id)],
        },
    ) in tracer.events


@pytest.mark.asyncio
async def test_that_record_persists_new_session_labels_and_emits_event() -> None:
    entity_commands = _FakeEntityCommands()
    matcher = _make_session_rules_matcher(entity_commands)
    rule = replace(
        create_rule(condition="customer requests priority support", action="prioritize support"),
        labels={"existing", "priority"},
    )
    context, tracer = _with_recording_tracer(_context_with_rules(rule, effort=Effort.MEDIUM))
    context.session = replace(context.session, labels={"existing"})
    updated_session = replace(context.session, labels={"existing", "priority"})
    entity_commands.upsert_session_labels = AsyncMock(return_value=updated_session)
    matches = [(RuleMatch(rule=rule, rationale="matched"), _ContextUsage.MATCH_CURRENT_TURN)]

    await matcher._record(context, matches, append=False)

    entity_commands.upsert_session_labels.assert_awaited_once_with(
        context.session.id,
        {"priority"},
    )
    assert context.session == updated_session
    assert (
        "action.add_label",
        {
            "labels": ["priority"],
            "rule_ids": [str(rule.id)],
        },
    ) in tracer.events


@pytest.mark.asyncio
async def test_that_session_rules_are_not_stored_when_metadata_is_unchanged() -> None:
    rule = create_rule(condition="customer asks for help", action="ask what they need")
    entity_commands = _FakeEntityCommands()
    matcher = _make_session_rules_matcher(entity_commands)
    context = _context_with_rules(rule, effort=Effort.MEDIUM)
    context.state.session_rules = [rule]
    context.state.ordinary_rule_matches = [RuleMatch(rule=rule, rationale="")]
    context.session = replace(
        context.session,
        metadata={_SESSION_RULE_IDS_METADATA_KEY: [str(rule.id)]},
    )
    matches = [(RuleMatch(rule=rule, rationale=""), _ContextUsage.MATCH_CURRENT_TURN)]

    await matcher._store_session_rules(context, matches)

    entity_commands.update_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_that_warm_up_prefills_the_selected_evaluator() -> None:
    matcher = _make_warm_up_matcher()
    rule = create_rule(condition="customer asks for help", action="ask what they need")
    context = _context_with_rules(rule, effort=Effort.HIGH)

    await matcher.warm_up(context)

    matcher._rule_evaluator.warm_up.assert_awaited_once_with(context)


@pytest.mark.asyncio
async def test_that_warm_up_prefills_the_evaluator_for_complex_rules_too() -> None:
    matcher = _make_warm_up_matcher()
    rule = replace(
        create_rule(
            condition="customer asks for help",
            action="ask what they need",
            description="Follow this detailed process. " * 20,
        ),
        criticality=Weight.MEDIUM,
    )
    context = _context_with_rules(rule, effort=Effort.HIGH)

    await matcher.warm_up(context)

    matcher._rule_evaluator.warm_up.assert_awaited_once_with(context)


@pytest.mark.asyncio
async def test_that_warm_up_skips_the_evaluator_when_strategy_needs_none() -> None:
    matcher = _make_warm_up_matcher()
    rule = create_rule(condition="customer asks for help", action="ask what they need")
    context = _context_with_rules(rule, effort=Effort.LOW)

    await matcher.warm_up(context)

    matcher._rule_evaluator.warm_up.assert_not_awaited()


@pytest.mark.asyncio
async def test_that_warm_up_ranks_rule_that_can_raise_effort() -> None:
    matcher = _make_warm_up_matcher()
    rule = replace(
        create_rule(condition="customer asks for regulated help", action="follow the policy"),
        effort=Effort.HIGH,
    )
    context = _context_with_rules(rule, effort=Effort.LOW)

    await matcher.warm_up(context)

    matcher._rule_evaluator.warm_up.assert_awaited_once_with(context)


@pytest.mark.asyncio
async def test_that_low_criticality_rule_that_can_raise_effort_is_matched_this_turn() -> None:
    matcher = _make_batch_matcher()
    rule = replace(
        create_rule(condition="customer asks for regulated help", action="follow the policy"),
        criticality=Weight.LOW,
        effort=Effort.HIGH,
    )
    matcher._rule_evaluator.evaluate = AsyncMock(
        return_value=RuleEvaluationResult(
            [
                RuleEvaluation(
                    rule=rule,
                    reasoning="Relevant.",
                    is_relevant=True,
                    score=1.0,
                )
            ],
            None,
        )
    )
    context = _context_with_rules(rule, effort=Effort.LOW)

    matches = await matcher._run_batches(context, [rule])

    matcher._rule_discoverer.discover.assert_awaited_once_with(context, [rule])
    matcher._rule_evaluator.evaluate.assert_awaited_once_with(context, [rule])
    assert matches == [
        (
            RuleMatch(rule=rule, rationale="Relevant."),
            _ContextUsage.MATCH_CURRENT_TURN,
        )
    ]


@pytest.mark.asyncio
async def test_that_ranked_rule_can_still_be_discovered_into_session_by_recall() -> None:
    matcher = _make_batch_matcher()
    rule = replace(
        create_rule(condition="customer asks for regulated help", action="follow the policy"),
        criticality=Weight.HIGH,
    )
    matcher._rule_discoverer.discover = AsyncMock(
        return_value=RuleDiscoveryResult(
            [DiscoveredRule(rule=rule, is_relevant=True, score=0.7)],
            0.0,
        )
    )
    matcher._rule_evaluator.evaluate = AsyncMock(
        return_value=RuleEvaluationResult(
            [
                RuleEvaluation(
                    rule=rule,
                    reasoning="Not currently relevant.",
                    is_relevant=False,
                    score=0.2,
                )
            ],
            None,
        )
    )
    context = _context_with_rules(rule, effort=Effort.MEDIUM)

    matches = await matcher._run_batches(context, [rule])

    matcher._rule_discoverer.discover.assert_awaited_once_with(context, [rule])
    matcher._rule_evaluator.evaluate.assert_awaited_once_with(context, [rule])
    assert matches == [
        (
            RuleMatch(
                rule=rule,
                rationale="This may or may not be relevant right now - use your judgment.",
            ),
            _ContextUsage.INCLUDE_IN_SESSION,
        )
    ]


@pytest.mark.asyncio
async def test_that_an_evaluated_rule_can_still_be_discovered_into_session_by_recall() -> None:
    matcher = _make_batch_matcher()
    rule = replace(
        create_rule(
            condition="customer asks for a regulated workflow",
            action="follow the detailed workflow exactly",
            description="This workflow has many details. " * 20,
        ),
        criticality=Weight.HIGH,
    )
    matcher._rule_discoverer.discover = AsyncMock(
        return_value=RuleDiscoveryResult(
            [DiscoveredRule(rule=rule, is_relevant=True, score=0.7)],
            0.0,
        )
    )
    matcher._rule_evaluator.evaluate = AsyncMock(
        return_value=RuleEvaluationResult(
            [
                RuleEvaluation(
                    rule=rule,
                    reasoning="No currently-relevant points remain.",
                    is_relevant=False,
                    highlights=[],
                )
            ],
            None,
        )
    )
    context = _context_with_rules(rule, effort=Effort.MEDIUM)

    matches = await matcher._run_batches(context, [rule])

    matcher._rule_discoverer.discover.assert_awaited_once_with(context, [rule])
    matcher._rule_evaluator.evaluate.assert_awaited_once_with(context, [rule])
    assert matches == [
        (
            RuleMatch(
                rule=rule,
                rationale="This may or may not be relevant right now - use your judgment.",
            ),
            _ContextUsage.INCLUDE_IN_SESSION,
        )
    ]


@pytest.mark.asyncio
async def test_that_turn_match_takes_precedence_over_session_recall_discovery() -> None:
    matcher = _make_batch_matcher()
    rule = replace(
        create_rule(condition="customer asks for regulated help", action="follow the policy"),
        criticality=Weight.HIGH,
    )
    matcher._rule_discoverer.discover = AsyncMock(
        return_value=RuleDiscoveryResult(
            [DiscoveredRule(rule=rule, is_relevant=True, score=0.7)],
            0.0,
        )
    )
    matcher._rule_evaluator.evaluate = AsyncMock(
        return_value=RuleEvaluationResult(
            [
                RuleEvaluation(
                    rule=rule,
                    reasoning="Relevant.",
                    is_relevant=True,
                    score=1.0,
                )
            ],
            None,
        )
    )
    context = _context_with_rules(rule, effort=Effort.MEDIUM)

    matches = await matcher._run_batches(context, [rule])

    assert matches == [
        (
            RuleMatch(rule=rule, rationale="Relevant."),
            _ContextUsage.MATCH_CURRENT_TURN,
        )
    ]


@pytest.mark.asyncio
async def test_that_an_evaluator_turn_match_takes_precedence_over_session_recall_discovery() -> (
    None
):
    matcher = _make_batch_matcher()
    rule = replace(
        create_rule(
            condition="customer asks for a regulated workflow",
            action="follow the detailed workflow exactly",
            description="This workflow has many details. " * 20,
        ),
        criticality=Weight.HIGH,
    )
    matcher._rule_discoverer.discover = AsyncMock(
        return_value=RuleDiscoveryResult(
            [DiscoveredRule(rule=rule, is_relevant=True, score=0.7)],
            0.0,
        )
    )
    matcher._rule_evaluator.evaluate = AsyncMock(
        return_value=RuleEvaluationResult(
            [
                RuleEvaluation(
                    rule=rule,
                    reasoning="Relevant.",
                    is_relevant=True,
                    highlights=["Follow the detailed workflow exactly."],
                )
            ],
            None,
        )
    )
    context = _context_with_rules(rule, effort=Effort.MEDIUM)

    matches = await matcher._run_batches(context, [rule])

    matcher._rule_discoverer.discover.assert_awaited_once_with(context, [rule])
    matcher._rule_evaluator.evaluate.assert_awaited_once_with(context, [rule])
    assert matches == [
        (
            RuleMatch(
                rule=rule,
                rationale="Relevant.",
                metadata={"highlights": ["Follow the detailed workflow exactly."]},
            ),
            _ContextUsage.MATCH_CURRENT_TURN,
        )
    ]


# --- Session rule pruning (cap + eviction) -------------------------------


def _make_pruning_matcher(
    entity_commands: _FakeEntityCommands | None = None,
) -> Matcher:
    matcher = object.__new__(Matcher)
    matcher._logger = _FakeLogger()
    matcher._entity_queries = _FakeEntityQueries()
    matcher._entity_commands = entity_commands or _FakeEntityCommands()
    matcher._rule_pruner = AsyncMock()
    return matcher


def _session_members(count: int, criticality: Weight = Weight.MEDIUM) -> list:
    return [
        replace(
            create_rule(condition=f"condition {i:03d}", action=f"action {i:03d}"),
            criticality=criticality,
        )
        for i in range(count)
    ]


def _pruning_context(
    members: list,
    *,
    conversation: list[tuple[EventSource, str]] | None = None,
    extra_metadata: dict[str, object] | None = None,
) -> EngineContext:
    context = create_engine_context(
        conversation=conversation
        if conversation is not None
        else [
            (EventSource.CUSTOMER, "hello"),
            (EventSource.AI_AGENT, "hi, how can I help?"),
            (EventSource.CUSTOMER, "I have a question"),
        ]
    )
    metadata: dict[str, object] = {
        _SESSION_RULE_IDS_METADATA_KEY: [str(g.id) for g in members],
        **(extra_metadata or {}),
    }
    context.session = replace(context.session, metadata=metadata)
    context.state = ResponseState(
        usable_rules=list(members),
        session_rules=set(members),
    )
    return context


def _pruning_result(members: list, scores: dict[str, float]) -> RulePruningResult:
    return RulePruningResult(
        pruned_rules=[
            PrunedRule(
                rule=g,
                reasoning="judged",
                is_still_relevant=scores[str(g.id)] >= 0.6,
                score=scores[str(g.id)],
            )
            for g in members
            if str(g.id) in scores
        ],
        generation_info=None,
    )


@pytest.mark.asyncio
async def test_that_session_rule_pruning_is_skipped_at_or_below_the_high_water_mark() -> None:
    matcher = _make_pruning_matcher()
    members = _session_members(Matcher._SESSION_RULES_HIGH_WATER_MARK)
    context = _pruning_context(members)

    await matcher.prune_session_rules(context)

    matcher._rule_pruner.prune.assert_not_awaited()
    matcher._entity_commands.update_session.assert_not_awaited()
    assert context.state.session_rules == set(members)


@pytest.mark.asyncio
async def test_that_session_rule_pruning_evicts_stale_rules_down_to_the_target() -> None:
    entity_commands = _FakeEntityCommands()
    matcher = _make_pruning_matcher(entity_commands)
    members = _session_members(Matcher._SESSION_RULES_HIGH_WATER_MARK + 1)
    context = _pruning_context(members)

    # Deterministic, distinct scores: earlier members are staler.
    scores = {str(g.id): i / 100.0 for i, g in enumerate(members)}
    matcher._rule_pruner.prune = AsyncMock(return_value=_pruning_result(members, scores))

    await matcher.prune_session_rules(context)

    entity_commands.update_session.assert_awaited_once()
    metadata = entity_commands.update_session.await_args.args[1]["metadata"]

    surviving_ids = set(metadata[_SESSION_RULE_IDS_METADATA_KEY])
    assert len(surviving_ids) == Matcher._SESSION_RULES_TARGET
    assert len(context.state.session_rules) == Matcher._SESSION_RULES_TARGET

    # Evictions are ledgered at the interaction's last event offset (2), and the
    # pruning itself is stamped for the rate limiter.
    last_offset = max(e.offset for e in context.interaction.events)
    ledger = metadata[_EVICTED_SESSION_RULES_METADATA_KEY]
    evicted_count = len(members) - Matcher._SESSION_RULES_TARGET
    assert len(ledger) == evicted_count
    assert all(offset == last_offset for offset in ledger.values())
    assert metadata[_LAST_SESSION_PRUNING_OFFSET_METADATA_KEY] == last_offset
    assert context.state.evicted_session_rules == {
        g.id: last_offset for g in members if str(g.id) in ledger
    }


@pytest.mark.asyncio
async def test_that_pruning_keeps_the_highest_scoring_rules_when_too_many_are_judged_stale() -> (
    None
):
    entity_commands = _FakeEntityCommands()
    matcher = _make_pruning_matcher(entity_commands)
    members = _session_members(Matcher._SESSION_RULES_HIGH_WATER_MARK + 1)
    context = _pruning_context(members)

    # EVERY member is judged confidently stale — but pruning must still land at
    # exactly the target, keeping the (relatively) highest-scoring ones.
    scores = {str(g.id): 0.2 - i / 1000.0 for i, g in enumerate(members)}
    matcher._rule_pruner.prune = AsyncMock(return_value=_pruning_result(members, scores))

    await matcher.prune_session_rules(context)

    metadata = entity_commands.update_session.await_args.args[1]["metadata"]
    surviving_ids = set(metadata[_SESSION_RULE_IDS_METADATA_KEY])

    expected_survivors = {
        str(g.id)
        for g in sorted(members, key=lambda g: scores[str(g.id)], reverse=True)[
            : Matcher._SESSION_RULES_TARGET
        ]
    }
    assert surviving_ids == expected_survivors


@pytest.mark.asyncio
async def test_that_pruning_evicts_lowest_scoring_then_lowest_criticality_rules_when_too_few_are_stale() -> (
    None
):
    entity_commands = _FakeEntityCommands()
    matcher = _make_pruning_matcher(entity_commands)
    # All members are judged still-relevant with IDENTICAL scores; ties break by
    # criticality (LOW evicted before MEDIUM).
    needed = Matcher._SESSION_RULES_HIGH_WATER_MARK + 1 - Matcher._SESSION_RULES_TARGET
    low_members = _session_members(needed, criticality=Weight.LOW)
    medium_members = _session_members(
        Matcher._SESSION_RULES_HIGH_WATER_MARK + 1 - needed,
        criticality=Weight.MEDIUM,
    )
    members = low_members + medium_members
    context = _pruning_context(members)

    scores = {str(g.id): 1.0 for g in members}
    matcher._rule_pruner.prune = AsyncMock(return_value=_pruning_result(members, scores))

    await matcher.prune_session_rules(context)

    metadata = entity_commands.update_session.await_args.args[1]["metadata"]
    evicted_ids = set(metadata[_EVICTED_SESSION_RULES_METADATA_KEY])

    assert evicted_ids == {str(g.id) for g in low_members}


@pytest.mark.asyncio
async def test_that_high_criticality_session_rules_are_never_evicted() -> None:
    entity_commands = _FakeEntityCommands()
    matcher = _make_pruning_matcher(entity_commands)
    high_members = _session_members(
        Matcher._SESSION_RULES_HIGH_WATER_MARK - 5, criticality=Weight.HIGH
    )
    medium_members = _session_members(6, criticality=Weight.MEDIUM)
    members = high_members + medium_members
    context = _pruning_context(members)

    scores = {str(g.id): 0.0 for g in members}
    matcher._rule_pruner.prune = AsyncMock(return_value=_pruning_result(members, scores))

    await matcher.prune_session_rules(context)

    # Only the non-exempt members were even evaluated...
    pruned = matcher._rule_pruner.prune.await_args.args[1]
    assert set(pruned) == set(medium_members)

    # ...and only they were evicted: every HIGH member survives, even though the
    # set stays above the target.
    metadata = entity_commands.update_session.await_args.args[1]["metadata"]
    surviving_ids = set(metadata[_SESSION_RULE_IDS_METADATA_KEY])
    assert {str(g.id) for g in high_members} <= surviving_ids


@pytest.mark.asyncio
async def test_that_rules_matched_in_the_current_turn_are_never_evicted() -> None:
    entity_commands = _FakeEntityCommands()
    matcher = _make_pruning_matcher(entity_commands)
    members = _session_members(Matcher._SESSION_RULES_HIGH_WATER_MARK + 1)
    context = _pruning_context(members)

    turn_matched = members[0]
    context.state.ordinary_rule_matches = [
        RuleMatch(rule=turn_matched, rationale="matched this turn")
    ]

    scores = {str(g.id): 0.0 for g in members}
    matcher._rule_pruner.prune = AsyncMock(return_value=_pruning_result(members, scores))

    await matcher.prune_session_rules(context)

    pruned = matcher._rule_pruner.prune.await_args.args[1]
    assert turn_matched not in pruned

    metadata = entity_commands.update_session.await_args.args[1]["metadata"]
    assert str(turn_matched.id) in metadata[_SESSION_RULE_IDS_METADATA_KEY]


@pytest.mark.asyncio
async def test_that_pruning_failure_leaves_session_rules_and_metadata_unchanged() -> None:
    entity_commands = _FakeEntityCommands()
    matcher = _make_pruning_matcher(entity_commands)
    members = _session_members(Matcher._SESSION_RULES_HIGH_WATER_MARK + 1)
    context = _pruning_context(members)

    matcher._rule_pruner.prune = AsyncMock(side_effect=RuntimeError("boom"))

    await matcher.prune_session_rules(context)  # must not raise

    entity_commands.update_session.assert_not_awaited()
    assert context.state.session_rules == set(members)


@pytest.mark.asyncio
async def test_that_pruning_is_rate_limited_to_once_per_ten_customer_messages() -> None:
    matcher = _make_pruning_matcher()
    members = _session_members(Matcher._SESSION_RULES_HIGH_WATER_MARK + 1)
    # A pruning already ran at offset 0; only 9 customer messages arrived since.
    conversation = [(EventSource.CUSTOMER, "hello")] + [
        (EventSource.CUSTOMER, f"message {i}") for i in range(9)
    ]
    context = _pruning_context(
        members,
        conversation=conversation,
        extra_metadata={_LAST_SESSION_PRUNING_OFFSET_METADATA_KEY: 0},
    )

    await matcher.prune_session_rules(context)

    matcher._rule_pruner.prune.assert_not_awaited()
    matcher._entity_commands.update_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_that_pruning_runs_again_after_ten_customer_messages() -> None:
    entity_commands = _FakeEntityCommands()
    matcher = _make_pruning_matcher(entity_commands)
    members = _session_members(Matcher._SESSION_RULES_HIGH_WATER_MARK + 1)
    conversation = [(EventSource.CUSTOMER, "hello")] + [
        (EventSource.CUSTOMER, f"message {i}") for i in range(10)
    ]
    context = _pruning_context(
        members,
        conversation=conversation,
        extra_metadata={_LAST_SESSION_PRUNING_OFFSET_METADATA_KEY: 0},
    )

    scores = {str(g.id): 0.0 for g in members}
    matcher._rule_pruner.prune = AsyncMock(return_value=_pruning_result(members, scores))

    await matcher.prune_session_rules(context)

    matcher._rule_pruner.prune.assert_awaited_once()
    entity_commands.update_session.assert_awaited_once()


# --- Eviction ledger: load, readmission, and recall wiring ---------------------


@pytest.mark.asyncio
async def test_that_sessions_without_an_eviction_ledger_load_with_an_empty_ledger() -> None:
    matcher = _make_batch_matcher()
    rule = create_rule(condition="customer asks for help", action="help them")
    context = _context_with_rules(rule, effort=Effort.MEDIUM)
    context.session = replace(
        context.session,
        metadata={_SESSION_RULE_IDS_METADATA_KEY: [str(rule.id)]},
    )

    await matcher._load_session_rules(context)

    assert context.state.session_rules == {rule}
    assert context.state.evicted_session_rules == {}


@pytest.mark.asyncio
async def test_that_the_eviction_ledger_is_loaded_from_session_metadata() -> None:
    matcher = _make_batch_matcher()
    member = create_rule(condition="customer asks for help", action="help them")
    evicted = create_rule(condition="customer wants a refund", action="refund them")
    context = _context_with_rules(member, evicted, effort=Effort.MEDIUM)
    context.session = replace(
        context.session,
        metadata={
            _SESSION_RULE_IDS_METADATA_KEY: [str(member.id)],
            _EVICTED_SESSION_RULES_METADATA_KEY: {str(evicted.id): 7},
        },
    )

    await matcher._load_session_rules(context)

    assert context.state.session_rules == {member}
    assert context.state.evicted_session_rules == {evicted.id: 7}


@pytest.mark.asyncio
async def test_that_the_discoverer_sees_the_eviction_ledger_for_ledgered_rules() -> None:
    matcher = _make_batch_matcher()
    rule = create_rule(condition="customer wants a refund", action="refund them")
    context = _context_with_rules(rule, effort=Effort.MEDIUM)
    context.state.evicted_session_rules = {rule.id: 5}

    matcher._rule_discoverer.discover = AsyncMock(return_value=RuleDiscoveryResult([], 0.0))

    await matcher._run_batches(context, [rule])

    matcher._rule_discoverer.discover.assert_awaited_once()
    # The ledger travels via the context state; the discoverer derives its own floors.
    awaited_context = matcher._rule_discoverer.discover.await_args.args[0]
    assert awaited_context.state.evicted_session_rules == {rule.id: 5}


@pytest.mark.asyncio
async def test_that_a_readmitted_rule_is_removed_from_the_eviction_ledger() -> None:
    entity_commands = _FakeEntityCommands()
    matcher = _make_batch_matcher()
    matcher._entity_commands = entity_commands
    rule = create_rule(condition="customer wants a refund", action="refund them")
    context = _context_with_rules(rule, effort=Effort.MEDIUM)
    context.session = replace(
        context.session,
        metadata={
            _SESSION_RULE_IDS_METADATA_KEY: [],
            _EVICTED_SESSION_RULES_METADATA_KEY: {str(rule.id): 5},
        },
    )
    context.state.evicted_session_rules = {rule.id: 5}

    await matcher._store_session_rules(
        context,
        [
            (
                RuleMatch(rule=rule, rationale="fresh trigger"),
                _ContextUsage.MATCH_CURRENT_TURN,
            )
        ],
    )

    entity_commands.update_session.assert_awaited_once()
    metadata = entity_commands.update_session.await_args.args[1]["metadata"]
    assert str(rule.id) in metadata[_SESSION_RULE_IDS_METADATA_KEY]
    assert str(rule.id) not in metadata[_EVICTED_SESSION_RULES_METADATA_KEY]
    assert rule.id not in context.state.evicted_session_rules


# --- Glossary loading delegation -------------------------------------------------


@pytest.mark.asyncio
async def test_that_the_matcher_delegates_glossary_loading_to_the_recaller() -> None:
    matcher = object.__new__(Matcher)
    matcher._glossary_recaller = AsyncMock()
    context = _context_with_rules(effort=Effort.MEDIUM)

    await matcher._load_glossary(context)

    matcher._glossary_recaller.recall.assert_awaited_once_with(context)


@pytest.mark.asyncio
async def test_that_the_matcher_delegates_glossary_pruning_to_the_recaller() -> None:
    matcher = object.__new__(Matcher)
    matcher._glossary_recaller = AsyncMock()
    context = _context_with_rules(effort=Effort.MEDIUM)

    await matcher.prune_session_glossary(context)

    matcher._glossary_recaller.prune.assert_awaited_once_with(context)


# --- Prompt rendering determinism ----------------------------------------------


def test_that_glossary_terms_are_rendered_in_deterministic_order() -> None:
    # Callers pass set-derived lists; rendering must not depend on iteration
    # order, or the cached prefixes it appears in churn across restarts.
    terms = [create_term(f"term {i}", f"description {i}") for i in range(5)]

    forward = PromptBuilder().add_glossary(terms).build()
    backward = PromptBuilder().add_glossary(terms[::-1]).build()

    assert forward == backward


def test_that_low_criticality_rule_instructions_are_rendered_in_deterministic_order() -> None:
    # The session rule set is a Python set; rendering must not depend on its
    # iteration order, or the responder's cached system prefix churns across
    # processes/restarts.
    rules = [
        replace(
            create_rule(condition=f"condition {i}", action=f"action {i}"),
            criticality=Weight.LOW,
        )
        for i in range(5)
    ]

    forward = PromptBuilder().add_low_criticality_rule_instructions(rules).build()
    backward = PromptBuilder().add_low_criticality_rule_instructions(rules[::-1]).build()

    assert forward == backward


# --- Per-rule glossary terms (interpretation dependencies) -----------------
#
# Turn evaluators judge one rule per prompt; a rule may depend on
# glossary terms to be interpreted correctly, so the matcher resolves each
# evaluated rule's relevant terms (top-k by the rule's own query)
# into state.terms_by_rule before the batches run.


def _glossary_term(name: str, description: str) -> Term:
    from parlant.core.common import generate_id
    from parlant.core.glossary import TermId

    now = datetime.now(timezone.utc)
    return Term(
        id=TermId(generate_id()),
        creation_utc=now,
        modified_utc=now,
        name=name,
        description=description,
        synonyms=[],
        groups=[],
    )


@pytest.mark.asyncio
async def test_that_rule_terms_are_loaded_for_evaluated_rules() -> None:
    matcher = _make_batch_matcher()
    rule = create_rule(condition="the customer reports PRS", action="escalate to a specialist")
    context = _context_with_rules(rule, effort=Effort.HIGH)  # MEDIUM crit -> RANK

    term = _glossary_term("PRS", "Pinewood Rash Syndrome - an allergy to pinewood.")
    matcher._entity_queries.glossary_inventory = [term]
    matcher._entity_queries.relevant_terms_by_keyword = {"PRS": [term]}

    await matcher._run_batches(context, [rule])

    assert context.state.terms_by_rule.get(rule.id) == [term]
    assert len(matcher._entity_queries.find_glossary_calls) == 1
    query, k = matcher._entity_queries.find_glossary_calls[0]
    assert "the customer reports PRS" in query
    assert k == Matcher._TERMS_PER_RULE


@pytest.mark.asyncio
async def test_that_recall_strategy_rules_do_not_fetch_terms() -> None:
    matcher = _make_batch_matcher()
    rule = create_rule(condition="the customer reports PRS", action="escalate")
    context = _context_with_rules(rule, effort=Effort.MEDIUM)  # MEDIUM crit -> RECALL

    term = _glossary_term("PRS", "Pinewood Rash Syndrome.")
    matcher._entity_queries.glossary_inventory = [term]
    matcher._entity_queries.relevant_terms_by_keyword = {"PRS": [term]}

    await matcher._run_batches(context, [rule])

    assert matcher._entity_queries.find_glossary_calls == []


@pytest.mark.asyncio
async def test_that_rule_term_lookups_are_memoized_across_turns() -> None:
    matcher = _make_batch_matcher()
    rule = create_rule(condition="the customer reports PRS", action="escalate to a specialist")
    context = _context_with_rules(rule, effort=Effort.HIGH)

    term = _glossary_term("PRS", "Pinewood Rash Syndrome.")
    matcher._entity_queries.glossary_inventory = [term]
    matcher._entity_queries.relevant_terms_by_keyword = {"PRS": [term]}

    await matcher._run_batches(context, [rule])
    await matcher._run_batches(context, [rule])

    # The embedding lookup ran once; the second turn was served from the memo —
    # but the state still carries the terms.
    assert len(matcher._entity_queries.find_glossary_calls) == 1
    assert context.state.terms_by_rule.get(rule.id) == [term]


@pytest.mark.asyncio
async def test_that_editing_a_rule_invalidates_its_term_lookup() -> None:
    from datetime import timedelta

    matcher = _make_batch_matcher()
    rule = create_rule(condition="the customer reports PRS", action="escalate to a specialist")
    context = _context_with_rules(rule, effort=Effort.HIGH)

    term = _glossary_term("PRS", "Pinewood Rash Syndrome.")
    matcher._entity_queries.glossary_inventory = [term]
    matcher._entity_queries.relevant_terms_by_keyword = {"PRS": [term]}

    await matcher._run_batches(context, [rule])

    edited = replace(rule, modified_utc=rule.modified_utc + timedelta(seconds=1))
    context.state.usable_rules = [edited]
    await matcher._run_batches(context, [edited])

    assert len(matcher._entity_queries.find_glossary_calls) == 2


@pytest.mark.asyncio
async def test_that_a_glossary_change_invalidates_rule_term_lookups() -> None:
    matcher = _make_batch_matcher()
    rule = create_rule(condition="the customer reports PRS", action="escalate to a specialist")
    context = _context_with_rules(rule, effort=Effort.HIGH)

    term = _glossary_term("PRS", "Pinewood Rash Syndrome.")
    matcher._entity_queries.glossary_inventory = [term]
    matcher._entity_queries.relevant_terms_by_keyword = {"PRS": [term]}

    await matcher._run_batches(context, [rule])

    # A new term joins the glossary at runtime: the mapping must be recomputed.
    matcher._entity_queries.glossary_inventory = [
        term,
        _glossary_term("RMA-3", "The escalation procedure for medical cases."),
    ]
    await matcher._run_batches(context, [rule])

    assert len(matcher._entity_queries.find_glossary_calls) == 2


@pytest.mark.asyncio
async def test_that_an_empty_glossary_skips_rule_term_lookups() -> None:
    matcher = _make_batch_matcher()
    rule = create_rule(condition="the customer reports PRS", action="escalate")
    context = _context_with_rules(rule, effort=Effort.HIGH)

    await matcher._run_batches(context, [rule])

    assert matcher._entity_queries.find_glossary_calls == []
    assert context.state.terms_by_rule == {}


# --- The evaluator-selection seam ------------------------------------------------
#
# The matcher selects a turn evaluator per rule (None = discovery-only) and
# groups rules by evaluator instance — natively multi-evaluator, so a subclass
# can route specific rules to its own evaluator by overriding _select_evaluator.


class _RoutingMatcher(Matcher):
    """A test subclass routing rules whose condition mentions 'special' to a
    second evaluator — the extension seam a plug-in matcher uses."""

    special_evaluator: AsyncMock

    def _select_evaluator(self, context, rule):  # type: ignore[no-untyped-def]
        if "special" in rule.content.condition:
            return self.special_evaluator
        return super()._select_evaluator(context, rule)


@pytest.mark.asyncio
async def test_that_select_evaluator_returns_the_bound_evaluator_for_rank_tier_rules() -> None:
    matcher = _make_batch_matcher()
    rank_tier = replace(create_rule(condition="high weight", action="act"), criticality=Weight.HIGH)
    recall_tier = create_rule(condition="medium weight", action="act")  # Weight.MEDIUM
    context = _context_with_rules(rank_tier, recall_tier, effort=Effort.MEDIUM)
    await matcher._load_strategy_choice_signals(context, [rank_tier, recall_tier])

    assert matcher._select_evaluator(context, rank_tier) is matcher._rule_evaluator
    assert matcher._select_evaluator(context, recall_tier) is None


@pytest.mark.asyncio
async def test_that_rules_are_grouped_by_selected_evaluator() -> None:
    matcher = cast(_RoutingMatcher, _make_batch_matcher(_RoutingMatcher))
    matcher.special_evaluator = AsyncMock()
    matcher.special_evaluator.evaluate = AsyncMock(return_value=RuleEvaluationResult([], None))

    special_1 = replace(
        create_rule(condition="special topic one", action="act"), criticality=Weight.HIGH
    )
    special_2 = replace(
        create_rule(condition="special topic two", action="act"), criticality=Weight.HIGH
    )
    normal = replace(create_rule(condition="a normal rule", action="act"), criticality=Weight.HIGH)
    discovery_only = create_rule(condition="recall tier", action="act")

    context = _context_with_rules(
        special_1, special_2, normal, discovery_only, effort=Effort.MEDIUM
    )

    await matcher._run_batches(context, [special_1, special_2, normal, discovery_only])

    matcher.special_evaluator.evaluate.assert_awaited_once()
    assert set(matcher.special_evaluator.evaluate.await_args.args[1]) == {special_1, special_2}

    matcher._rule_evaluator.evaluate.assert_awaited_once()
    assert set(matcher._rule_evaluator.evaluate.await_args.args[1]) == {normal}


@pytest.mark.asyncio
async def test_that_matches_from_a_routed_evaluator_carry_its_highlights() -> None:
    matcher = cast(_RoutingMatcher, _make_batch_matcher(_RoutingMatcher))
    rule = replace(create_rule(condition="special billing", action="act"), criticality=Weight.HIGH)
    matcher.special_evaluator = AsyncMock()
    matcher.special_evaluator.evaluate = AsyncMock(
        return_value=RuleEvaluationResult(
            [
                RuleEvaluation(
                    rule=rule,
                    reasoning="applies now",
                    is_relevant=True,
                    highlights=["Charge exactly $5."],
                )
            ],
            None,
        )
    )
    context = _context_with_rules(rule, effort=Effort.MEDIUM)

    matches = await matcher._run_batches(context, [rule])

    turn_matches = [m for m, usage in matches if usage == _ContextUsage.MATCH_CURRENT_TURN]
    assert len(turn_matches) == 1
    assert turn_matches[0].metadata == {"highlights": ["Charge exactly $5."]}


@pytest.mark.asyncio
async def test_that_warm_up_prefills_each_distinct_selected_evaluator_once() -> None:
    matcher = cast(_RoutingMatcher, _make_warm_up_matcher(_RoutingMatcher))
    matcher._logger = _FakeLogger()
    matcher.special_evaluator = AsyncMock()

    special = replace(create_rule(condition="special topic", action="act"), criticality=Weight.HIGH)
    normal = replace(create_rule(condition="a normal rule", action="act"), criticality=Weight.HIGH)
    context = _context_with_rules(special, normal, effort=Effort.MEDIUM)

    await matcher.warm_up(context)

    matcher._rule_evaluator.warm_up.assert_awaited_once_with(context)
    matcher.special_evaluator.warm_up.assert_awaited_once_with(context)
