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

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from parlant.core.agents import Effort, MessageOutputMode
from parlant.core.canned_responses import (
    CannedResponse,
    CannedResponseRelevanceResult,
)
from parlant.core.entity_cq import TurnCannedResponses
from parlant.core.engines.compass.responder.responder import Responder
from parlant.core.engines.compass.context import (
    DEFAULT_AGENT_MODERATION_FALLBACK,
    EngineContext,
    ResponseState,
)
from parlant.core.engines.compass.preambles import (
    PreambleConfiguration,
    PreambleConfigurationProvider,
)
from parlant.core.engines.compass.turn_directives import TurnDirective
from parlant.core.sessions import EventSource
from parlant.core.tools import ToolId

from tests.core.stable.engines.compass.matching.utils import create_engine_context


class _StubEntityQueries:
    """Hand-written entity-queries stub scoped to what the responder calls."""

    def __init__(self, rule_scoped: list[Any] | None = None) -> None:
        self._rule_scoped = rule_scoped or []

    async def find_canned_responses_for_turn(
        self,
        rules: Any,
        agent_id: Any,
    ) -> TurnCannedResponses:
        return TurnCannedResponses(rule_scoped=self._rule_scoped, agent_wide=[])


class _StubCannedResponseStore:
    def __init__(self, relevant: list[Any]) -> None:
        self._relevant = relevant
        self.ranking_calls = 0

    async def filter_relevant_canned_responses(
        self,
        query: str,
        available_canned_responses: Any,
        max_count: int,
    ) -> list[CannedResponseRelevanceResult]:
        self.ranking_calls += 1
        return [
            CannedResponseRelevanceResult(canned_response=cr, score=0.9) for cr in self._relevant
        ]


class _PassthroughCannedResponseRenderer:
    def render(self, canned_response: Any, context: Any) -> str:
        return cast(str, canned_response.value)

    def can_satisfy(self, canned_response: Any, context: Any) -> bool:
        return True


class _SilentLogger:
    """Only the warning path is exercised here; keep them for assertions."""

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def _make_responder() -> Responder:
    from parlant.core.nlp.prompt_routing import PromptConfigurationService

    responder = object.__new__(Responder)
    responder._streaming_loop = cast(Any, object())
    responder._blocking_loop = cast(Any, object())
    responder._prompt_config = PromptConfigurationService()
    responder._preamble_config_provider = PreambleConfigurationProvider()
    # Default to no canned responses in scope; the canned-response tests below
    # override this with their own entity-queries / store / renderer.
    responder._entity_queries = cast(Any, _StubEntityQueries())
    responder._logger = cast(Any, _SilentLogger())
    responder.max_canned_responses = 5
    responder.max_unranked_canned_responses = 20
    responder.moderation_fallback = DEFAULT_AGENT_MODERATION_FALLBACK
    return responder


def test_that_the_responder_builds_jobs_with_the_agents_preamble_configuration() -> None:
    responder = _make_responder()
    context = _context(MessageOutputMode.STREAM)
    configuration = PreambleConfiguration.discourage()
    responder._preamble_config_provider.set(context.agent.id, configuration)

    job = responder._build_job(context)

    assert job.preamble_config is configuration


def _context(output_mode: MessageOutputMode) -> EngineContext:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.agent = replace(context.agent, message_output_mode=output_mode)
    context.state = ResponseState()
    return context


def test_that_text_output_modes_select_their_loops() -> None:
    responder = _make_responder()

    assert (
        responder._loop_for(_context(MessageOutputMode.STREAM))
        is cast(Any, responder)._streaming_loop
    )
    assert (
        responder._loop_for(_context(MessageOutputMode.BLOCK))
        is cast(Any, responder)._blocking_loop
    )


def test_that_voice_output_mode_requires_a_voice_capable_responder() -> None:
    # The mode stays in the enum and API surface so it can be configured, but
    # this responder has no voice-capable loop that can deliver it.
    responder = _make_responder()

    with pytest.raises(NotImplementedError, match="voice-capable"):
        responder._loop_for(_context(MessageOutputMode.VOICE))


def test_that_the_base_responder_has_no_spoken_style_prompt_section() -> None:
    responder = _make_responder()

    instructions = responder._build_system_instructions(_context(MessageOutputMode.STREAM))

    assert "SPOKEN CONVERSATION STYLE" not in instructions


# --- Composition-mode independence -----------------------------------------------


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_that_turn_directives_create_step_instructions_for_low_effort_agents() -> None:
    responder = _make_responder()
    context = _context(MessageOutputMode.STREAM)
    context.state.agent_effort = Effort.LOW
    context.state.turn_directives = [
        TurnDirective(
            instruction="Tell the customer the booking is confirmed.",
            rationale="Follow up with the customer.",
        )
    ]

    build = responder._get_step_instructions(context, refresh_state=None)

    assert build is not None
    instructions = await build(context)
    assert "DIRECTIVES FOR THIS RESPONSE" in instructions
    assert "booking is confirmed" in instructions
    assert "Follow up with the customer" in instructions


@pytest.mark.asyncio
async def test_that_turn_directives_are_omitted_from_the_terminal_give_up_step() -> None:
    responder = _make_responder()
    context = _context(MessageOutputMode.STREAM)
    context.state.turn_directives = [
        TurnDirective(
            instruction="Tell the customer the booking is confirmed.",
            rationale="Follow up with the customer.",
        )
    ]
    context.state.giving_up = True

    instructions = await responder._build_step_instructions(context)

    assert "booking is confirmed" not in instructions


# --- Canned responses as style references ----------------------------------------


def _canned_response(value: str, note: str = "") -> Any:
    cr = CannedResponse.create_transient(value=value)
    return cast(Any, SimpleNamespace(**{**cr.__dict__, "note": note}))


def _rule_match() -> Any:
    from datetime import datetime, timezone

    from parlant.core.common import Weight, generate_id
    from parlant.core.engines.rule_evaluation import RuleEvaluation
    from parlant.core.rules import Rule, RuleId

    now = datetime.now(timezone.utc)
    rule = Rule(
        id=RuleId(generate_id()),
        creation_utc=now,
        modified_utc=now,
        condition="c",
        content="a",
        enabled=True,
        groups=[],
        metadata={},
        weight=Weight.MEDIUM,
    )
    return RuleEvaluation(rule=rule, reasoning="matched", metadata={})


def _style_reference_responder(
    canned_responses: list[Any],
    relevant: list[Any],
) -> Responder:
    responder = _make_responder()
    responder._entity_queries = cast(Any, _StubEntityQueries(rule_scoped=canned_responses))
    responder._canned_response_store = cast(Any, _StubCannedResponseStore(relevant))
    responder._canned_response_renderer = cast(Any, _PassthroughCannedResponseRenderer())
    return responder


@pytest.mark.asyncio
async def test_that_relevant_canned_responses_are_rendered_as_style_references_in_step_instructions() -> (
    None
):
    canrep = _canned_response("Would you like a banana?", note="Always offer a banana.")
    responder = _style_reference_responder([canrep], [canrep])
    context = _context(MessageOutputMode.STREAM)
    context.state.ordinary_rule_matches = [_rule_match()]

    instructions = await responder._build_step_instructions(context)

    assert "STYLE REFERENCE" in instructions
    assert "Would you like a banana?" in instructions
    assert "Always offer a banana." in instructions


@pytest.mark.asyncio
async def test_that_step_instructions_omit_style_references_when_no_canned_responses_are_relevant() -> (
    None
):
    responder = _style_reference_responder([], [])
    context = _context(MessageOutputMode.STREAM)
    context.state.ordinary_rule_matches = [_rule_match()]

    instructions = await responder._build_step_instructions(context)

    assert "STYLE REFERENCE" not in instructions


def _bare_rule(weight: Any = None) -> Any:
    # A distinctive marker condition: `_rule_match()`'s condition is the single
    # letter "c", which appears in any prompt — useless for presence asserts.
    from dataclasses import replace as _replace

    from parlant.core.common import Weight

    from tests.core.stable.engines.compass.matching.utils import create_rule

    return _replace(
        create_rule(
            condition="ZZQ_BARE the request is not covered by any of our policies",
            action="say you cannot help",
        ),
        weight=weight or Weight.MEDIUM,
        metadata={"references_other_policies": True},
    )


def test_that_statically_included_rules_are_rendered_in_the_system_instructions() -> None:
    # Bare cross-referential rules (no attachments) skip matching entirely and
    # are always rendered. See matching/common.py.
    responder = _make_responder()
    context = _context(MessageOutputMode.STREAM)
    context.state.statically_included_rules = [_bare_rule()]

    instructions = responder._build_system_instructions(context)

    assert "ZZQ_BARE" in instructions


def test_that_a_low_weight_statically_included_rule_is_rendered_in_the_system_instructions() -> (
    None
):
    # A bare cross-referential rule is excluded from discovery, so the
    # renderer cannot rely on the discovered set for it — whatever its
    # weight. A LOW-weight bare rule must still be rendered.
    from parlant.core.common import Weight

    responder = _make_responder()
    context = _context(MessageOutputMode.STREAM)
    context.state.statically_included_rules = [_bare_rule(Weight.LOW)]

    instructions = responder._build_system_instructions(context)

    assert "ZZQ_BARE" in instructions


def test_that_a_statically_included_high_weight_rule_keeps_its_high_weight_note() -> None:
    # Statically included rules render with their declared weight — a HIGH
    # bare rule keeps the high-weight IMPORTANT note, not a demoted framing.
    from parlant.core.common import Weight

    responder = _make_responder()
    context = _context(MessageOutputMode.STREAM)
    context.state.statically_included_rules = [_bare_rule(Weight.HIGH)]

    instructions = responder._build_system_instructions(context)

    assert "ZZQ_BARE" in instructions
    assert "high-weight instruction" in instructions


# --- Overlapping tools note -------------------------------------------------------


@pytest.mark.asyncio
async def test_that_overlapping_tools_render_an_alternatives_note_in_step_instructions() -> None:
    responder = _make_responder()
    context = _context(MessageOutputMode.STREAM)
    context.state.overlapping_tool_ids = [
        (ToolId("first_service", "schedule"), ToolId("second_service", "schedule")),
    ]

    instructions = await responder._build_step_instructions(context)

    assert "OVERLAPPING TOOLS" in instructions
    assert "first_service:schedule" in instructions
    assert "second_service:schedule" in instructions


@pytest.mark.asyncio
async def test_that_no_overlap_note_is_rendered_without_overlap_configuration() -> None:
    responder = _make_responder()
    context = _context(MessageOutputMode.STREAM)

    instructions = await responder._build_step_instructions(context)

    assert "OVERLAPPING TOOLS" not in instructions


@pytest.mark.asyncio
async def test_that_low_effort_agents_get_step_instructions_without_re_matching() -> None:
    responder = _make_responder()
    context = _context(MessageOutputMode.STREAM)
    context.state.agent_effort = Effort.LOW
    context.state.iterations = [cast(Any, object())]

    refreshes = 0

    async def refresh_state(ctx: Any) -> None:
        nonlocal refreshes
        refreshes += 1

    build = responder._get_step_instructions(context, refresh_state=refresh_state)

    assert build is not None

    await build(context)

    # Re-matching is the part low effort pays to avoid; the instructions
    # themselves render from state that is already loaded.
    assert refreshes == 0


@pytest.mark.asyncio
async def test_that_canned_responses_reach_low_effort_agents_without_ranking() -> None:
    canrep = _canned_response("Would you like a banana?", note="Always offer a banana.")
    responder = _style_reference_responder([canrep], [canrep])
    context = _context(MessageOutputMode.STREAM)
    context.state.agent_effort = Effort.LOW
    context.state.ordinary_rule_matches = [_rule_match()]

    instructions = await responder._build_step_instructions(context)

    assert "STYLE REFERENCE" in instructions
    assert "Would you like a banana?" in instructions
    assert cast(Any, responder._canned_response_store).ranking_calls == 0


@pytest.mark.asyncio
async def test_that_medium_effort_agents_still_rank_canned_responses() -> None:
    canrep = _canned_response("Would you like a banana?", note="Always offer a banana.")
    responder = _style_reference_responder([canrep], [canrep])
    context = _context(MessageOutputMode.STREAM)
    context.state.agent_effort = Effort.MEDIUM
    context.state.ordinary_rule_matches = [_rule_match()]

    instructions = await responder._build_step_instructions(context)

    assert "STYLE REFERENCE" in instructions
    assert cast(Any, responder._canned_response_store).ranking_calls == 1


@pytest.mark.asyncio
async def test_that_high_effort_agents_still_re_match_before_building() -> None:
    # The regression guard for the expensive path: above LOW, a step that
    # follows a tool call still refreshes the match.
    responder = _make_responder()
    context = _context(MessageOutputMode.STREAM)
    context.state.agent_effort = Effort.HIGH
    context.state.iterations = [cast(Any, object())]

    refreshes = 0

    async def refresh_state(ctx: Any) -> None:
        nonlocal refreshes
        refreshes += 1

    build = responder._get_step_instructions(context, refresh_state=refresh_state)

    assert build is not None

    await build(context)

    assert refreshes == 1
