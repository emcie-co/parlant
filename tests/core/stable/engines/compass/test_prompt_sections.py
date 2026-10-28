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

"""The compass section-name registry test.

Section names are public API: prompt providers (see
docs/prompt-customization.md) intercept sections by name, so an unregistered
ad-hoc string is an undocumented, unstable contract. Every module that names
sections exposes a ``PromptSections`` enum beside the ``add_section`` calls
it names; this test builds each compass prompt through its real component and
asserts every section belongs to that union (or to ``BuiltInSection``, for
the shared builder helpers). A brand-new prompt-building module must add its
enum to ``_REGISTERED_ENUMS`` — forgetting fails loudly here.
"""

from dataclasses import replace
from typing import Any, cast

from parlant.core.agents import Effort
from parlant.core.entity_cq import TurnCannedResponses
from parlant.core.common import Weight
from parlant.core.engines.compass.prompt_builder import BuiltInSection, PromptBuilder
from parlant.core.engines.compass.compacter import Compacter
from parlant.core.engines.compass.compacter import PromptSections as CompacterSections
from parlant.core.engines.compass.matching.common import PromptSections as SharedSections
from parlant.core.engines.compass.matching.rule_pruner import PromptSections as PrunerSections
from parlant.core.engines.compass.matching.rule_pruner import RulePruner
from parlant.core.engines.compass.matching.rule_ranker import PromptSections as RankerSections
from parlant.core.engines.compass.matching.rule_ranker import RuleRanker
from parlant.core.engines.compass.responder.responder import PromptSections as ResponderSections
from parlant.core.engines.compass.responder.responder import Responder
from parlant.core.engines.compass.context import EngineContext, ResponseState
from parlant.core.engines.compass.reviewer import BasicReviewer
from parlant.core.engines.compass.reviewer import PromptSections as ReviewerSections
from parlant.core.loggers import StdoutLogger
from parlant.core.nlp.prompt_routing import PromptConfigurationService
from parlant.core.nlp.react import ToolCallPart
from parlant.core.sessions import EventSource
from parlant.core.tracer import LocalTracer

from tests.core.stable.engines.compass.matching.utils import create_engine_context, create_rule

_REGISTERED_ENUMS = (
    RankerSections,
    PrunerSections,
    SharedSections,
    CompacterSections,
    ReviewerSections,
    ResponderSections,
)

_REGISTERED_NAMES = {str(member) for enum in _REGISTERED_ENUMS for member in enum}


def assert_sections_are_registered(builder: PromptBuilder, extra: set[str] = set()) -> None:
    for name in builder.sections:
        assert isinstance(name, BuiltInSection) or str(name) in (_REGISTERED_NAMES | extra), (
            f"Unregistered prompt section name: '{name}'. Add it to the "
            "PromptSections enum of the module that adds it."
        )


def _context(effort: Effort = Effort.HIGH) -> EngineContext:
    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hello")])
    rule = replace(
        create_rule(condition="the customer asks for help", action="help them"),
        weight=Weight.HIGH,
        metadata={"references_other_policies": True},  # renders the policy-context path
    )
    other = create_rule(condition="the customer wants a refund", action="refund them")
    context.state = ResponseState(
        agent_effort=effort,
        usable_rules=[rule, other],
        discovered_rules={other},
        session_summary="Earlier parts of this session were summarized.",
        reasoning_steps=["I concluded the customer needs help."],
        # renders the outprioritized-this-turn note
        turn_deprioritized_rules=[other],
    )
    return context


def _rule(context: EngineContext) -> Any:
    return context.state.usable_rules[0]


class _StubEntityQueries:
    """Hand-written entity-queries stub scoped to what prompt building calls."""

    async def find_canned_responses_for_turn(
        self,
        rules: Any,
        agent_id: Any,
    ) -> TurnCannedResponses:
        return TurnCannedResponses(rule_scoped=[], agent_wide=[])


def test_that_the_ranker_prompt_uses_only_registered_sections() -> None:
    tracer = LocalTracer()
    ranker = RuleRanker(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        model_router=cast(Any, None),
        prompt_config=PromptConfigurationService(),
    )
    context = _context()

    builder = ranker._build_prompt(context, _rule(context), shots=[])

    assert_sections_are_registered(builder)


def test_that_the_pruner_prompt_uses_only_registered_sections() -> None:
    tracer = LocalTracer()
    pruner = RulePruner(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        model_router=cast(Any, None),
        prompt_config=PromptConfigurationService(),
    )
    context = _context()

    builder = pruner._build_prompt(context, _rule(context), shots=[])

    assert_sections_are_registered(builder)


def test_that_the_compacter_prompt_uses_only_registered_sections() -> None:
    tracer = LocalTracer()
    compacter = Compacter(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        model_router=cast(Any, None),
        prompt_config=PromptConfigurationService(),
    )
    context = _context()

    builder = compacter._build_prompt(context, compacter._policy_for(context))

    assert_sections_are_registered(builder)


def _reviewer() -> BasicReviewer:
    tracer = LocalTracer()
    return BasicReviewer(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        model_router=cast(Any, None),
        prompt_config=PromptConfigurationService(),
        entity_queries=cast(Any, _StubEntityQueries()),
        canned_response_store=cast(Any, None),
    )


def test_that_the_message_review_prompt_uses_only_registered_sections() -> None:
    context = _context()

    builder = _reviewer()._build_message_review_prompt(
        context,
        "A drafted message.",
        task_description="TASK",
        output_format="{}",
        additional_guidance="Be careful.",
    )

    assert_sections_are_registered(builder)


def test_that_the_tool_review_prompt_uses_only_registered_sections() -> None:
    context = _context()

    builder = _reviewer()._build_basic_prompt(
        context,
        reasoning="I should look something up.",
        tool_calls=[ToolCallPart(id="call-1", name="lookup", args={})],
    )

    assert_sections_are_registered(builder)


async def test_that_the_responder_prompts_use_only_registered_sections() -> None:
    captured: list[PromptBuilder] = []

    class _CapturingBuilder(PromptBuilder):
        def __init__(self) -> None:
            super().__init__()
            captured.append(self)

    prompt_config = PromptConfigurationService()
    from parlant.core.nlp.model_routing import ModelScope

    await prompt_config.configure(ModelScope.server(), {"react": _CapturingBuilder})

    responder = object.__new__(Responder)
    responder._prompt_config = prompt_config
    responder._streaming_loop = cast(Any, None)
    responder._blocking_loop = cast(Any, None)
    responder._entity_queries = cast(Any, _StubEntityQueries())
    responder._canned_response_store = cast(Any, None)
    from parlant.core.engines.compass.preambles import PreambleConfigurationProvider

    responder._preamble_config_provider = PreambleConfigurationProvider()
    context = _context()

    responder._build_system_instructions(context)
    await responder._build_step_instructions(context)

    assert captured, "The responder did not build through the configured provider"
    for builder in captured:
        assert_sections_are_registered(builder)
