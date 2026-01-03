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

from __future__ import annotations
import asyncio
import copy
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from itertools import chain
import json
from pprint import pformat
import traceback
from typing import Awaitable, Callable, Mapping, Optional, Sequence, cast
from croniter import croniter
from typing_extensions import override

from parlant.core import async_utils
from parlant.core.agents import Agent, AgentId, CompositionMode
from parlant.core.capabilities import Capability
from parlant.core.common import Criticality, JSONSerializable
from parlant.core.context_variables import (
    ContextVariable,
    ContextVariableValue,
    ContextVariableStore,
)
from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.engines.alpha.engine_context import (
    Interaction,
    IterationState,
    EngineContext,
    ResponseState,
)
from parlant.core.engines.alpha.entity_context import EntityContext
from parlant.core.engines.alpha.message_generator import MessageGenerator
from parlant.core.engines.alpha.hooks import EngineHooks
from parlant.core.engines.alpha.perceived_performance_policy import (
    PerceivedPerformancePolicyProvider,
)
from parlant.core.engines.alpha.relational_guideline_resolver import RelationalGuidelineResolver
from parlant.core.engines.alpha.tool_calling.tool_caller import (
    MissingToolData,
    ToolInsights,
    InvalidToolData,
    ProblematicToolData,
)
from parlant.core.engines.alpha.canned_response_generator import CannedResponseGenerator
from parlant.core.engines.alpha.message_event_composer import (
    MessageEventComposer,
)
from parlant.core.guidelines import Guideline, GuidelineId, GuidelineContent
from parlant.core.glossary import Term
from parlant.core.journey_guideline_projection import (
    extract_node_id_from_journey_node_guideline_id,
)
from parlant.core.journeys import Journey, JourneyId
from parlant.core.meter import Meter
from parlant.core.app_modules.sessions import SessionUpdateParamsModel
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.sessions import (
    AgentState,
    EventKind,
    Session,
    ToolEventData,
)
from parlant.core.engines.alpha.guideline_matching.guideline_matcher import (
    GuidelineMatcher,
    GuidelineMatchingResult,
)
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.tool_event_generator import (
    ToolEventGenerationResult,
    ToolEventGenerator,
    ToolPreexecutionState,
)
from parlant.core.engines.alpha.utils import context_variables_to_json
from parlant.core.engines.types import Context, Engine, UtteranceRationale, UtteranceRequest
from parlant.core.emissions import EventEmitter, EmittedEvent
from parlant.core.tracer import Tracer
from parlant.core.loggers import Logger
from parlant.core.entity_cq import EntityQueries, EntityCommands
from parlant.core.tools import ToolContext, ToolId


_PREPARATION_ITERATION_SPAN_NAME = "preparation_iteration_{iteration_number}"
_GUIDELINE_MATCHER_SPAN_NAME = "guideline_matcher"
_RESPONSE_ANALYSIS_SPAN_NAME = "response_analysis"
_MESSAGE_GENERATION_SPAN_NAME = "message_generation"
_TOOL_CALLER_SPAN_NAME = "tool_caller"


class _PreparationIterationResolution(Enum):
    COMPLETED = "continue"
    """Continue with the next preparation iteration"""

    BAIL = "bail"
    """Bail out of the preparation iterations, as requested by a hook"""


@dataclass
class _PreparationIterationResult:
    state: IterationState
    resolution: _PreparationIterationResolution


@dataclass(frozen=True)
class _GuidelineAndJourneyMatchingResult:
    matching_result: GuidelineMatchingResult
    matches_guidelines: list[GuidelineMatch]
    resolved_guidelines: list[GuidelineMatch]
    journeys: list[Journey]


@dataclass(frozen=True)
class _MessageGeneration:
    generations: Mapping[str, GenerationInfo]
    messages: Sequence[str | None]


class AlphaEngine(Engine):
    """The main AI processing engine (as of Feb 25, the latest and greatest processing engine)"""

    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        entity_queries: EntityQueries,
        entity_commands: EntityCommands,
        guideline_matcher: GuidelineMatcher,
        relational_guideline_resolver: RelationalGuidelineResolver,
        tool_event_generator: ToolEventGenerator,
        fluid_message_generator: MessageGenerator,
        canned_response_generator: CannedResponseGenerator,
        perceived_performance_policy_provider: PerceivedPerformancePolicyProvider,
        hooks: EngineHooks,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._meter = meter

        self._entity_queries = entity_queries
        self._entity_commands = entity_commands

        self._guideline_matcher = guideline_matcher
        self._relational_guideline_resolver = relational_guideline_resolver
        self._tool_event_generator = tool_event_generator
        self._fluid_message_generator = fluid_message_generator
        self._canned_response_generator = canned_response_generator
        self._perceived_performance_policy_provider = perceived_performance_policy_provider

        self._hooks = hooks

        self._hist_engine_process_duration = self._meter.create_duration_histogram(
            name="eng.process",
            description="Duration of engine processing in milliseconds",
        )
        self._hist_engine_utter_duration = self._meter.create_duration_histogram(
            name="eng.utter",
            description="Duration of engine utter in milliseconds",
        )

    @override
    async def process(
        self,
        context: Context,
        event_emitter: EventEmitter,
    ) -> bool:
        """Processes a context and emits new events as needed"""

        # Load the full relevant information from storage.
        loaded_context = await self._load_context(context, event_emitter)

        if loaded_context.session.mode == "manual":
            return True

        try:
            with self._tracer.span("process", {"session_id": context.session_id}):
                async with self._hist_engine_process_duration.measure():
                    await self._do_process(loaded_context)
            return True
        except asyncio.CancelledError:
            return False
        except Exception as exc:
            formatted_exception = pformat(traceback.format_exception(exc))

            self._logger.error(f"Processing error: {formatted_exception}")

            if await self._hooks.call_on_error(loaded_context, exc):
                await self._emit_error_event(loaded_context, formatted_exception)

            return False
        except BaseException as exc:
            self._logger.critical(f"Critical processing error: {traceback.format_exception(exc)}")
            raise

    @override
    async def utter(
        self,
        context: Context,
        event_emitter: EventEmitter,
        requests: Sequence[UtteranceRequest],
    ) -> bool:
        """Produces a new message into a session, guided by specific utterance requests"""

        # Load the full relevant information from storage.
        loaded_context = await self._load_context(
            context,
            event_emitter,
            load_interaction=True,
        )

        try:
            async with self._hist_engine_utter_duration.measure(
                {"session_id": context.session_id},
            ):
                with self._tracer.span("utter", {"session_id": context.session_id}):
                    await self._do_utter(loaded_context, requests)
            return True

        except asyncio.CancelledError:
            self._logger.warning(f"Uttering in session {context.session_id} was cancelled.")
            return False
        except Exception as exc:
            formatted_exception = pformat(traceback.format_exception(exc))

            self._logger.error(
                f"Error during uttering in session {context.session_id}: {formatted_exception}"
            )

            if await self._hooks.call_on_error(loaded_context, exc):
                await self._emit_error_event(loaded_context, formatted_exception)

            return False
        except BaseException as exc:
            self._logger.critical(
                f"Critical error during uttering in session {context.session_id}: "
                f"{traceback.format_exception(type(exc), exc, exc.__traceback__)}"
            )
            raise

    async def _load_interaction_state(self, context: Context) -> Interaction:
        history = await self._entity_queries.find_events(context.session_id)

        return Interaction(
            events=history,
        )

    async def _do_process(
        self,
        context: EngineContext,
    ) -> None:
        if not await self._hooks.call_on_acknowledging(context):
            return  # Hook requested to bail out

        # Mark that this latest session state has been seen by the agent.
        await self._emit_acknowledgement_event(context)

        if not await self._hooks.call_on_acknowledged(context):
            return  # Hook requested to bail out

        try:
            if not await self._hooks.call_on_preparing(context):
                return  # Hook requested to bail out

            await self._initialize_response_state(context)

            while not context.state.prepared_to_respond:
                # Need more data before we're ready to respond

                preamble_task = await self._get_preamble_task(context)

                if not await self._hooks.call_on_preparation_iteration_start(context):
                    break  # Hook requested to finish preparing

                # Get more data (guidelines, tools, etc.,)
                # This happens in iterations in order to support a feedback loop
                # where particular tool-call results may trigger new or different
                # guidelines that we need to follow.
                iteration_result = await self._run_preparation_iteration(context, preamble_task)

                if iteration_result.resolution == _PreparationIterationResolution.BAIL:
                    return

                # Some tools may update session mode (e.g. from automatic to manual).
                # This is particularly important to support human handoff.
                await self._update_session_mode(context)

                if not await self._hooks.call_on_preparation_iteration_end(context):
                    break

            # Filter missing and invalid tool parameters jointly
            problematic_data = await self._filter_problematic_tool_parameters_based_on_precedence(
                list(context.state.tool_insights.missing_data)
                + list(context.state.tool_insights.invalid_data)
            )
            context.state.tool_insights = ToolInsights(
                evaluations=context.state.tool_insights.evaluations,
                missing_data=[p for p in problematic_data if isinstance(p, MissingToolData)],
                invalid_data=[p for p in problematic_data if isinstance(p, InvalidToolData)],
            )

            async def uncancellable_section(
                latch: async_utils.CancellationSuppressionLatch[None],
            ) -> None:
                if not await self._hooks.call_on_generating_messages(context):
                    return

                # Call on_match handlers for all matched guidelines (before generating messages)
                await self._call_guideline_handlers(
                    context, self._hooks.on_guideline_match_handlers
                )

                # Money time: communicate with the customer given
                # all of the information we have prepared.
                with self._tracer.span(_MESSAGE_GENERATION_SPAN_NAME):
                    _ = await self._generate_messages(context, latch)

                # Mark that the agent is ready to receive and respond to new events.
                await self._emit_ready_event(context)

                await self._add_agent_state(
                    context=context,
                    session=context.session,
                    guideline_matches=list(
                        chain(
                            context.state.ordinary_guideline_matches,
                            context.state.tool_enabled_guideline_matches,
                        )
                    ),
                )

                await self._hooks.call_on_messages_emitted(context)

                # Call on_message handlers for matched guidelines (after messages emitted)
                await self._call_guideline_handlers(
                    context, self._hooks.on_guideline_message_handlers
                )

            await async_utils.latched_shield(uncancellable_section)

        except asyncio.CancelledError:
            # Task was cancelled. This usually happens for 1 of 2 reasons:
            #   1. The server is shutting down
            #   2. New information arrived and the currently loaded
            #      processing context is likely to be obsolete
            self._logger.warning("Processing cancelled")
            await self._emit_cancellation_event(context)
            await self._emit_ready_event(context)
            raise
        except Exception:
            # Mark that the agent is ready to receive and respond to new events.
            await self._emit_ready_event(context)
            raise

    async def _do_utter(
        self,
        context: EngineContext,
        requests: Sequence[UtteranceRequest],
    ) -> None:
        try:
            await self._initialize_response_state(context)

            # Only use the specified utterance requests as guidelines here.
            context.state.ordinary_guideline_matches.extend(
                # Utterance requests are reduced to guidelines, to take advantage
                # of the engine's ability to consistently adhere to guidelines.
                await self._utterance_requests_to_guideline_matches(requests)
            )

            async def uncancellable_section(
                latch: async_utils.CancellationSuppressionLatch[None],
            ) -> None:
                # Money time: communicate with the customer given the
                # specified utterance requests.
                _ = await self._generate_messages(context, latch)

            await async_utils.latched_shield(uncancellable_section)

        except asyncio.CancelledError:
            self._logger.warning("Uttering cancelled")
            raise
        finally:
            # Mark that the agent is ready to receive and respond to new events.
            await self._emit_ready_event(context)

    async def _load_context(
        self,
        context: Context,
        event_emitter: EventEmitter,
        load_interaction: bool = True,
    ) -> EngineContext:
        # Load the full entities from storage.

        agent = await self._entity_queries.read_agent(context.agent_id)
        session = await self._entity_queries.read_session(context.session_id)
        customer = await self._entity_queries.read_customer(session.customer_id)

        if load_interaction:
            interaction = await self._load_interaction_state(context)
        else:
            interaction = Interaction([])

        result = EngineContext(
            info=context,
            logger=self._logger,
            tracer=self._tracer,
            agent=agent,
            customer=customer,
            session=session,
            session_event_emitter=event_emitter,
            response_event_emitter=EventBuffer(agent),
            interaction=interaction,
            state=ResponseState(
                context_variables=[],
                glossary_terms=set(),
                capabilities=[],
                iterations=[],
                ordinary_guideline_matches=[],
                tool_enabled_guideline_matches={},
                journeys=[],
                journey_paths={
                    k: list(v) for k, v in session.agent_states[-1].journey_paths.items()
                }
                if session.agent_states
                else {},
                tool_events=[],
                tool_insights=ToolInsights(),
                prepared_to_respond=False,
                message_events=[],
            ),
        )

        # Set in context for access by hooks and other components
        EntityContext.set(result)

        return result

    async def _initialize_response_state(
        self,
        context: EngineContext,
    ) -> None:
        # Load the relevant context variable values.
        context.state.context_variables = await self._load_context_variables(context)

        # Load relevant glossary terms and capabilities, initially based
        # mostly on the current interaction history.
        glossary, capabilities = await async_utils.safe_gather(
            self._load_glossary_terms(context),
            self._load_capabilities(context),
        )

        context.state.glossary_terms.update(glossary)
        context.state.capabilities = list(capabilities)

    async def _run_preparation_iteration(
        self,
        context: EngineContext,
        preamble_task: asyncio.Task[bool],
    ) -> _PreparationIterationResult:
        with self._tracer.span(
            _PREPARATION_ITERATION_SPAN_NAME.format(
                iteration_number=len(context.state.iterations) + 1
            )
        ):
            if len(context.state.iterations) == 0:
                # This is the first iteration, so we need to run the initial preparation iteration.
                result = await self._run_initial_preparation_iteration(context, preamble_task)

            else:
                # This is an additional iteration, so we run the additional preparation iteration.
                result = await self._run_additional_preparation_iteration(context)

            context.state.iterations.append(result.state)
            context.state.journey_paths = self._list_journey_paths(context=context)

            # If there's no new information to consider (which would have come from
            # the tools), then we can consider ourselves prepared to respond.
            if await self._check_if_prepared(context, result):
                context.state.prepared_to_respond = True

            # Alternatively, we we've reached the max number of iterations,
            # we should just go ahead and respond anyway, despite possibly
            # needing more data for a fully accurate response.
            #
            # This is a trade-off that can be controlled by adjusting the max.
            elif len(context.state.iterations) == context.agent.max_engine_iterations:
                self._logger.warning(
                    f"Reached max tool call iterations ({context.agent.max_engine_iterations})"
                )
                context.state.prepared_to_respond = True

            return result

    async def _check_if_prepared(
        self,
        context: EngineContext,
        result: _PreparationIterationResult,
    ) -> bool:
        # If there's no new information to consider (which would have come from
        # the tools), then we can consider ourselves prepared to respond.
        def check_if_journey_node_with_tool_is_matched() -> bool:
            for m in context.state.tool_enabled_guideline_matches:
                if m.guideline.metadata.get("journey_node"):
                    return True
            return False

        if result.state.executed_tools or check_if_journey_node_with_tool_is_matched():
            return False

        return True

    async def _run_initial_preparation_iteration(
        self,
        context: EngineContext,
        preamble_task: asyncio.Task[bool],
    ) -> _PreparationIterationResult:
        matching_finished = False

        async def extended_thinking_status_emission() -> None:
            nonlocal matching_finished

            while not preamble_task.done():
                await asyncio.sleep(0.1)

            if matching_finished:
                return

            policy = self._perceived_performance_policy_provider.get_policy(context.agent.id)
            timeout = async_utils.Timeout(await policy.get_extended_processing_indicator_delay())

            while not matching_finished:
                if await timeout.wait_up_to(0.1):
                    await self._emit_processing_event(context, stage="Thinking")
                    return

        extended_thinking_status_task = asyncio.create_task(extended_thinking_status_emission())

        try:
            # For optimization concerns, it's useful to capture the exact state
            # we were in before matching guidelines.
            tool_preexecution_state = await self._capture_tool_preexecution_state(context)

            # Match relevant guidelines, retrieving them in a
            # structured format such that we can distinguish
            # between ordinary and tool-enabled ones.
            guideline_and_journey_matching_result = (
                await self._load_matched_guidelines_and_journeys(context)
            )

            matching_finished = True

            context.state.journeys = guideline_and_journey_matching_result.journeys
        finally:
            await extended_thinking_status_task

        if not await preamble_task:
            # Bail out on the rest of the processing, as the preamble
            # hook decided we should not proceed with processing.
            return _PreparationIterationResult(
                state=IterationState(
                    matched_guidelines=guideline_and_journey_matching_result.matches_guidelines,
                    resolved_guidelines=guideline_and_journey_matching_result.resolved_guidelines,
                    tool_insights=ToolInsights(),
                    executed_tools=[],
                ),
                resolution=_PreparationIterationResolution.BAIL,
            )

        # Matched guidelines may use glossary terms, so we need to ground our
        # response by reevaluating the relevant terms given these new guidelines.
        context.state.glossary_terms.update(await self._load_glossary_terms(context))

        # Distinguish between ordinary and tool-enabled guidelines.
        # We do this here as it creates a better subsequent control flow in the engine.
        context.state.tool_enabled_guideline_matches = (
            await self._find_tool_enabled_guideline_matches(
                guideline_matches=guideline_and_journey_matching_result.resolved_guidelines,
            )
        )

        context.state.ordinary_guideline_matches = list(
            set(guideline_and_journey_matching_result.resolved_guidelines).difference(
                set(context.state.tool_enabled_guideline_matches.keys())
            ),
        )

        # Infer any needed tool calls and execute them,
        # adding the resulting tool events to the session.
        if tool_calling_result := await self._call_tools(context, tool_preexecution_state):
            (
                tool_event_generation_result,
                new_tool_events,
                tool_insights,
            ) = tool_calling_result

            context.state.tool_events += new_tool_events
            context.state.tool_insights = tool_insights

            self._add_tool_events_to_tracer(new_tool_events)

        else:
            new_tool_events = []

        # Tool calls may have returned with data that uses glossary terms,
        # so we need to ground our response again by reevaluating terms.
        context.state.glossary_terms.update(await self._load_glossary_terms(context))

        # Return structured inspection information, useful for later troubleshooting.
        return _PreparationIterationResult(
            state=IterationState(
                matched_guidelines=guideline_and_journey_matching_result.matches_guidelines,
                resolved_guidelines=guideline_and_journey_matching_result.resolved_guidelines,
                tool_insights=tool_insights,
                executed_tools=[
                    ToolId.from_string(tool_call["tool_id"])
                    for tool_event in new_tool_events
                    for tool_call in cast(ToolEventData, tool_event.data)["tool_calls"]
                ],
            ),
            resolution=_PreparationIterationResolution.COMPLETED,
        )

    async def _run_additional_preparation_iteration(
        self,
        context: EngineContext,
    ) -> _PreparationIterationResult:
        # For optimization concerns, it's useful to capture the exact state
        # we were in before matching guidelines.
        tool_preexecution_state = await self._capture_tool_preexecution_state(context)

        # Match and retrieve guidelines and journeys based on the results of the previous iteration.
        guideline_and_journey_matching_result = (
            await self._load_additional_matched_guidelines_and_journeys(context)
        )

        # FIXME: There might be cases where a journey got ACTIVATED, and then, during
        # an additional iteration actually became INACTIVE. In those cases, we wouldn't
        # actually want to perform the additional processing (matching, etc.) on it.
        # Yet, currently, we keep it active and we do do that.
        # I don't expect that this behavior causes actual issues beyond the occasional
        # added costs (and perhaps latency) in this type of edge case, but still it's not ideal
        # - Dorzo
        context.state.journeys += guideline_and_journey_matching_result.journeys

        # Matched guidelines may use glossary terms, so we need to ground our
        # response by reevaluating the relevant terms given these new guidelines.
        context.state.glossary_terms.update(await self._load_glossary_terms(context))

        # Distinguish between ordinary and tool-enabled guidelines.
        # We do this here as it creates a better subsequent control flow in the engine.
        # Since its iteration > 1, we consider only newly matched guidelines.
        context.state.tool_enabled_guideline_matches = (
            await self._find_tool_enabled_guideline_matches(
                guideline_matches=list(
                    set(guideline_and_journey_matching_result.matches_guidelines).intersection(
                        set(guideline_and_journey_matching_result.resolved_guidelines)
                    )
                ),
            )
        )

        context.state.ordinary_guideline_matches = list(
            set(guideline_and_journey_matching_result.resolved_guidelines).difference(
                set(context.state.tool_enabled_guideline_matches.keys())
            ),
        )

        # Infer any needed tool calls and execute them,
        # adding the resulting tool events to the session.
        if tool_calling_result := await self._call_tools(context, tool_preexecution_state):
            (
                tool_event_generation_result,
                new_tool_events,
                tool_insights,
            ) = tool_calling_result

            context.state.tool_events += new_tool_events
            context.state.tool_insights = ToolInsights(
                evaluations=list(
                    chain(context.state.tool_insights.evaluations, tool_insights.evaluations)
                ),
                missing_data=list(
                    chain(context.state.tool_insights.missing_data, tool_insights.missing_data)
                ),
                invalid_data=list(
                    chain(context.state.tool_insights.invalid_data, tool_insights.invalid_data)
                ),
            )

            self._add_tool_events_to_tracer(new_tool_events)

        else:
            new_tool_events = []

        # Tool calls may have returned with data that uses glossary terms,
        # so we need to ground our response again by reevaluating terms.
        context.state.glossary_terms.update(await self._load_glossary_terms(context))

        return _PreparationIterationResult(
            state=IterationState(
                matched_guidelines=guideline_and_journey_matching_result.matches_guidelines,
                resolved_guidelines=guideline_and_journey_matching_result.resolved_guidelines,
                tool_insights=tool_insights,
                executed_tools=[
                    ToolId.from_string(tool_call["tool_id"])
                    for tool_event in new_tool_events
                    for tool_call in cast(ToolEventData, tool_event.data)["tool_calls"]
                ],
            ),
            resolution=_PreparationIterationResolution.COMPLETED,
        )

    async def _update_session_mode(self, context: EngineContext) -> None:
        # Do we even have control-requests coming from any called tools?
        if tool_call_control_outputs := [
            tool_call["result"]["control"]
            for tool_event in context.state.tool_events
            for tool_call in cast(ToolEventData, tool_event.data)["tool_calls"]
        ]:
            # Yes we do. Update session mode as needed.

            current_session_mode = context.session.mode
            new_session_mode = current_session_mode

            for control_output in tool_call_control_outputs:
                new_session_mode = control_output.get("mode") or current_session_mode

            if new_session_mode != current_session_mode:
                self._logger.info(
                    f"Changing session {context.session.id} mode to '{new_session_mode}'"
                )

                await self._entity_commands.update_session(
                    session_id=context.session.id,
                    params={
                        "mode": new_session_mode,
                    },
                )

    async def _get_preamble_task(self, context: EngineContext) -> asyncio.Task[bool]:
        async def preamble_task() -> bool:
            policy = self._perceived_performance_policy_provider.get_policy(context.agent.id)

            if (
                # Only consider a preamble in the first iteration
                len(context.state.iterations) == 0 and await policy.is_preamble_required(context)
            ):
                if not await self._hooks.call_on_generating_preamble(context):
                    return False

                await asyncio.sleep(
                    await policy.get_preamble_delay(context),
                )

                if await self._generate_preamble(context):
                    context.interaction = await self._load_interaction_state(context.info)

                await self._emit_ready_event(context)

                if not await self._hooks.call_on_preamble_emitted(context):
                    return False

                # Emit a processing event to indicate that the agent is thinking

                await asyncio.sleep(
                    await policy.get_processing_indicator_delay(context),
                )

                await self._emit_processing_event(context, stage="Interpreting")

                return True

            else:
                return True  # No preamble message is needed

        return asyncio.create_task(preamble_task())

    async def _generate_preamble(
        self,
        context: EngineContext,
    ) -> bool:
        generated_messages = False

        for event_generation_result in await self._get_message_composer(
            context.agent
        ).generate_preamble(context=context):
            generated_messages = True
            context.state.message_events += [e for e in event_generation_result.events if e]

        return generated_messages

    async def _generate_messages(
        self,
        context: EngineContext,
        latch: async_utils.CancellationSuppressionLatch[None],
    ) -> Sequence[_MessageGeneration]:
        message_generation = []

        for event_generation_result in await self._get_message_composer(
            context.agent
        ).generate_response(
            context=context,
            latch=latch,
        ):
            context.state.message_events += [e for e in event_generation_result.events if e]

            message_generation.append(
                _MessageGeneration(
                    generations=event_generation_result.generation_info,
                    messages=[
                        e.data.get("message")
                        if e and e.kind == EventKind.MESSAGE and isinstance(e.data, dict)
                        else None
                        for e in event_generation_result.events
                    ],
                )
            )

        return message_generation

    async def _emit_error_event(self, context: EngineContext, exception_details: str) -> None:
        await context.session_event_emitter.emit_status_event(
            trace_id=self._tracer.trace_id,
            data={
                "status": "error",
                "data": {"exception": exception_details},
            },
        )

    async def _emit_acknowledgement_event(self, context: EngineContext) -> None:
        await context.session_event_emitter.emit_status_event(
            trace_id=self._tracer.trace_id,
            data={
                "status": "acknowledged",
                "data": {},
            },
        )

    async def _emit_processing_event(self, context: EngineContext, stage: str) -> None:
        await context.session_event_emitter.emit_status_event(
            trace_id=self._tracer.trace_id,
            data={
                "status": "processing",
                "data": {"stage": stage},
            },
        )

    async def _emit_cancellation_event(self, context: EngineContext) -> None:
        await context.session_event_emitter.emit_status_event(
            trace_id=self._tracer.trace_id,
            data={
                "status": "cancelled",
                "data": {},
            },
        )

    async def _call_guideline_handlers(
        self,
        context: EngineContext,
        handlers: dict[
            GuidelineId, list[Callable[[EngineContext, GuidelineMatch], Awaitable[None]]]
        ],
    ) -> None:
        """Call handlers for all matched guidelines.

        Args:
            context: The engine context
            handlers: Dict mapping GuidelineId to list of handlers to call
        """
        all_guideline_matches = list(
            chain(
                context.state.ordinary_guideline_matches,
                context.state.tool_enabled_guideline_matches,
            )
        )

        handler_tasks = [
            handler(context, match)
            for match in all_guideline_matches
            if match.guideline.id in handlers
            for handler in handlers[match.guideline.id]
        ]

        if handler_tasks:
            await async_utils.safe_gather(*handler_tasks)

    async def _emit_ready_event(self, context: EngineContext) -> None:
        await context.session_event_emitter.emit_status_event(
            trace_id=self._tracer.trace_id,
            data={
                "status": "ready",
                "data": {},
            },
        )

    def _get_message_composer(self, agent: Agent) -> MessageEventComposer:
        # Each agent may use a different composition mode,
        # and, moreover, the same agent can change composition
        # modes every now and then. This makes sure that we are
        # composing the message using the right mechanism for this agent.
        match agent.composition_mode:
            case CompositionMode.FLUID:
                return self._fluid_message_generator
            case (
                CompositionMode.CANNED_STRICT
                | CompositionMode.CANNED_COMPOSITED
                | CompositionMode.CANNED_FLUID
            ):
                return self._canned_response_generator

        raise Exception("Unsupported agent composition mode")

    async def _load_context_variables(
        self,
        context: EngineContext,
    ) -> list[tuple[ContextVariable, ContextVariableValue]]:
        variables_supported_by_agent = (
            await self._entity_queries.find_context_variables_for_context(
                agent_id=context.agent.id,
            )
        )

        result = []

        keys_to_check_in_order_of_importance = (
            [context.customer.id]  # Customer-specific value
            + [f"tag:{tag_id}" for tag_id in context.customer.tags]  # Tag-specific value
            + [ContextVariableStore.GLOBAL_KEY]  # Global value
        )

        # TODO: Parallelize this, as some tool-enabled context vars
        # might run long-running tasks. One example we've encountered
        # is analyzing an image and putting the analysis into a variable.
        for variable in variables_supported_by_agent:
            # Try keys in order of importance, stopping at and using
            # the first (and most important) set key for each variable.
            for key in keys_to_check_in_order_of_importance:
                if value := await self._load_context_variable_value(context, variable, key):
                    result.append((variable, value))
                    break

        return result

    async def _capture_tool_preexecution_state(
        self, context: EngineContext
    ) -> ToolPreexecutionState:
        return await self._tool_event_generator.create_preexecution_state(
            context.session_event_emitter,
            context.session.id,
            context.agent,
            context.customer,
            context.state.context_variables,
            context.interaction.events,
            list(context.state.glossary_terms),
            context.state.ordinary_guideline_matches,
            context.state.tool_enabled_guideline_matches,
            context.state.tool_events,
        )

    def _add_tool_events_to_tracer(
        self,
        tool_events: Sequence[EmittedEvent],
    ) -> None:
        for tool_event in tool_events:
            tool_calls = cast(ToolEventData, tool_event.data)["tool_calls"]
            for tool_call in tool_calls:
                self._tracer.add_event(
                    "tc",
                    attributes={
                        "tool_id": tool_call["tool_id"],
                        "arguments": json.dumps(tool_call["arguments"]),
                        "result": json.dumps(tool_call["result"]),
                    },
                )

    def _add_match_events_to_tracer(
        self,
        matches: Sequence[GuidelineMatch],
    ) -> None:
        for match in matches:
            if match.guideline.metadata.get("journey_node"):
                self._tracer.add_event(
                    "journey.state.activate",
                    attributes={
                        "node_id": extract_node_id_from_journey_node_guideline_id(
                            match.guideline.id
                        ),
                        "condition": match.guideline.content.condition,
                        "action": match.guideline.content.action or "",
                        "rationale": match.rationale,
                        "journey_id": cast(
                            str,
                            cast(
                                dict[str, JSONSerializable],
                                match.guideline.metadata["journey_node"],
                            )["journey_id"],
                        ),
                        **(
                            {
                                "sub_journey_id": cast(
                                    str,
                                    cast(
                                        dict[str, JSONSerializable],
                                        match.guideline.metadata["journey_node"],
                                    )["sub_journey_id"],
                                )
                            }
                            if "sub_journey_id"
                            in cast(
                                dict[str, JSONSerializable],
                                match.guideline.metadata["journey_node"],
                            )
                            else {}
                        ),
                    },
                )

            else:
                self._tracer.add_event(
                    "gm.activate",
                    attributes={
                        "guideline_id": match.guideline.id,
                        "condition": match.guideline.content.condition,
                        "action": match.guideline.content.action or "",
                        "rationale": match.rationale,
                    },
                )

    async def _load_matched_guidelines_and_journeys(
        self,
        context: EngineContext,
    ) -> _GuidelineAndJourneyMatchingResult:
        # Step 1: Retrieve the journeys likely to be activated for this agent
        available_journeys = await self._entity_queries.finds_journeys_for_context(
            agent_id=context.agent.id,
        )

        # Step 2 : Retrieve all the guidelines for the context.
        all_stored_guidelines = {
            g.id: g
            for g in await self._entity_queries.find_guidelines_for_context(
                agent_id=context.agent.id,
                journeys=available_journeys,
            )
            if g.enabled
        }

        # Step 3: Exclude guidelines whose prerequisite journeys are less likely to be activated
        # (everything beyond the first `top_k` journeys), and also remove all journey graph guidelines.
        # Removing these guidelines
        # matching pass fast and focused on the most likely flows.
        top_k = 1
        (
            relevant_guidelines,
            high_prob_journeys,
        ) = await self._prune_low_prob_guidelines_and_all_graph(
            context,
            available_journeys=list(available_journeys),
            all_stored_guidelines=all_stored_guidelines,
            top_k=top_k,
        )

        # Step 4: Filter the best matches out of those.
        with self._tracer.span(_GUIDELINE_MATCHER_SPAN_NAME, attributes={"phase": "initial"}):
            matching_result = await self._guideline_matcher.match_guidelines(
                context=context,
                active_journeys=high_prob_journeys,  # Only consider the top K journeys
                guidelines=relevant_guidelines,
            )

        self._add_match_events_to_tracer(matching_result.matches)

        # Step 5: Filter the journeys that are activated by the matched guidelines.
        activated_journeys = self._filter_activated_journeys(
            context, matching_result.matches, available_journeys
        )

        # Step 6: If any of the lower-probability journeys (those originally filtered out)
        # have in fact been activated, run an additional matching pass for the guidelines
        # that depend on them so we don’t miss relevant behavior.
        if second_match_result := await self._process_activated_low_probability_journey_guidelines(
            context=context,
            all_stored_guidelines=all_stored_guidelines,
            high_prob_journeys=high_prob_journeys,
            activated_journeys=activated_journeys,
        ):
            batches = list(chain(matching_result.batches, second_match_result.batches))
            matches = list(chain.from_iterable(batches))

            matching_result = GuidelineMatchingResult(
                total_duration=matching_result.total_duration + second_match_result.total_duration,
                batch_count=matching_result.batch_count + second_match_result.batch_count,
                batch_generations=list(
                    chain(
                        matching_result.batch_generations,
                        second_match_result.batch_generations,
                    )
                ),
                batches=batches,
                matches=matches,
            )

            self._add_match_events_to_tracer(second_match_result.matches)

        # Step 7: Build the set of matched guidelines:
        matched_guidelines = await self._build_matched_guidelines(
            context=context,
            evaluated_guidelines=relevant_guidelines,
            current_matched=set(matching_result.matches),
            active_journeys=activated_journeys,
        )

        # Step 8: Resolve guideline matches by loading related guidelines that may not have
        # been inferrable just by looking at the interaction.
        all_relevant_guidelines = await self._relational_guideline_resolver.resolve(
            usable_guidelines=list(all_stored_guidelines.values()),
            matches=matched_guidelines,
            journeys=activated_journeys,
        )

        return _GuidelineAndJourneyMatchingResult(
            matching_result=matching_result,
            matches_guidelines=list(matching_result.matches),
            resolved_guidelines=list(all_relevant_guidelines),
            journeys=activated_journeys,
        )

    async def _load_additional_matched_guidelines_and_journeys(
        self,
        context: EngineContext,
    ) -> _GuidelineAndJourneyMatchingResult:
        # Step 1: Retrieve all the possible journeys for this agent
        all_journeys = await self._entity_queries.finds_journeys_for_context(
            agent_id=context.agent.id,
        )

        # Step 2 : Retrieve all the guidelines for this agent the journeys that are enabled
        all_stored_guidelines = {
            g.id: g
            for g in await self._entity_queries.find_guidelines_for_context(
                agent_id=context.agent.id,
                journeys=all_journeys,
            )
            if g.enabled
        }

        # Step 3: Retrieve guidelines that need reevaluation based on tool calls made
        # in case no guidelines need reevaluation, we can skip the rest of the steps.
        guidelines_to_reevaluate = (
            await self._entity_queries.find_guidelines_that_need_reevaluation(
                all_stored_guidelines,
                context.state.journeys,
                tool_insights=context.state.iterations[-1].tool_insights,
            )
        )

        # Step 4: Reevaluate those guidelines using the latest context.
        with self._tracer.span(_GUIDELINE_MATCHER_SPAN_NAME, attributes={"phase": "reevaluation"}):
            matching_result = await self._guideline_matcher.match_guidelines(
                context=context,
                active_journeys=context.state.journeys,
                guidelines=guidelines_to_reevaluate,
            )

        self._add_match_events_to_tracer(matching_result.matches)

        # Step 5: Filter out the journeys activated by the matched guidelines.
        # If a journey was already active in a previous guideline-matching iteration, we still retrieve it
        # so we can exclude it from the next guideline-matching iteration.
        activated_journeys = self._filter_activated_journeys_for_advanced_iterations(
            matching_result.matches,
            all_journeys,
        )

        # Step 6: If any of the journeys have been activated,
        # run an additional matching pass for the guidelines
        # that depend on them so we don’t miss relevant behavior.
        if second_match_result := await self._match_dependent_guidelines_and_active_journeys(
            context=context,
            all_stored_guidelines=all_stored_guidelines,
            already_examined_guidelines={g.id for g in guidelines_to_reevaluate},
            activated_journeys=activated_journeys,
        ):
            batches = list(chain(matching_result.batches, second_match_result.batches))
            matches = list(chain.from_iterable(batches))

            matching_result = GuidelineMatchingResult(
                total_duration=matching_result.total_duration + second_match_result.total_duration,
                batch_count=matching_result.batch_count + second_match_result.batch_count,
                batch_generations=list(
                    chain(
                        matching_result.batch_generations,
                        second_match_result.batch_generations,
                    )
                ),
                batches=batches,
                matches=matches,
            )
            self._add_match_events_to_tracer(second_match_result.matches)

        # Step 7: Build the final set of matched guidelines:
        all_activated_journeys = list(set(context.state.journeys + activated_journeys))

        matched_guidelines = await self._build_matched_guidelines(
            context=context,
            evaluated_guidelines=guidelines_to_reevaluate,
            current_matched=set(matching_result.matches),
            active_journeys=all_activated_journeys,
        )

        all_relevant_guidelines = await self._relational_guideline_resolver.resolve(
            usable_guidelines=list(all_stored_guidelines.values()),
            matches=list(matched_guidelines),
            journeys=all_activated_journeys,
        )

        return _GuidelineAndJourneyMatchingResult(
            matching_result=matching_result,
            matches_guidelines=list(matching_result.matches),
            resolved_guidelines=list(all_relevant_guidelines),
            journeys=activated_journeys,
        )

    def _list_journey_paths(
        self,
        context: EngineContext,
    ) -> dict[JourneyId, list[Optional[str]]]:
        journey_paths = copy.deepcopy(context.state.journey_paths)

        new_journey_paths = self._list_journey_paths_from_guideline_matches(context)

        for journey_id, path in new_journey_paths.items():
            journey_paths[journey_id] = path

        return journey_paths

    def _filter_activated_journeys(
        self,
        context: EngineContext,
        matches: Sequence[GuidelineMatch],
        all_journeys: Sequence[Journey],
    ) -> list[Journey]:
        # We consider a journey to be activated if either:
        # 1. Journey was activated before and match return a journey path with a step that is not None.
        # 2. The journey’s conditions match any of the currently matched guideline IDs.
        journeys_with_paths: set[JourneyId] = {
            id
            for id, j in context.state.journey_paths.items()
            if context.state.journey_paths[id] != [None]
        }

        active_journey_ids_by_path = {
            m.metadata.get("step_selection_journey_id")
            for m in matches
            if m.metadata.get("journey_path", [])
            and cast(list[GuidelineId], m.metadata["journey_path"])[-1] is not None
            and m.metadata.get("step_selection_journey_id") in journeys_with_paths
        }

        active_journeys_by_conditions = [
            j
            for j in all_journeys
            if set(j.conditions).intersection({m.guideline.id for m in matches})
        ]

        active_journeys = list(
            set(
                active_journeys_by_conditions
                + [j for j in all_journeys if j.id in active_journey_ids_by_path]
            )
        )

        return active_journeys

    def _filter_activated_journeys_for_advanced_iterations(
        self,
        matches: Sequence[GuidelineMatch],
        all_journeys: Sequence[Journey],
    ) -> list[Journey]:
        # We consider a journey to be activated if either:
        # 1. Match return a journey path with a step that is not None for journey that .
        # 2. The journey’s conditions match any of the currently matched guideline IDs.
        active_journeys_by_conditions = [
            j
            for j in all_journeys
            if set(j.conditions).intersection({m.guideline.id for m in matches})
        ]

        active_journey_ids_by_path = {
            m.metadata.get("step_selection_journey_id")
            for m in matches
            if m.metadata.get("journey_path", [])
            and cast(list[GuidelineId], m.metadata["journey_path"])[-1] is not None
        }

        active_journeys = list(
            set(
                active_journeys_by_conditions
                + [j for j in all_journeys if j.id in active_journey_ids_by_path]
            )
        )

        return active_journeys

    async def _build_matched_guidelines(
        self,
        context: EngineContext,
        evaluated_guidelines: Sequence[Guideline],
        current_matched: set[GuidelineMatch],
        active_journeys: Sequence[Journey],
    ) -> Sequence[GuidelineMatch]:
        # Build the set of matched guidelines as follows:
        # 1. Collect all previously matched guidelines (from earlier iterations if were) — call this set (1).
        # 2. Collect the newly matched guidelines from the current iteration — call this set (2).
        #
        # For each guideline:
        # - If it was ACTIVE in (1) and is still ACTIVE in (2), include it.
        # - If it was INACTIVE in (1) and became ACTIVE in (2), include it.
        # - If it was ACTIVE in (1), was re-evaluated, and is now INACTIVE in (2), exclude it.
        #
        # - For each journey, keep only the last matched guideline associated with that journey.
        #   (This assumes matches are ordered.)
        #
        # The goal is to determine the currently relevant guidelines, considering for both continuity and change.
        # After filtering, RESOLVE this updated group of matched guidelines to handle:
        # 1. Cases where a guideline just became ACTIVE and may take priority over other ACTIVE guidelines.
        # 2. Cases where a previously ACTIVE guideline became INACTIVE — we may need to re-prioritize those it previously suppressed.
        latest_match_per_journey: dict[JourneyId, Optional[GuidelineId]] = {
            journey.id: None for journey in active_journeys
        }
        filtered_out_matches: set[GuidelineId] = set()
        result: dict[GuidelineId, GuidelineMatch] = {}

        previous_matches = list(
            OrderedDict.fromkeys(
                chain.from_iterable(
                    iteration.matched_guidelines for iteration in context.state.iterations
                )
            )
        )

        reevaluated_guideline_ids = {g.id for g in evaluated_guidelines}

        combined: OrderedDict[GuidelineMatch, None] = OrderedDict()

        for match in previous_matches:
            if match in current_matched:
                combined[match] = None
            elif match.guideline.id not in reevaluated_guideline_ids:
                combined[match] = None

        for match in current_matched:
            combined[match] = None

        for match in combined.keys():
            if journey_id := match.metadata.get("step_selection_journey_id"):
                journey_id = cast(JourneyId, journey_id)

                if journey_id not in latest_match_per_journey:
                    filtered_out_matches.add(match.guideline.id)
                    continue  # Skip if the journey is not in the active journeys

                if (
                    latest_match_per_journey[journey_id] is not None
                    and latest_match_per_journey[journey_id] != match.guideline.id
                ):
                    filtered_out_matches.add(
                        cast(GuidelineId, latest_match_per_journey[journey_id])
                    )

                latest_match_per_journey[journey_id] = match.guideline.id

        for m in combined.keys():
            if m.guideline.id not in filtered_out_matches:
                result[m.guideline.id] = m

        return list(result.values())

    async def _find_tool_enabled_guideline_matches(
        self,
        guideline_matches: Sequence[GuidelineMatch],
    ) -> dict[GuidelineMatch, list[ToolId]]:
        # Create a convenient accessor dict for tool-enabled guidelines (and their tools).
        # This allows for optimized control and data flow in the engine.

        guideline_tool_associations = list(
            await self._entity_queries.find_guideline_tool_associations()
        )
        guideline_matches_by_id = {p.guideline.id: p for p in guideline_matches}

        relevant_associations = [
            a for a in guideline_tool_associations if a.guideline_id in guideline_matches_by_id
        ]

        tools_for_guidelines: dict[GuidelineMatch, list[ToolId]] = defaultdict(list)

        for association in relevant_associations:
            tools_for_guidelines[guideline_matches_by_id[association.guideline_id]].append(
                association.tool_id
            )

        # Fetch node tool associations
        node_guidelines = [
            m.guideline for m in guideline_matches if m.guideline.id.startswith("journey_node:")
        ]

        node_tools_associations = {
            guideline_matches_by_id[g.id]: list(tools)
            for g, tools in zip(
                node_guidelines,
                await async_utils.safe_gather(
                    *[
                        self._entity_queries.find_journey_node_tool_associations(
                            extract_node_id_from_journey_node_guideline_id(g.id),
                        )
                        for g in node_guidelines
                    ]
                ),
            )
            if tools
        }

        tools_for_guidelines.update(node_tools_associations)

        return dict(tools_for_guidelines)

    async def _prune_low_prob_guidelines_and_all_graph(
        self,
        context: EngineContext,
        available_journeys: list[Journey],
        all_stored_guidelines: dict[GuidelineId, Guideline],
        top_k: int,
    ) -> tuple[list[Guideline], list[Journey]]:
        # High-level algorithm:
        #
        # 1. If we have journey paths in the context:
        #    We send *all* journeys that appear in those paths. These journeys are either:
        #      • currently active, or
        #      • already finished but may need to be resumed or re-activated.
        #
        #    For active journeys, we assume the next user message is highly likely
        #    to continue the journey. For finished journeys, the journey-node
        #    selection logic determines whether we should:
        #        • jump back into a specific node that reactivates the journey, or
        #        • start the journey over again from the beginning.
        #
        #    In this case, we do *not* need to re-rank by semantic relevance:
        #    the journey paths already encode the highest-probability journeys.
        #
        # 2. If no journeys are currently active (no journey paths):
        #    We fall back to semantic relevance:
        #      • sort all available journeys by relevance to the current context, and
        #      • take the top `top_k` journeys as the high-probability candidates.
        #
        #    This is the only case where we pay the embedding cost, since we have
        #    no strong signal from prior interactions about which journey is most
        #    likely to be active next.
        #
        # 3. Edge cases for `top_k` handling:
        #      • If the number of previously-active journeys exceeds `top_k`,
        #        keep all of their guidelines.
        #      • If there are fewer than `top_k` active journeys (X where 0 ≤ X < top_k),
        #        supplement them with the top `(top_k - X)` most relevant journeys
        #        from the remaining `relevant_journeys`.
        #
        # 4. Guideline pruning:
        #      • Collect guideline IDs related to all journeys we decided to keep.
        #      • Build a pruned guideline list that:
        #          – keeps guidelines whose IDs belong to those high-probability journeys, and
        #          – also keeps guidelines that are *not* tied to any journey at all.
        #
        # The result is a focused set of high-probability guidelines that:
        #   • favors journey continuity when we already know which journeys are active/finished,
        #   • falls back to top-`k` relevance when no journeys are active, and
        #   • avoids unnecessary embedding cost whenever possible.
        journey_paths = context.state.journey_paths or {}

        # Journeys that appear in journey_paths are considered "known" journeys:
        # either active or finished (but still relevant and potentially reactivatable).
        # A journey path can be [None] if we assumed the journey would be active, but the
        # journey-node selection did not select any node for it—meaning it was not active in the past.
        journeys_with_paths_ids: set[JourneyId] = set(journey_paths.keys())
        journeys_with_paths: list[Journey] = [
            j
            for j in available_journeys
            if j.id in journeys_with_paths_ids and journey_paths[j.id] != [None]
        ]

        # Decide which journeys are "high probability"
        if journeys_with_paths:
            # There *are* journeys with paths:
            #   • If their count exceeds `top_k`, keep all of them.
            #   • If fewer than `top_k`, supplement with the most relevant remaining journeys.
            if len(journeys_with_paths) >= top_k:
                high_prob_journeys = journeys_with_paths
            else:
                sorted_journeys_by_relevance = await self._sort_journeys_by_relevance(
                    context, available_journeys
                )

                supplemental_journeys: list[Journey] = []
                for journey in sorted_journeys_by_relevance:
                    if journey.id in journeys_with_paths_ids:
                        continue
                    supplemental_journeys.append(journey)
                    if len(journeys_with_paths) + len(supplemental_journeys) >= top_k:
                        break

                high_prob_journeys = journeys_with_paths + supplemental_journeys
        else:
            # No journeys were active/finished (no journey paths):
            # fall back to semantic relevance and take the top_k journeys.
            sorted_journeys_by_relevance = await self._sort_journeys_by_relevance(
                context, available_journeys
            )
            high_prob_journeys = sorted_journeys_by_relevance[:top_k]

        # Build a single cache of guideline IDs per journey for all available journeys.
        journey_to_guideline_ids: dict[JourneyId, set[GuidelineId]] = {}
        for journey in available_journeys:
            journey_to_guideline_ids[journey.id] = set(
                await self._entity_queries.find_journey_related_guidelines(journey)
            )

        # All guideline IDs that are tied to any *available* journey.
        available_journeys_related_ids: set[GuidelineId] = (
            set().union(*journey_to_guideline_ids.values()) if journey_to_guideline_ids else set()
        )

        # Guideline IDs related specifically to the high-probability journeys.
        high_prob_journey_related_ids: set[GuidelineId] = set()
        for journey in high_prob_journeys:
            high_prob_journey_related_ids.update(journey_to_guideline_ids.get(journey.id, set()))

        pruned_guidelines = [
            g
            for guideline_id, g in all_stored_guidelines.items()
            if (
                guideline_id in high_prob_journey_related_ids
                or guideline_id not in available_journeys_related_ids
            )
        ]

        return pruned_guidelines, high_prob_journeys

    async def _process_activated_low_probability_journey_guidelines(
        self,
        context: EngineContext,
        all_stored_guidelines: dict[GuidelineId, Guideline],
        high_prob_journeys: Sequence[Journey],
        activated_journeys: Sequence[Journey],
    ) -> Optional[GuidelineMatchingResult]:
        activated_low_prob_related_ids = set(
            chain.from_iterable(
                [
                    await self._entity_queries.find_journey_related_guidelines(j)
                    for j in [
                        activated_journey
                        for activated_journey in activated_journeys
                        if activated_journey not in high_prob_journeys
                    ]
                ]
            )
        )

        if activated_low_prob_related_ids:
            journey_conditions = list(
                chain.from_iterable([j.conditions for j in activated_journeys if j.conditions])
            )

            additional_matching_guidelines = [
                g
                for id, g in all_stored_guidelines.items()
                if id in activated_low_prob_related_ids or id in journey_conditions
            ]

            with self._tracer.span(
                _GUIDELINE_MATCHER_SPAN_NAME, attributes={"phase": "low_probability_journeys"}
            ):
                return await self._guideline_matcher.match_guidelines(
                    context=context,
                    active_journeys=activated_journeys,
                    guidelines=additional_matching_guidelines,
                )

        return None

    async def _match_dependent_guidelines_and_active_journeys(
        self,
        context: EngineContext,
        all_stored_guidelines: dict[GuidelineId, Guideline],
        already_examined_guidelines: set[GuidelineId],
        activated_journeys: Sequence[Journey],
    ) -> Optional[GuidelineMatchingResult]:
        related_guidelines = list(
            chain.from_iterable(
                [
                    await self._entity_queries.find_journey_related_guidelines(j)
                    for j in [activated_journey for activated_journey in activated_journeys]
                ]
            )
        )

        if related_guidelines:
            additional_matching_guidelines = [
                g for id, g in all_stored_guidelines.items() if id in related_guidelines
            ]

            filtered_guidelines = [
                g for g in additional_matching_guidelines if g.id not in already_examined_guidelines
            ]

            with self._tracer.span(
                _GUIDELINE_MATCHER_SPAN_NAME,
                attributes={"phase": "reevaluated_dependent_guidelines"},
            ):
                return await self._guideline_matcher.match_guidelines(
                    context=context,
                    active_journeys=activated_journeys,
                    guidelines=filtered_guidelines,
                )

        return None

    async def _load_capabilities(self, context: EngineContext) -> Sequence[Capability]:
        # Capabilities are retrieved using semantic similarity.
        # The querying process is done with a text query, for which
        # the K most relevant terms are retrieved.
        #
        # We thus build an optimized query here based on our context.
        query = ""

        if context.interaction.events:
            query += str([e.data for e in context.interaction.events])

        if query:
            return await self._entity_queries.find_capabilities_for_agent(
                agent_id=context.agent.id,
                query=query,
                max_count=3,
            )

        return []

    async def _load_glossary_terms(self, context: EngineContext) -> Sequence[Term]:
        # Glossary terms are retrieved using semantic similarity.
        # The querying process is done with a text query, for which
        # the K most relevant terms are retrieved.
        #
        # We thus build an optimized query here based on our context and state.
        query = ""

        if context.state.context_variables:
            query += f"\n{context_variables_to_json(context.state.context_variables)}"

        if context.interaction.events:
            query += str([e.data for e in context.interaction.events])

        if context.state.guidelines:
            query += str(
                [
                    f"When {g.content.condition}, then {g.content.action}"
                    if g.content.action
                    else f"When {g.content.condition}"
                    for g in context.state.guidelines
                ]
            )

        if context.state.tool_events:
            query += str([e.data for e in context.state.tool_events])

        if query:
            return await self._entity_queries.find_glossary_terms_for_context(
                agent_id=context.agent.id,
                query=query,
            )

        return []

    async def _sort_journeys_by_relevance(
        self,
        context: EngineContext,
        relevant_journeys: Sequence[Journey],
    ) -> list[Journey]:
        # Journeys are retrieved using semantic similarity.
        # The querying process is done with a text query
        #
        # We thus build an optimized query here based on our context and state.
        query = ""

        if context.state.context_variables:
            query += f"\n{context_variables_to_json(context.state.context_variables)}"

        if context.state.glossary_terms:
            query += str([t.name for t in context.state.glossary_terms])

        if context.interaction.events:
            query += str([e.data for e in context.interaction.events])

        if query:
            return list(
                await self._entity_queries.sort_journeys_by_contextual_relevance(
                    available_journeys=relevant_journeys,
                    query=query,
                )
            )

        return []

    async def _call_tools(
        self,
        context: EngineContext,
        preexecution_state: ToolPreexecutionState,
    ) -> tuple[ToolEventGenerationResult, list[EmittedEvent], ToolInsights] | None:
        with self._tracer.span(_TOOL_CALLER_SPAN_NAME):
            result = await self._tool_event_generator.generate_events(preexecution_state, context)

        tool_events = [e for e in result.events if e] if result else []

        return result, tool_events, result.insights

    async def _utterance_requests_to_guideline_matches(
        self,
        requests: Sequence[UtteranceRequest],
    ) -> Sequence[GuidelineMatch]:
        # Utterance requests are reduced to guidelines, to take advantage
        # of the engine's ability to consistently adhere to guidelines.

        def utterance_request_to_match(
            i: int,
            utterance_request: UtteranceRequest,
        ) -> GuidelineMatch:
            rationales = {
                UtteranceRationale.UNSPECIFIED: "An external module has determined that this response is necessary, and you must adhere to it.",
                UtteranceRationale.BUY_TIME: "You must buy time while you're working on a task in the background.",
                UtteranceRationale.FOLLOW_UP: "You need to follow up with the customer.",
            }

            return GuidelineMatch(
                guideline=Guideline(
                    id=GuidelineId(f"<canrep-request-{i}>"),
                    creation_utc=datetime.now(timezone.utc),
                    content=GuidelineContent(
                        condition="",  # FIXME: Change this to None when we support `str | None` conditions
                        action=utterance_request.action,
                    ),
                    criticality=Criticality.MEDIUM,
                    enabled=True,
                    tags=[],
                    metadata={},
                ),
                rationale=rationales[utterance_request.rationale],
                score=10,
            )

        return [
            utterance_request_to_match(i, request) for i, request in enumerate(requests, start=1)
        ]

    async def _load_context_variable_value(
        self,
        context: EngineContext,
        variable: ContextVariable,
        key: str,
    ) -> Optional[ContextVariableValue]:
        return await load_fresh_context_variable_value(
            entity_queries=self._entity_queries,
            entity_commands=self._entity_commands,
            agent_id=context.agent.id,
            session=context.session,
            variable=variable,
            key=key,
        )

    async def _filter_problematic_tool_parameters_based_on_precedence(
        self, problematic_parameters: Sequence[ProblematicToolData]
    ) -> Sequence[ProblematicToolData]:
        precedence_values = [
            m.precedence for m in problematic_parameters if m.precedence is not None
        ]

        if precedence_values == []:
            return problematic_parameters

        return [m for m in problematic_parameters if m.precedence == min(precedence_values)]

    def _todo_add_associated_guidelines(self, guideline_matches: Sequence[GuidelineMatch]) -> None:
        # TODO write this method - it should add guidelines that are associated with the previously matched guidelines (due to having similar actions, as flagged by the conversation designer)
        return

    async def _add_agent_state(
        self,
        context: EngineContext,
        session: Session,
        guideline_matches: Sequence[GuidelineMatch],
    ) -> None:
        applied_guideline_ids = (
            list(session.agent_states[-1].applied_guideline_ids) if session.agent_states else []
        )

        matches_to_analyze = [
            match
            for match in guideline_matches
            if match.guideline.id not in applied_guideline_ids
            and not match.guideline.metadata.get("continuous", False)
            and match.guideline.content.action
            and "journey_node" not in match.guideline.metadata  # Exclude journey node guidelines
            and not match.guideline.id.startswith("<transient")  # Exclude transient guidelines
            and not match.guideline.criticality
            != Criticality.LOW  # Exclude low criticality guidelines
        ]

        self._todo_add_associated_guidelines(matches_to_analyze)

        with self._tracer.span(_RESPONSE_ANALYSIS_SPAN_NAME):
            result = await self._guideline_matcher.analyze_response(
                agent=context.agent,
                session=session,
                customer=context.customer,
                context_variables=context.state.context_variables,
                interaction_history=context.interaction.events,
                terms=list(context.state.glossary_terms),
                staged_tool_events=context.state.tool_events,
                staged_message_events=context.state.message_events,
                guideline_matches=matches_to_analyze,
            )

        new_applied_guideline_ids = [
            a.guideline.id for a in result.analyzed_guidelines if a.is_previously_applied
        ]

        applied_guideline_ids.extend(new_applied_guideline_ids)

        await self._entity_commands.update_session(
            session_id=session.id,
            params=SessionUpdateParamsModel(
                agent_states=list(session.agent_states)
                + [
                    AgentState(
                        trace_id=self._tracer.trace_id,
                        applied_guideline_ids=applied_guideline_ids,
                        journey_paths=context.state.journey_paths,
                    )
                ]
            ),
        )

    def _list_journey_paths_from_guideline_matches(
        self,
        context: EngineContext,
    ) -> dict[JourneyId, list[Optional[str]]]:
        # 1. Iterate over all guideline matches:
        #       • If a `journey_id` is found in the matched guideline metadata:
        #             – Remove that journey from the `journeys` set, since it
        #               successfully matched a guideline. This also ensures we catch the
        #               unexpected case where multiple matches appear for the same journey.
        #             – Validate that `journey_path` metadata exists on the match.
        #               If missing, log an error and skip.
        #             – Store the extracted path as:
        #                   journey_paths[journey_id] = <list[GuidelineId | None]>
        #
        #             – If the matched guideline represents the *root* journey node:
        #                   • Treat it as a placeholder and insert `None` into the path.
        #                   • Remove the root-node guideline ID from the returned path
        #                     (root guidelines have empty content and do not represent
        #                     actionable journey steps).
        #
        #
        #       • If no `journey_id` can be resolved from the match metadata:
        #             – Skip this match.
        #
        # 2. After processing all matches, any remaining journeys in the `journeys`
        #    set did *not* match any node guideline. Assign:
        #         journey_paths[journey_id] = [None]
        #    This indicates that the journey is inactive for the current interaction.
        guideline_matches = list(
            chain(
                context.state.ordinary_guideline_matches,
                context.state.tool_enabled_guideline_matches,
            )
        )

        journeys = {j.id: j for j in context.state.journeys}
        journey_paths: dict[JourneyId, list[Optional[str]]] = {}

        for match in guideline_matches:
            # Validate that this guideline belongs to a journey-node
            node_metadata = cast(
                dict[str, JSONSerializable], match.guideline.metadata.get("journey_node", {})
            )
            if not node_metadata:
                continue

            journey_id = cast(JourneyId, node_metadata.get("journey_id"))
            if not journey_id:
                continue

            # Remove journey ID so we can detect unmatched journeys afterwards
            journey = journeys.pop(journey_id, None)
            if journey is None:
                # This means journey matched twice → unexpected behavior
                self._logger.error(
                    f"Multiple guideline-node matches found for journey {journey_id}. Match: {match}"
                )
                continue

            # Validate required metadata exists
            if "journey_path" not in match.metadata:
                self._logger.error(
                    f"Journey path not found in guideline journey-node match metadata. Match: {match}"
                )
                continue

            path = cast(list[Optional[str]], match.metadata.get("journey_path"))

            # Detect whether this guideline is the root node
            # root node are placeholder for exit the journey
            # since they have no content, will be deleted from the guideline matches as well
            # we only look it in ordinary guidelines since root nodes cannot have tools attached
            if journey.root_id == extract_node_id_from_journey_node_guideline_id(
                match.guideline.id
            ):
                for i, m in enumerate(context.state.ordinary_guideline_matches):
                    if m.guideline.id == match.guideline.id:
                        del context.state.ordinary_guideline_matches[i]
                        break

            journey_paths[journey_id] = path

        # Any journey still in `journeys` received *no* match → inactive
        for journey_id in journeys:
            journey_paths[journey_id] = [None]

        return journey_paths


# This is module-level and public for isolated testability purposes.
async def load_fresh_context_variable_value(
    entity_queries: EntityQueries,
    entity_commands: EntityCommands,
    agent_id: AgentId,
    session: Session,
    variable: ContextVariable,
    key: str,
    current_time: datetime = datetime.now(timezone.utc),
) -> Optional[ContextVariableValue]:
    # Load the existing value
    value = await entity_queries.read_context_variable_value(
        variable_id=variable.id,
        key=key,
    )

    # If there's no tool attached to this variable,
    # return the value we found for the key.
    # Note that this may be None here, which is okay.
    if not variable.tool_id:
        return value

    # So we do have a tool attached.
    # Do we already have a value, and is it sufficiently fresh?
    if value and variable.freshness_rules:
        cron_iterator = croniter(variable.freshness_rules, value.last_modified)

        if cron_iterator.get_next(datetime) > current_time:
            # We already have a fresh value in store. Return it.
            return value

    # We don't have a sufficiently fresh value.
    # Get an updated one, utilizing the associated tool.

    tool_context = ToolContext(
        agent_id=agent_id,
        session_id=session.id,
        customer_id=session.customer_id,
    )

    tool_service = await entity_queries.read_tool_service(variable.tool_id.service_name)

    tool_result = await tool_service.call_tool(
        variable.tool_id.tool_name,
        context=tool_context,
        arguments={},
    )

    return await entity_commands.update_context_variable_value(
        variable_id=variable.id,
        key=key,
        data=tool_result.data,
    )
