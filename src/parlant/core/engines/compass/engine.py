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

from collections.abc import Mapping, Sequence
import asyncio
import hashlib
import time
from itertools import chain
import traceback
from typing import cast
from typing_extensions import override

from parlant.core.common import JSONSerializable
from parlant.core.diagnostic_reports import (
    DIAGNOSTIC_REPORT_URI_METADATA_KEY,
    capture_engine_exception,
    format_exception,
)
from parlant.core.cost_control import CostContext, CostControlPolicy, WorkKind
from parlant.core.pricing import PricingConfigurationError, SessionBudgetService
from parlant.core.async_utils import (
    CancellationSuppressionLatch,
    Stopwatch,
    delay,
    latched_shield,
    safe_gather,
)
from parlant.core.health import (
    ENGINE_TURN_KIND,
    ENGINE_TURNS_COUNTER,
    EngineHealthView,
    HealthReporter,
)
from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.emissions import EventEmitter
from parlant.core.engines.entity_context import EntityContext
from parlant.core.engines.session_metadata_workspace import (
    SessionMetadataWorkspace,
    WorkspaceState,
)
from parlant.core.engines.engine_context import Interaction
from parlant.core.engines.compass.compacter import Compacter
from parlant.core.engines.compass.matcher import Matcher
from parlant.core.engines.compass.responder import Responder
from parlant.core.engines.compass.context import (
    CompassDependencies,
    EngineContext,
    ResponseState,
)
from parlant.core.engines.compass.hooks import Compacted, CompassHookResult
from parlant.core.engines.compass.tracing import CompassTracer
from parlant.core.engines.compass.turn_directives import TurnDirective
from parlant.core.engines.types import (
    Context,
    Engine,
    UtteranceRationale,
    UtteranceRequest,
)
from parlant.core.entity_cq import EntityCommands, EntityQueries
from parlant.core.loggers import Logger
from parlant.core import metadata as metadata_utils
from parlant.core.meter import Meter
from parlant.core.sessions import (
    Event,
    EventKind,
    EventSource,
    MessageEventData,
    StatusEventData,
    ToolEventData,
)
from parlant.core.tracer import Tracer, TURN_TRACE_ID_ATTRIBUTE
from parlant.core.usage_reporter import UsageReporter


class CompassEngine(Engine):
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        matcher: Matcher,
        responder: Responder,
        compacter: Compacter,
        entity_queries: EntityQueries,
        entity_commands: EntityCommands,
        usage_reporter: UsageReporter,
        session_budget_service: SessionBudgetService,
        cost_control_policy: CostControlPolicy,
        health_reporter: HealthReporter,
        dependencies: CompassDependencies,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._compass_tracer = CompassTracer(tracer)
        self._meter = meter

        self._matcher = matcher
        self._responder = responder
        self._compacter = compacter

        self._entity_queries = entity_queries
        self._entity_commands = entity_commands
        self._usage_reporter = usage_reporter
        self._session_budget_service = session_budget_service
        self._cost_control_policy = cost_control_policy
        self._health_reporter = health_reporter
        self._dependencies = dependencies

    @override
    async def initialize(
        self,
        context: Context,
        event_emitter: EventEmitter,
    ) -> None:
        engine_context = await self._load_context(
            context,
            event_emitter,
            load_interaction=False,
        )

        try:
            with (
                self._logger.scope("Engine"),
                self._tracer.span(
                    "compass.engine.initialize",
                    {
                        "session_id": context.session_id,
                        "agent_id": context.agent_id,
                        "agent_last_modified": engine_context.agent.modified_utc.isoformat(),
                    },
                ),
            ):
                # Warm the provider cache for the system prompt as soon as the session
                # exists, before any message arrives. The interaction is empty here, so
                # only the stable system prefix is warmed; the real turn later reads it.
                await self._matcher.preload(engine_context)

                if await self._gate_background(engine_context):
                    await safe_gather(
                        # Warming up the matcher is important because it
                        # ensures that the N matcher LLM requests will
                        # reuse a KV cache, ultimately lowering costs.
                        self._matcher.warm_up(engine_context),
                        # Responder warm up isn't so useful actually.
                        # self._responder.warm_up(engine_context),
                    )

                await self._update_usage(engine_context)
                await self._commit_workspace(engine_context)
        except BaseException:
            await self._finalize_abandoned_workspace(engine_context)
            raise
        finally:
            EntityContext.clear(engine_context)

    @override
    async def process(
        self,
        context: Context,
        event_emitter: EventEmitter,
    ) -> bool:
        engine_context: EngineContext | None = None
        response_visible = False
        # Coarse phase marker for telemetry only (process.failed carries it);
        # advanced at the same boundaries the flow already has.
        stage = "loading_context"

        with (
            self._logger.scope("Engine"),
            self._tracer.span(
                "compass.engine.process",
                {
                    "session_id": context.session_id,
                    "agent_id": context.agent_id,
                },
            ),
            self._compass_tracer.process_scope(),
        ):
            # A new customer execution owns a new turn. Child asyncio tasks
            # capture this reference, independent of later session messages.
            self._tracer.set_attribute(TURN_TRACE_ID_ATTRIBUTE, self._tracer.trace_id)
            # Health accounting: the stopwatch feeds the turn-latency sample,
            # the monotonic stamp (set once the context is loaded) lets the
            # loop report time-to-first-message.
            turn_watch = Stopwatch.start()

            try:
                engine_context = await self._load_context(context, event_emitter)
                self._tracer.set_attribute(
                    "agent_last_modified", engine_context.agent.modified_utc.isoformat()
                )

                engine_context.state.turn_started_monotonic = time.monotonic()

                if engine_context.session.mode == "manual":
                    self._compass_tracer.process_resolved(
                        outcome="stopped",
                        stage=stage,
                        reply_emitted=response_visible,
                        stop_reason="manual_mode",
                    )
                    return await self._finish_normal_exit(engine_context, True)

                self._logger.info(f"Turn started on session {engine_context.session.id}")
                stage = "preparing"

                await self._wait_before_processing(engine_context)

                if not await engine_context.dependencies.hooks.call_on_acknowledging(
                    engine_context
                ):
                    self._compass_tracer.process_resolved(
                        outcome="stopped",
                        stage=stage,
                        reply_emitted=response_visible,
                        stop_reason="hook_bail",
                    )
                    return await self._finish_normal_exit(
                        engine_context,
                        False,
                    )  # Hook requested to bail out

                await event_emitter.emit_status_event(
                    trace_id=self._tracer.trace_id,
                    data=StatusEventData(status="acknowledged"),
                )

                if not await engine_context.dependencies.hooks.call_on_acknowledged(engine_context):
                    self._compass_tracer.process_resolved(
                        outcome="stopped",
                        stage=stage,
                        reply_emitted=response_visible,
                        stop_reason="hook_bail",
                    )
                    return await self._finish_normal_exit(
                        engine_context,
                        False,
                    )  # Hook requested to bail out

                await self._matcher.match_early(engine_context)
                await self._call_rule_handlers(engine_context)
                if await self._stop_if_session_became_manual(engine_context):
                    self._compass_tracer.process_resolved(
                        outcome="stopped",
                        stage=stage,
                        reply_emitted=response_visible,
                        stop_reason="manual_mode",
                    )
                    return await self._finish_normal_exit(engine_context, True)

                # TURN cost-control choke point: a denied turn is acknowledged and
                # terminated with a `ready` status event only (see _gate_turn).
                if not await self._gate_turn_with_cost_control(engine_context):
                    self._compass_tracer.process_resolved(
                        outcome="stopped",
                        stage=stage,
                        reply_emitted=response_visible,
                        stop_reason="cost_control",
                    )
                    return await self._finish_normal_exit(engine_context, False)

                await delay(
                    0.3,
                    event_emitter.emit_status_event(
                        trace_id=self._tracer.trace_id,
                        data=StatusEventData(status="processing", message="Thinking"),
                    ),
                )

                # Fire on_preparing before the (latency-heavy) rule/tool loading
                # so preparation-time hooks — e.g. global retrievers — start fetching
                # in parallel and have their results ready by message generation.
                # BAIL stops the turn; RESOLVE skips discovery/matching and responds
                # directly (e.g. a dev-detected greeting); CALL_NEXT is the normal path.
                preparing = await engine_context.dependencies.hooks.call_on_preparing_result(
                    engine_context
                )
                if preparing is CompassHookResult.BAIL:
                    self._compass_tracer.process_resolved(
                        outcome="stopped",
                        stage=stage,
                        reply_emitted=response_visible,
                        stop_reason="hook_bail",
                    )
                    return await self._finish_normal_exit(
                        engine_context,
                        False,
                    )  # Hook requested to bail out

                await self._matcher.preload(engine_context)

                if preparing is not CompassHookResult.RESOLVE:
                    # Initial match before responding: both the composition-mode decision
                    # and the rule-gated retriever hook (fired at the start of the
                    # response loop) need the matches already in place.
                    await self._matcher.fill(engine_context)
                    await self._call_rule_handlers(engine_context)
                    self._stage_input_checkpoint(engine_context)
                    if await self._stop_if_session_became_manual(engine_context):
                        self._compass_tracer.process_resolved(
                            outcome="stopped",
                            stage=stage,
                            reply_emitted=response_visible,
                            stop_reason="manual_mode",
                        )
                        return await self._finish_normal_exit(engine_context, True)
                    await self._stamp_session_labels(engine_context)

                # The responder re-invokes _refresh_state when (re)building the turn
                # instructions after each step, to reevaluate rules gated on the tools
                # that just ran. On RESOLVE we skip matching entirely and pass no
                # refresh_state, so the loop never re-matches (mirrors utter()).
                stage = "responding"
                await self._responder.respond(
                    engine_context,
                    None if preparing is CompassHookResult.RESOLVE else self._refresh_state,
                )
                response_visible = True
                await self._update_session_mode(engine_context)
                # Report as soon as the response is out: latency measures up to
                # the visible response, and the (background) finalization below
                # is deliberately excluded — both from the sample and from the
                # report's timing, so /healthz reflects the turn immediately.
                self._report_turn_health(turn_watch.elapsed, success=True, error=None)
                self._logger.info(f"Turn completed on session {engine_context.session.id}")

                # Post-response finalization. Refresh the interaction with this turn's reply
                # (so compaction's trigger/summary and the cache prefill work off fresh data
                # — the responder persisted the reply but didn't update the in-memory
                # snapshot), compact if needed, then warm next turn's cache. Run as one
                # uncancellable unit: the response already went out, so a mid-finalization
                # cancellation must not leave the session half-compacted or the cache
                # un-warmed.
                stage = "finalizing"

                async def finalize_turn(latch: CancellationSuppressionLatch[None]) -> None:
                    latch.enable()

                    await self._refresh_interaction_history(engine_context)
                    # Compaction is exempt from the background cost gate: it
                    # reduces future cost, so blocking it during a cost event
                    # would be self-defeating.
                    await self._compact_if_needed(engine_context)

                    # BACKGROUND cost-control choke point: pruning and cache
                    # warm-ups are pure spend with no visible turn.
                    if await self._gate_background(engine_context):
                        # After compaction (the pruner reads the fresh summary),
                        # before warm-up: cap the discovered rule and glossary working
                        # sets so the evictions' responder-cache misses land
                        # between turns.
                        await self._matcher.prune_discovered_rules(engine_context)
                        await self._matcher.prune_session_glossary(engine_context)
                        await self._warm_up_best_effort(
                            engine_context
                        )  # Optimize the cache for next turn

                    await self._update_usage(engine_context)
                    await self._commit_workspace(engine_context)
                    self._flush_rule_decisions(engine_context)

                await latched_shield(finalize_turn)
                self._compass_tracer.process_resolved(
                    outcome="completed",
                    stage=stage,
                    reply_emitted=response_visible,
                )
            except asyncio.CancelledError:
                if engine_context is not None:
                    self._logger.warning(
                        f"Processing cancelled on session {engine_context.session.id}"
                    )
                    self._compass_tracer.turn_interrupted(
                        cause="cancelled",
                        session_id=str(engine_context.session.id),
                        agent_id=str(engine_context.agent.id),
                    )
                    await event_emitter.emit_status_event(
                        trace_id=self._tracer.trace_id,
                        data=StatusEventData(status="cancelled", data={}),
                    )
                    await event_emitter.emit_status_event(
                        trace_id=self._tracer.trace_id,
                        data=StatusEventData(status="ready", data={"stage": "completed"}),
                    )
                    await self._finalize_abandoned_workspace(engine_context)
                raise
            except Exception as exc:
                # A successful turn was already reported at response-visible
                # time; a later finalization failure must not double-count it.
                if not response_visible:
                    self._report_turn_health(turn_watch.elapsed, success=False, error=exc)
                self._compass_tracer.process_failed(
                    exc,
                    stage=stage,
                    reply_emitted=response_visible,
                )
                formatted_exception = format_exception(exc)
                self._logger.error(f"Error processing context: {formatted_exception}")
                report_uri = capture_engine_exception(
                    tracer=self._tracer,
                    session_id=str(context.session_id),
                    exception=exc,
                    formatted_traceback=formatted_exception,
                )
                metadata = (
                    metadata_utils.set_internal_value(
                        {}, (DIAGNOSTIC_REPORT_URI_METADATA_KEY,), report_uri
                    )
                    if report_uri
                    else None
                )

                emit_error = engine_context is None
                if engine_context is not None:
                    try:
                        emit_error = await engine_context.dependencies.hooks.call_on_error(
                            engine_context, exc
                        )
                    except Exception as hook_exc:
                        self._logger.error(
                            f"Error hook failed while handling the original error: {hook_exc}"
                        )
                        emit_error = True
                    finally:
                        await self._finalize_abandoned_workspace(engine_context)

                if emit_error and not response_visible:
                    await event_emitter.emit_status_event(
                        trace_id=self._tracer.trace_id,
                        data=StatusEventData(
                            status="error",
                            data={"exception": formatted_exception},
                        ),
                        metadata=metadata,
                    )

                return False
            finally:
                if engine_context is not None:
                    EntityContext.clear(engine_context)

        return True

    def _report_turn_health(
        self,
        duration_seconds: float,
        *,
        success: bool,
        error: BaseException | None,
    ) -> None:
        self._health_reporter.report(
            ENGINE_TURN_KIND,
            {
                EngineHealthView.ATTR_SUCCESS: success,
                EngineHealthView.ATTR_LATENCY_MS: duration_seconds * 1000.0,
                EngineHealthView.ATTR_ERROR_CLASS: (
                    type(error).__name__ if error is not None else None
                ),
            },
        )
        self._health_reporter.increment_counter(ENGINE_TURNS_COUNTER, 1)

    async def _wait_before_processing(self, context: EngineContext) -> None:
        customer_message_index = self._customer_message_run_index(context)
        delay_ms = context.dependencies.optimization_policy.get_process_delay(
            customer_message_index
        )
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1_000)

    def _customer_message_run_index(self, context: EngineContext) -> int:
        customer_message_count = 0

        for event in reversed(context.interaction.events):
            if event.kind != EventKind.MESSAGE:
                continue
            if event.source == EventSource.CUSTOMER:
                customer_message_count += 1
            elif event.is_from_server():
                break

        return max(customer_message_count - 1, 0)

    @override
    async def utter(
        self,
        context: Context,
        event_emitter: EventEmitter,
        requests: Sequence[UtteranceRequest],
    ) -> bool:
        with (
            self._logger.scope("Engine"),
            self._tracer.span(
                "compass.engine.utter",
                {
                    "session_id": context.session_id,
                    "agent_id": context.agent_id,
                },
            ),
        ):
            try:
                engine_context = await self._load_context(context, event_emitter)
                self._tracer.set_attribute(
                    "agent_last_modified", engine_context.agent.modified_utc.isoformat()
                )
                return await self._do_utter(engine_context, requests)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._compass_tracer.process_failed(
                    exc,
                    stage="utter",
                    reply_emitted=False,
                )
                formatted_exception = format_exception(exc)
                self._logger.error(
                    f"Error during uttering in session {context.session_id}: {formatted_exception}"
                )
                report_uri = capture_engine_exception(
                    tracer=self._tracer,
                    session_id=str(context.session_id),
                    exception=exc,
                    formatted_traceback=formatted_exception,
                )
                metadata = (
                    metadata_utils.set_internal_value(
                        {}, (DIAGNOSTIC_REPORT_URI_METADATA_KEY,), report_uri
                    )
                    if report_uri
                    else None
                )
                await event_emitter.emit_status_event(
                    trace_id=self._tracer.trace_id,
                    data=StatusEventData(
                        status="error",
                        data={"exception": formatted_exception},
                    ),
                    metadata=metadata,
                )
                return False

    async def _do_utter(
        self,
        engine_context: EngineContext,
        requests: Sequence[UtteranceRequest],
    ) -> bool:
        try:
            if not await self._gate_turn_with_cost_control(engine_context):
                return await self._finish_normal_exit(engine_context, False)

            # Load the stable response context, but deliberately skip fill():
            # utter() is a direct response and must not freshly match rules or
            # offer tools.
            await self._matcher.preload(engine_context)
            engine_context.state.turn_directives.extend(
                self._utterance_requests_to_turn_directives(requests)
            )

            result = await self._responder.respond(engine_context)

            if not any(step.message.text.strip() for step in result.steps):
                raise RuntimeError("Compass utter produced no message")

            async def finalize_utter(
                latch: CancellationSuppressionLatch[bool],
            ) -> bool:
                latch.enable()
                return await self._finish_normal_exit(engine_context, True)

            return await latched_shield(finalize_utter)
        except BaseException:
            await self._finalize_abandoned_workspace(engine_context)
            raise
        finally:
            EntityContext.clear(engine_context)

    def _utterance_requests_to_turn_directives(
        self,
        requests: Sequence[UtteranceRequest],
    ) -> list[TurnDirective]:
        rationales = {
            UtteranceRationale.UNSPECIFIED: (
                "An external module has determined that this response is necessary."
            ),
            UtteranceRationale.BUY_TIME: (
                "Buy time while work continues on a task in the background."
            ),
            UtteranceRationale.FOLLOW_UP: "Follow up with the customer.",
        }

        return [
            TurnDirective(
                instruction=request.action,
                rationale=rationales[request.rationale],
            )
            for request in requests
        ]

    async def _load_context(
        self,
        context: Context,
        event_emitter: EventEmitter,
        load_interaction: bool = True,
    ) -> EngineContext:
        # Load the full entities from storage.

        agent = await self._entity_queries.read_agent(context.agent_id)
        session = await self._entity_queries.read_session(context.session_id)
        session_metadata = SessionMetadataWorkspace(
            session,
            self._entity_queries.session_store,
        )
        session = session_metadata.session
        customer = await self._entity_queries.read_customer(session.customer_id)

        state = ResponseState(
            agent_effort=agent.effort,
            agent_message_review=agent.message_review,
            agent_tool_review=agent.tool_review,
        )

        if load_interaction:
            interaction = await self._load_interaction_state(
                context,
                state,
                session_metadata=session_metadata,
            )
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
            state=state,
            dependencies=self._dependencies,
            session_metadata=session_metadata,
        )

        # Set in context for access by hooks and other components
        EntityContext.set(result)

        return result

    async def _load_interaction_state(
        self,
        context: Context,
        state: ResponseState,
        session_metadata: SessionMetadataWorkspace,
    ) -> Interaction:
        history = await self._entity_queries.find_events(context.session_id)
        self._prepare_workspace_recovery(session_metadata, history)

        state.session_summary = ""
        events: list[Event] = []

        for event in history:
            if self._is_compaction_event(event):
                events.clear()
                state.session_summary = self._compaction_event_summary(event)
                self._logger.debug(
                    f"Loaded compaction marker for session {context.session_id}; "
                    "discarding earlier interaction events."
                )
                continue

            events.append(event)

        return Interaction(
            events=events,
        )

    def _prepare_workspace_recovery(
        self,
        workspace: SessionMetadataWorkspace,
        raw_history: Sequence[Event],
    ) -> None:
        """Invalidate adaptive Compass state when its input cursor is unsafe."""

        checkpoint = cast(
            Mapping[str, JSONSerializable],
            workspace.get_internal(("compass", "input_checkpoint"), {}),
        )
        needs_rebuild = bool(workspace.get_internal(("compass", "needs_rebuild"), False))

        if checkpoint:
            checkpoint_event_id = str(checkpoint.get("event_id", ""))
            if checkpoint_event_id and not any(
                str(event.id) == checkpoint_event_id for event in raw_history
            ):
                needs_rebuild = True

        if not needs_rebuild:
            return

        # Rebuild is deliberately bounded: discard adaptive caches and let
        # matching reconstruct them from the surviving compacted interaction
        # and the next customer input. Keep the durable rebuild marker set until
        # a successful full match stages a replacement input checkpoint.
        workspace.set_internal(("compass", "needs_rebuild"), True)
        for path in (
            ("compass", "discovered_rule_ids"),
            ("compass", "evicted_discovered_rules"),
            ("compass", "last_discovered_rule_pruning_offset"),
            ("compass", "session_terms"),
            ("compass", "input_checkpoint"),
        ):
            workspace.unset_internal(path)

        self._logger.info(
            f"Rebuilding adaptive Compass metadata for session {workspace.session.id}"
        )

    async def _refresh_interaction_history(self, context: EngineContext) -> None:
        # Reload the interaction from the store so it reflects events persisted during
        # this process() call (notably the agent's reply). Called after responding so
        # downstream steps — compaction and the cache prefill — work off fresh data.
        context.interaction = await self._load_interaction_state(
            context.info,
            context.state,
            context.session_metadata,
        )

    def _stage_input_checkpoint(self, context: EngineContext) -> None:
        """Stage the newest customer message incorporated by full matching."""

        workspace = context.session_metadata
        event = context.interaction.last_customer_message_event
        if event is None:
            return

        content = cast(MessageEventData, event.data)["message"]
        workspace.set_internal(
            ("compass", "input_checkpoint"),
            {
                "event_id": str(event.id),
                "offset": event.offset,
                "modified_utc": event.modified_utc.isoformat(),
                "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            },
        )
        workspace.set_internal(("compass", "needs_rebuild"), False)
        context.session = workspace.session

    async def _refresh_session_preserving_workspace(self, context: EngineContext) -> None:
        """Refresh first-class session fields without detaching staged metadata."""

        refreshed = await self._entity_queries.read_session(context.session.id)
        workspace = context.session_metadata
        workspace.rebase(refreshed)
        context.session = workspace.session

    async def _commit_workspace(self, context: EngineContext) -> None:
        """Commit an active Compass workspace once and install its final view."""

        workspace = context.session_metadata

        if workspace.state is WorkspaceState.ACTIVE:
            context.session = await workspace.commit()

        self._acknowledge_usage()

    def _rollback_workspace(self, context: EngineContext) -> None:
        """Close an active Compass workspace without persisting staged state."""

        workspace = context.session_metadata
        if workspace.state is WorkspaceState.ACTIVE:
            context.session = workspace.rollback()

    async def _finalize_abandoned_workspace(self, context: EngineContext) -> None:
        """Discard behavioral state while retaining already-incurred accounting."""

        await self._cancel_background_tasks(context)
        workspace = context.session_metadata
        if workspace.state is not WorkspaceState.ACTIVE:
            return

        workspace.rollback_behavioral_state()
        try:
            await self._update_usage(context)
            await self._commit_workspace(context)
        except Exception as finalization_exc:
            self._logger.error(
                f"Failed to finalize accounting after an abandoned Compass turn: {finalization_exc}"
            )
            if workspace.state is WorkspaceState.ACTIVE:
                self._rollback_workspace(context)

    async def _finish_normal_exit(self, context: EngineContext, result: bool) -> bool:
        """Commit staged metadata for a non-exceptional early return."""

        await self._cancel_background_tasks(context)
        await self._update_usage(context)
        await self._commit_workspace(context)
        self._flush_rule_decisions(context)
        return result

    def _flush_rule_decisions(self, context: EngineContext) -> None:
        """Emit the turn's staged authoritative rule decisions.

        Called only after the behavioral workspace committed: an abandoned
        turn rolls its match state back and its staged decisions are simply
        discarded with the context, so telemetry never claims a selection
        the store does not hold.
        """
        for decision in context.state.matcher.pending_rule_decisions:
            self._compass_tracer.rule_decision(
                decision.rule,
                selected=decision.selected,
                rationale=decision.rationale,
            )

        context.state.matcher.pending_rule_decisions.clear()

    async def _warm_up_best_effort(self, context: EngineContext) -> None:
        """Warm the next-turn cache without jeopardizing durable turn metadata."""

        try:
            await self._matcher.warm_up(context)
        except Exception as exc:
            self._logger.warning(f"Post-response cache warm-up failed (continuing): {exc}")

    async def _cancel_background_tasks(self, context: EngineContext) -> None:
        """Cancel and await SDK-managed tasks that can no longer be consumed."""

        tasks = list(context.state.background_tasks)
        context.state.background_tasks.clear()
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _is_compaction_event(self, event: Event) -> bool:
        return (
            event.kind == EventKind.MESSAGE
            and event.source == EventSource.SYSTEM
            and metadata_utils.internal_value(event.metadata, ("source",)) == "compacter"
        )

    def _compaction_event_summary(self, event: Event) -> str:
        return cast(MessageEventData, event.data)["message"]

    async def _compact_if_needed(self, context: EngineContext) -> None:
        # Guard against errors so a compaction failure can't fail the turn whose
        # response already went out.
        try:
            if not await self._compacter.needs_compaction(context):
                return

            await self._refresh_interaction_history(context)

            # Measure over the refreshed history — the exact content being compacted.
            tokens_before = await self._compacter.estimate_session_token_count(context)
            await context.dependencies.hooks.call_on_compacting(context, tokens_before)

            await context.session_event_emitter.emit_status_event(
                trace_id=self._tracer.trace_id,
                data=StatusEventData(status="processing", message="Compacting session"),
            )

            result = await self._compacter.compact(context)
            context.state.session_summary = result.summary
            self._compass_tracer.compaction_compacted(
                model=result.generation_info.model,
                summary=result.summary,
            )

            await context.dependencies.hooks.call_on_compacted(
                context,
                Compacted(before=tokens_before, after=result.generation_info.usage.output_tokens),
            )

            self._logger.debug(
                f"Compacted session {context.session.id}: {result.generation_info}\n\n"
                f"Summary:\n{result.summary}"
            )

            await context.session_event_emitter.emit_message_event(
                trace_id=self._tracer.trace_id,
                data=MessageEventData(
                    message=result.summary,
                    participant={
                        "id": None,
                        "display_name": "System",
                    },
                ),
                metadata=metadata_utils.set_internal_value({}, ("source",), "compacter"),
                source=EventSource.SYSTEM,
            )
        except Exception as exc:
            self._compass_tracer.compaction_failed(exc)
            self._logger.error(
                "Session compaction failed after response generation: "
                f"{exc}\n\n{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}"
            )

    async def _refresh_state(self, engine_context: EngineContext) -> None:
        # Called by the responder when (re)building the turn instructions —
        # after a step ran, or before the first step when retrievers already
        # staged tool events in the pre-generation hook.
        if engine_context.state.iterations or engine_context.state.tool_events:
            await self._matcher.update(engine_context)
            await self._call_rule_handlers(engine_context)
            await self._stamp_session_labels(engine_context)

    async def _update_session_mode(self, context: EngineContext) -> None:
        """Apply a tool-requested session-mode switch (ToolResult
        control={"mode": ...}) — e.g. handing the session off to a human
        operator. The mode is applied after the response goes out, so the
        hand-off turn still answers; the next turn's manual-mode check then
        keeps the engine silent."""
        control_outputs = [
            tool_call["result"]["control"]
            for tool_event in context.state.tool_events
            if tool_event.kind == EventKind.TOOL
            for tool_call in cast(ToolEventData, tool_event.data)["tool_calls"]
        ]

        new_mode = context.session.mode
        for control_output in control_outputs:
            new_mode = control_output.get("mode") or new_mode

        if new_mode != context.session.mode:
            self._logger.info(
                f"Session {context.session.id} mode changed to '{new_mode}' by tool control"
            )
            await self._entity_commands.update_session(
                session_id=context.session.id,
                params={"mode": new_mode},
            )

    async def _stamp_session_labels(self, context: EngineContext) -> None:
        """Stamp the session with the labels carried by the rules confirmed as
        matched THIS turn.

        Only rank- or code-confirmed matches are stamped, never mere recall
        candidates: recalled rules land in the sticky discovered set (tagged
        ADMIT_TO_DISCOVERED) and never enter ``*_rule_matches``, while any rule
        carrying labels is forced through RANK by the matcher — so the labels we
        read here always reflect a real per-turn verdict. Upsert is a set union,
        so stamping across the initial match and each post-step reevaluation
        accumulates idempotently.
        """
        labels: set[str] = set()

        for match in chain(
            context.state.ordinary_rule_matches,
            context.state.tool_enabled_rule_matches.keys(),
        ):
            labels.update(match.rule.labels)

        if labels:
            await self._entity_commands.upsert_session_labels(context.session.id, labels)
            self._compass_tracer.labels_added(
                labels,
                [
                    str(match.rule.id)
                    for match in chain(
                        context.state.ordinary_rule_matches,
                        context.state.tool_enabled_rule_matches.keys(),
                    )
                    if match.rule.labels
                ],
            )

    async def _gate_turn_with_cost_control(self, engine_context: EngineContext) -> bool:
        """The TURN cost-control choke point.

        Consults the cost-control policy before any chargeable preparation
        begins. On denial, the turn ends with the cooldown status protocol: a
        terminal `ready` status event carrying a namespaced payload (so clients
        waiting for completion terminate cleanly, and the frontend decides the
        presentation) and NO message event. Fails open: a policy error is
        logged and the turn proceeds."""
        try:
            verdict = await self._cost_control_policy.check(
                CostContext(
                    agent_id=engine_context.agent.id,
                    session_id=engine_context.session.id,
                    customer_id=engine_context.customer.id,
                    trace_id=self._tracer.trace_id,
                    metadata_workspace=engine_context.session_metadata,
                ),
                WorkKind.TURN,
            )
        except PricingConfigurationError:
            raise
        except Exception as exc:
            self._logger.warning(f"Cost-control check failed (failing open): {exc}")
            return True

        for warning in verdict.warnings:
            self._logger.warning(f"Cost-control warning: {warning}")

        if verdict.allowed:
            return True

        payload: dict[str, JSONSerializable] = {
            "circuit_breaker": "open",
            "scope": verdict.scope or "session",
        }

        if verdict.retry_after_utc is not None:
            payload["retry_after_utc"] = verdict.retry_after_utc.isoformat()

        if verdict.reason:
            payload["reason"] = verdict.reason

        await engine_context.session_event_emitter.emit_status_event(
            trace_id=self._tracer.trace_id,
            data=StatusEventData(status="ready", data={"cost_control": payload}),
        )

        self._logger.warning(
            f"Turn denied by cost-control policy for session {engine_context.session.id} "
            f"(reason: {verdict.reason or 'unspecified'})"
        )

        return False

    async def _gate_background(self, engine_context: EngineContext) -> bool:
        """The BACKGROUND cost-control choke point: gates post-response/startup
        work that costs money but has no visible turn (cache warm-ups, session
        pruning). No client protocol on denial — the work is simply skipped.
        Fails open."""
        try:
            verdict = await self._cost_control_policy.check(
                CostContext(
                    agent_id=engine_context.agent.id,
                    session_id=engine_context.session.id,
                    customer_id=engine_context.customer.id,
                    trace_id=self._tracer.trace_id,
                    metadata_workspace=engine_context.session_metadata,
                ),
                WorkKind.BACKGROUND,
            )
        except PricingConfigurationError:
            raise
        except Exception as exc:
            self._logger.warning(f"Cost-control check failed (failing open): {exc}")
            return True

        for warning in verdict.warnings:
            self._logger.warning(f"Cost-control warning: {warning}")

        if not verdict.allowed:
            self._logger.warning(
                f"Background work skipped by cost-control policy for session "
                f"{engine_context.session.id} (reason: {verdict.reason or 'unspecified'})"
            )

        return verdict.allowed

    async def _update_usage(self, context: EngineContext) -> None:
        trace_id = self._tracer.trace_id
        await self._usage_reporter.wait_for_pending(trace_id)
        usage = self._usage_reporter.snapshot_trace(trace_id)
        if not usage:
            return

        try:
            await self._session_budget_service.record_trace(
                context.session.id,
                trace_id,
                usage,
                workspace=context.session_metadata,
            )
        except PricingConfigurationError as exc:
            # The record and its configuration error are already staged.
            self._logger.error(f"Usage pricing failed: {exc}")

    def _acknowledge_usage(self) -> None:
        """Clear in-memory accounting after its workspace commit succeeds."""

        trace_id = self._tracer.trace_id
        self._session_budget_service.acknowledge_trace(trace_id)
        self._usage_reporter.acknowledge_trace(trace_id)

    async def _call_rule_handlers(self, context: EngineContext) -> None:
        discoveries = list(context.state.discovered_rule_evaluations)
        context.state.discovered_rule_evaluations.clear()
        for evaluation in discoveries:
            for handler in context.dependencies.hooks.on_rule_discovered_handlers.get(
                evaluation.rule.id, []
            ):
                await handler(context, evaluation)

        matches = [
            *context.state.ordinary_rule_matches,
            *context.state.tool_enabled_rule_matches.keys(),
        ]
        for match in matches:
            if match.rule.id in context.state.matched_handler_rule_ids:
                continue
            for handler in context.dependencies.hooks.on_rule_matched_handlers.get(
                match.rule.id, []
            ):
                await handler(context, match)
            context.state.matched_handler_rule_ids.add(match.rule.id)

        await self._refresh_session_preserving_workspace(context)

    async def _stop_if_session_became_manual(self, context: EngineContext) -> bool:
        if context.session.mode != "manual":
            return False

        await context.session_event_emitter.emit_status_event(
            trace_id=self._tracer.trace_id,
            data=StatusEventData(status="ready", data={"session_mode": "manual"}),
        )
        return True
