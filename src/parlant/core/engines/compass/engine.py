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
from dataclasses import asdict, replace
import traceback
from typing import cast
from typing_extensions import override

from parlant.core.common import JSONSerializable
from parlant.core.cost_control import CostContext, CostControlPolicy, WorkKind
from parlant.core.async_utils import (
    CancellationSuppressionLatch,
    delay,
    latched_shield,
    safe_gather,
)
from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.emissions import EventEmitter
from parlant.core.engines.entity_context import EntityContext
from parlant.core.engines.alpha.hooks import EngineHooks
from parlant.core.engines.engine_context import Interaction
from parlant.core.engines.compass.compacter import Compacter
from parlant.core.engines.compass.matcher import Matcher
from parlant.core.engines.compass.responder import Responder
from parlant.core.engines.compass.response_state import EngineContext, ResponseState
from parlant.core.engines.compass.tracing import CompassTracer
from parlant.core.engines.types import Context, Engine, UtteranceRequest
from parlant.core.entity_cq import EntityCommands, EntityQueries
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.sessions import Event, EventKind, EventSource, MessageEventData, StatusEventData
from parlant.core.tracer import Tracer
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
        hooks: EngineHooks,
        usage_reporter: UsageReporter,
        cost_control_policy: CostControlPolicy,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._meter = meter

        self._matcher = matcher
        self._responder = responder
        self._compacter = compacter

        self._entity_queries = entity_queries
        self._entity_commands = entity_commands
        self._hooks = hooks
        self._usage_reporter = usage_reporter
        self._cost_control_policy = cost_control_policy

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

        with self._tracer.span(
            "engine.initialize",
            {
                "session_id": context.session_id,
                "agent_id": context.agent_id,
                "agent_last_modified": engine_context.agent.modified_utc.isoformat(),
            },
        ):
            # Warm the provider cache for the system prompt as soon as the session
            # exists, before any message arrives. The interaction is empty here, so
            # only the stable system prefix is warmed; the real turn later reads it.
            await self._matcher.preload(engine_context)

            if await self._gate_background(engine_context):
                await safe_gather(
                    self._matcher.warm_up(engine_context),
                    self._responder.warm_up(engine_context),
                )

    @override
    async def process(
        self,
        context: Context,
        event_emitter: EventEmitter,
    ) -> bool:
        with self._tracer.span(
            "engine.process",
            {
                "session_id": context.session_id,
                "agent_id": context.agent_id,
            },
        ):
            engine_context: EngineContext | None = None

            try:
                # Load the context inside the process span so failures during
                # loading are observable as process failures too.
                engine_context = await self._load_context(context, event_emitter)
                self._tracer.set_attribute(
                    "agent_last_modified",
                    engine_context.agent.modified_utc.isoformat(),
                )

                with self._tracer.attributes(
                    {
                        "agent_id": engine_context.agent.id,
                        "agent_last_modified": engine_context.agent.modified_utc.isoformat(),
                    }
                ):
                    if not await self._hooks.call_on_acknowledging(engine_context):
                        return False  # Hook requested to bail out

                    await event_emitter.emit_status_event(
                        trace_id=self._tracer.trace_id,
                        data=StatusEventData(status="acknowledged"),
                    )

                    if not await self._hooks.call_on_acknowledged(engine_context):
                        return False  # Hook requested to bail out

                    # TURN cost-control choke point: a denied turn is acknowledged and
                    # terminated with a `ready` status event only (see _gate_turn).
                    if not await self._gate_turn_with_cost_control(engine_context):
                        return False

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
                    if not await self._hooks.call_on_preparing(engine_context):
                        return False  # Hook requested to bail out

                    await self._matcher.preload(engine_context)

                    # Initial match before responding: both the composition-mode decision
                    # and the rule-gated retriever hook (fired at the start of the
                    # response loop) need the matches already in place.
                    await self._matcher.fill(engine_context)

                    # The responder re-invokes _refresh_state when (re)building the turn
                    # instructions after each step, to reevaluate rules gated on the
                    # tools that just ran.
                    await self._responder.respond(engine_context, self._refresh_state)

                    # Post-response finalization. Refresh the interaction with this turn's
                    # reply (so compaction's trigger/summary and the cache prefill work off
                    # fresh data — the responder persisted the reply but didn't update the
                    # in-memory snapshot), compact if needed, then warm next turn's cache.
                    # Run as one uncancellable unit: the response already went out, so a
                    # mid-finalization cancellation must not leave the session half-compacted
                    # or the cache un-warmed.
                    async def finalize_turn(latch: CancellationSuppressionLatch[None]) -> None:
                        with self._tracer.span("engine.process.finalize"):
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
                                # before warm-up: cap the session rule and glossary working
                                # sets so the evictions' responder-cache misses land
                                # between turns.
                                await self._matcher.prune_session_rules(engine_context)
                                await self._matcher.prune_session_glossary(engine_context)
                                with self._tracer.span("engine.process.warmup"):
                                    # Optimize the cache for next turn.
                                    await self._matcher.warm_up(engine_context)
                            await self._update_usage(engine_context)

                    await latched_shield(finalize_turn)
            except asyncio.CancelledError:
                session_id = engine_context.session.id if engine_context else context.session_id
                self._logger.warning(f"Processing cancelled on session {session_id}")
                await event_emitter.emit_status_event(
                    trace_id=self._tracer.trace_id,
                    data=StatusEventData(status="cancelled", data={}),
                )
                await event_emitter.emit_status_event(
                    trace_id=self._tracer.trace_id,
                    data=StatusEventData(status="ready", data={"stage": "completed"}),
                )
                raise
            except Exception as e:
                CompassTracer(self._tracer).process_failed(e)
                self._logger.error(
                    f"Error processing context: {e}\n\n{''.join(traceback.format_exception(type(e), e, e.__traceback__))}"
                )

                if engine_context is None:
                    raise

                if await self._hooks.call_on_error(engine_context, e):
                    await event_emitter.emit_status_event(
                        trace_id=self._tracer.trace_id,
                        data=StatusEventData(status="error"),
                    )

                return False

        return True

    @override
    async def utter(
        self,
        context: Context,
        event_emitter: EventEmitter,
        requests: Sequence[UtteranceRequest],
    ) -> bool:
        with self._tracer.span(
            "utter",
            {
                "session_id": context.session_id,
                "agent_id": context.agent_id,
            },
        ):
            return False

    async def _load_context(
        self,
        context: Context,
        event_emitter: EventEmitter,
        load_interaction: bool = True,
    ) -> EngineContext:
        with self._tracer.span(
            "load.context",
            {
                "session_id": context.session_id,
                "agent_id": context.agent_id,
            },
        ):
            # Load the full entities from storage.

            agent = await self._entity_queries.read_agent(context.agent_id)
            session = await self._entity_queries.read_session(context.session_id)
            customer = await self._entity_queries.read_customer(session.customer_id)

            state = ResponseState(agent_effort=agent.effort)

            if load_interaction:
                interaction = await self._load_interaction_state(context, state)
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
            )

            # Set in context for access by hooks and other components
            EntityContext.set(result)

            return result

    async def _load_interaction_state(
        self,
        context: Context,
        state: ResponseState,
    ) -> Interaction:
        history = await self._entity_queries.find_events(context.session_id)

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

    async def _refresh_interaction_history(self, context: EngineContext) -> None:
        # Reload the interaction from the store so it reflects events persisted during
        # this process() call (notably the agent's reply). Called after responding so
        # downstream steps — compaction and the cache prefill — work off fresh data.
        context.interaction = await self._load_interaction_state(context.info, context.state)

    def _is_compaction_event(self, event: Event) -> bool:
        return (
            event.kind == EventKind.MESSAGE
            and event.source == EventSource.SYSTEM
            and event.metadata.get("source") == "compacter"
        )

    def _compaction_event_summary(self, event: Event) -> str:
        return cast(MessageEventData, event.data)["message"]

    async def _compact_if_needed(self, context: EngineContext) -> None:
        # Guard against errors so a compaction failure can't fail the turn whose
        # response already went out.
        failure_event_emitted = False

        try:
            if not await self._compacter.needs_compaction(context):
                return

            await self._refresh_interaction_history(context)

            await context.session_event_emitter.emit_status_event(
                trace_id=self._tracer.trace_id,
                data=StatusEventData(status="processing", message="Compacting session"),
            )

            try:
                result = await self._compacter.compact(context)
            except Exception:
                failure_event_emitted = True
                raise

            context.state.session_summary = result.summary

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
                metadata={"source": "compacter"},
                source=EventSource.SYSTEM,
            )
            CompassTracer(self._tracer).compaction_compacted(
                result.generation_info.model,
                result.summary,
            )
        except Exception as exc:
            if not failure_event_emitted:
                CompassTracer(self._tracer).compaction_failed(exc)
            self._logger.error(
                "Session compaction failed after response generation: "
                f"{exc}\n\n{''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}"
            )

    async def _refresh_state(self, engine_context: EngineContext) -> None:
        # Called by the responder when (re)building the turn instructions
        if engine_context.state.iterations:
            await self._matcher.update(engine_context)

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
                ),
                WorkKind.TURN,
            )
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
                ),
                WorkKind.BACKGROUND,
            )
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
        usage = self._usage_reporter.get_usage()
        if not usage:
            return

        trace_id = self._tracer.trace_id
        metadata = dict(context.session.metadata)
        usage_metadata = dict(cast(Mapping[str, JSONSerializable], metadata.get("usage", {})))
        usage_metadata[trace_id] = {
            model_name: cast(JSONSerializable, asdict(usage_info))
            for model_name, usage_info in usage.items()
        }
        metadata["usage"] = usage_metadata

        context.session = replace(
            context.session,
            metadata=metadata,
        )
        await self._entity_commands.update_session(
            context.session.id,
            {"metadata": metadata},
        )
