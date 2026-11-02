import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock
import zipfile

import pytest

from parlant.core.diagnostic_reports import (
    DIAGNOSTIC_REPORT_URI_METADATA_KEY,
    DiagnosticReportRepository,
    DiagnosticReportService,
    DiagnosticStore,
    RecordingTracer,
    TraceLogBuffer,
)
from parlant.core.emission.event_buffer import EventBuffer
from parlant.core.emissions import EventEmitter
from parlant.core.engines.compass.compacter import CompactionResult
from parlant.core.engines.compass.engine import CompassEngine
from parlant.core.engines.compass.tracing import CompassTracer, RuleDecision
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.engines.compass.context import (
    CompassDependencies,
    EngineContext,
    ResponseState,
)
from parlant.core.engines.compass.optimization_policy import CompassOptimizationPolicy
from parlant.core.engines.session_metadata_workspace import SessionMetadataWorkspace
from parlant.core.engines.compass.hooks import CompassHooks, CompassHookResult
from parlant.core.engines.rule_evaluation import RuleEvaluation
from parlant.core.engines.types import (
    Context,
    UtteranceRationale,
    UtteranceRequest,
)
from parlant.core.loggers import StdoutLogger
from parlant.core.meter import Meter
from parlant.core import metadata as metadata_utils
from parlant.core.sessions import Event, EventKind, EventSource, SessionId
from parlant.core.tracer import LocalTracer
from parlant.core.cost_control import NullCostControlPolicy
from parlant.core.usage_reporter import UsageReporter

from tests.core.stable.engines.compass.matching.utils import (
    create_agent,
    create_engine_context,
    create_customer,
    create_rule,
    create_session,
)
from tests.core.common.utils import create_event_message
from parlant.core.application_context import ApplicationContext
from parlant.core.health import NullHealthReporter


def _health_reporter() -> NullHealthReporter:
    return NullHealthReporter(ApplicationContext(instance_id="test-instance"))


class _FakeEntityQueries:
    def __init__(self) -> None:
        self.agent = create_agent()
        self.customer = create_customer()
        self.session = create_session(self.agent, self.customer)
        self.session_store = self

    async def read_agent(self, agent_id: str) -> Any:
        return self.agent

    async def read_session(self, session_id: SessionId) -> Any:
        return self.session

    async def read_customer(self, customer_id: str) -> Any:
        return self.customer

    async def find_events(self, session_id: SessionId) -> list[Event]:
        return []

    async def commit_session_metadata(
        self,
        session_id: SessionId,
        expected_revision: int,
        metadata: dict[str, Any],
    ) -> Any:
        assert self.session.metadata_revision == expected_revision
        self.session = replace(
            self.session,
            metadata=metadata,
            metadata_revision=expected_revision + 1,
        )
        return self.session


class _CancellableMatcher:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.background_cancelled = asyncio.Event()

    async def preload(self, context: Any) -> None:
        async def background() -> None:
            try:
                await asyncio.Future[None]()
            finally:
                self.background_cancelled.set()

        context.state.background_tasks.add(asyncio.create_task(background()))
        self.started.set()
        await asyncio.Future[None]()

    async def match_early(self, context: Any) -> None:
        pass

    async def fill(self, context: Any) -> None:
        raise AssertionError("fill should not be reached")


class _FailingEntityQueries(_FakeEntityQueries):
    async def read_agent(self, agent_id: str) -> Any:
        raise RuntimeError("compass-context-load-sentinel")


class _FakeHooks:
    async def call_on_acknowledging(self, context: Any) -> bool:
        return True

    async def call_on_acknowledged(self, context: Any) -> bool:
        return True

    async def call_on_preparing(self, context: Any) -> bool:
        return True

    async def call_on_error(self, context: Any, exc: Exception) -> bool:
        return True


@pytest.mark.parametrize(
    ("conversation", "expected"),
    [
        ([(EventSource.CUSTOMER, "first")], 0),
        (
            [
                (EventSource.CUSTOMER, "first"),
                (EventSource.CUSTOMER, "second"),
            ],
            1,
        ),
        (
            [
                (EventSource.CUSTOMER, "first"),
                (EventSource.CUSTOMER, "second"),
                (EventSource.CUSTOMER, "third"),
            ],
            2,
        ),
        (
            [
                (EventSource.CUSTOMER, "first"),
                (EventSource.AI_AGENT, "reply"),
                (EventSource.CUSTOMER, "new first"),
            ],
            0,
        ),
        (
            [
                (EventSource.CUSTOMER, "first"),
                (EventSource.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT, "reply"),
                (EventSource.CUSTOMER, "new first"),
                (EventSource.CUSTOMER, "new second"),
            ],
            1,
        ),
    ],
)
def test_that_customer_message_run_index_resets_after_an_agent_reply(
    conversation: list[tuple[EventSource, str]],
    expected: int,
) -> None:
    context = create_engine_context(conversation=conversation)
    engine = object.__new__(CompassEngine)

    assert engine._customer_message_run_index(context) == expected


@pytest.mark.parametrize(("delay_ms", "expected_seconds"), [(0, None), (1_250, 1.25)])
async def test_that_process_delay_sleeps_only_for_positive_delays(
    delay_ms: int,
    expected_seconds: float | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ProcessDelayPolicy(CompassOptimizationPolicy):
        def __init__(self) -> None:
            self.calls: list[int] = []

        def get_process_delay(self, k: int) -> int:
            self.calls.append(k)
            return delay_ms

    context = create_engine_context(
        conversation=[
            (EventSource.CUSTOMER, "first"),
            (EventSource.CUSTOMER, "second"),
        ]
    )
    policy = _ProcessDelayPolicy()
    context.dependencies = CompassDependencies(
        optimization_policy=policy,
        hooks=context.dependencies.hooks,
    )
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)
    engine = object.__new__(CompassEngine)

    await engine._wait_before_processing(context)

    assert policy.calls == [1]
    if expected_seconds is None:
        sleep.assert_not_awaited()
    else:
        sleep.assert_awaited_once_with(expected_seconds)


async def test_that_process_delay_is_applied_before_acknowledgement() -> None:
    calls: list[str] = []

    class _RecordingPolicy(CompassOptimizationPolicy):
        def get_process_delay(self, k: int) -> int:
            calls.append("delay")
            return 0

    async def bail_during_acknowledgement(
        hook_context: EngineContext,
        payload: Any,
        exc: Exception | None,
    ) -> CompassHookResult:
        calls.append("acknowledgement")
        return CompassHookResult.BAIL

    hooks = CompassHooks()
    hooks.on_acknowledging.append(bail_during_acknowledgement)

    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hello")])
    context.dependencies = CompassDependencies(
        optimization_policy=_RecordingPolicy(),
        hooks=hooks,
    )

    preloaded_context = context

    class _PreloadedContextEngine(CompassEngine):
        finish_exit_results: list[bool]

        async def _load_context(
            self,
            context: Context,
            event_emitter: EventEmitter,
            load_interaction: bool = True,
        ) -> EngineContext:
            return preloaded_context

        async def _finish_normal_exit(self, engine_context: EngineContext, result: bool) -> bool:
            self.finish_exit_results.append(result)
            return result

    engine = object.__new__(_PreloadedContextEngine)
    engine.finish_exit_results = []
    engine._tracer = context.tracer
    engine._compass_tracer = CompassTracer(context.tracer)
    # process() wraps its body in the component logging scope.
    engine._logger = context.logger

    result = await engine.process(context.info, context.session_event_emitter)

    assert result is False
    assert calls == ["delay", "acknowledgement"]
    assert engine.finish_exit_results == [False]


async def test_that_compass_calls_discovered_then_matched_handlers_once() -> None:
    context = create_engine_context(conversation=[])
    context.state = ResponseState(
        agent_effort=context.agent.effort,
    )
    evaluation = RuleEvaluation(rule=create_rule("Customer says hello"), reasoning="Relevant")
    context.state.discovered_rule_evaluations.append(evaluation)
    context.state.ordinary_rule_matches.append(evaluation)

    hooks = context.dependencies.hooks
    calls: list[tuple[str, RuleEvaluation]] = []

    async def discovered_handler(handler_context: Any, handler_evaluation: RuleEvaluation) -> None:
        assert handler_context is context
        calls.append(("discovered", handler_evaluation))

    async def matched_handler(handler_context: Any, handler_evaluation: RuleEvaluation) -> None:
        assert handler_context is context
        calls.append(("matched", handler_evaluation))

    hooks.on_rule_discovered_handlers[evaluation.rule.id].append(discovered_handler)
    hooks.on_rule_matched_handlers[evaluation.rule.id].append(matched_handler)

    engine = object.__new__(CompassEngine)
    engine._entity_queries = cast(Any, _FakeEntityQueries())

    await engine._call_rule_handlers(context)
    await engine._call_rule_handlers(context)

    assert calls == [("discovered", evaluation), ("matched", evaluation)]
    assert calls[0][1] is calls[1][1] is evaluation
    hooks = CompassHooks()
    assert hasattr(hooks, "on_rule_discovered_handlers")
    assert hasattr(hooks, "on_rule_matched_handlers")


async def test_that_loaded_context_carries_the_engine_dependency_bundle() -> None:
    tracer = LocalTracer()
    entity_queries = _FakeEntityQueries()
    dependencies = CompassDependencies(CompassOptimizationPolicy())
    engine = CompassEngine(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        meter=cast(Meter, object()),
        matcher=cast(Any, object()),
        responder=cast(Any, object()),
        compacter=cast(Any, object()),
        entity_queries=cast(Any, entity_queries),
        entity_commands=cast(Any, object()),
        usage_reporter=UsageReporter(tracer),
        session_budget_service=cast(Any, _NoopBudgetService()),
        cost_control_policy=NullCostControlPolicy(),
        health_reporter=_health_reporter(),
        dependencies=dependencies,
    )

    context = await engine._load_context(
        Context(
            session_id=entity_queries.session.id,
            agent_id=entity_queries.agent.id,
        ),
        EventBuffer(entity_queries.agent),
        load_interaction=False,
    )

    assert context.dependencies is dependencies
    assert context.dependencies.optimization_policy is dependencies.optimization_policy


def test_that_compass_converts_utterance_requests_to_ordered_turn_directives() -> None:
    engine = object.__new__(CompassEngine)

    directives = engine._utterance_requests_to_turn_directives(
        [
            UtteranceRequest(
                action="Say that more information is coming.",
                rationale=UtteranceRationale.BUY_TIME,
            ),
            UtteranceRequest(
                action="Ask whether the customer wants to continue.",
                rationale=UtteranceRationale.FOLLOW_UP,
            ),
            UtteranceRequest(
                action="Mention the updated status.",
                rationale=UtteranceRationale.UNSPECIFIED,
            ),
        ]
    )

    assert [directive.instruction for directive in directives] == [
        "Say that more information is coming.",
        "Ask whether the customer wants to continue.",
        "Mention the updated status.",
    ]
    assert "background" in directives[0].rationale
    assert "Follow up" in directives[1].rationale
    assert "external module" in directives[2].rationale


class _RecordingUtterMatcher:
    def __init__(self) -> None:
        self.preload_calls: list[Any] = []
        self.fill_calls: list[Any] = []
        self.update_calls: list[Any] = []

    async def preload(self, context: Any) -> None:
        self.preload_calls.append(context)

    async def fill(self, context: Any) -> None:
        self.fill_calls.append(context)

    async def update(self, context: Any) -> None:
        self.update_calls.append(context)


class _RecordingUtterResponder:
    def __init__(self) -> None:
        self.respond_calls: list[Any] = []

    async def respond(self, context: Any, refresh_state: Any = None) -> Any:
        self.respond_calls.append(context)
        return SimpleNamespace(
            steps=[
                SimpleNamespace(message=SimpleNamespace(text="I am still checking that for you."))
            ]
        )


class _UtterProbeEngine(CompassEngine):
    async def _gate_turn_with_cost_control(self, engine_context: EngineContext) -> bool:
        return True

    async def _finish_normal_exit(self, context: EngineContext, result: bool) -> bool:
        return result

    async def _finalize_abandoned_workspace(self, context: EngineContext) -> None:
        return None


async def test_that_compass_utter_preloads_without_matching_or_offering_tools() -> None:
    engine_context = create_engine_context(
        conversation=[(EventSource.CUSTOMER, "Please keep me updated.")]
    )
    matcher = _RecordingUtterMatcher()
    responder = _RecordingUtterResponder()

    engine = object.__new__(_UtterProbeEngine)
    engine._tracer = engine_context.tracer
    engine._matcher = cast(Any, matcher)
    engine._responder = cast(Any, responder)

    succeeded = await engine._do_utter(
        engine_context,
        [
            UtteranceRequest(
                action="Tell the customer that the check is still in progress.",
                rationale=UtteranceRationale.BUY_TIME,
            )
        ],
    )

    assert succeeded is True
    assert matcher.preload_calls == [engine_context]
    assert matcher.fill_calls == []
    assert matcher.update_calls == []
    assert responder.respond_calls == [engine_context]
    assert engine_context.state.available_tools == []
    assert engine_context.state.ordinary_rule_matches == []
    assert [directive.instruction for directive in engine_context.state.turn_directives] == [
        "Tell the customer that the check is still in progress."
    ]


async def test_process_cancellation_during_preparation_emits_cancelled_and_ready() -> None:
    tracer = LocalTracer()
    entity_queries = _FakeEntityQueries()
    matcher = _CancellableMatcher()
    event_emitter = EventBuffer(entity_queries.agent)

    engine = CompassEngine(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        meter=cast(Meter, object()),
        matcher=cast(Any, matcher),
        responder=cast(Any, object()),
        compacter=cast(Any, object()),
        entity_queries=cast(Any, entity_queries),
        entity_commands=cast(Any, object()),
        usage_reporter=UsageReporter(tracer),
        session_budget_service=cast(Any, object()),
        cost_control_policy=NullCostControlPolicy(),
        health_reporter=_health_reporter(),
        dependencies=CompassDependencies(CompassOptimizationPolicy()),
    )

    task = asyncio.create_task(
        engine.process(
            Context(session_id=entity_queries.session.id, agent_id=entity_queries.agent.id),
            event_emitter,
        )
    )

    await asyncio.wait_for(matcher.started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert matcher.background_cancelled.is_set()

    status_events = [event for event in event_emitter.events if event.kind == EventKind.STATUS]

    assert [cast(dict[str, Any], event.data)["status"] for event in status_events] == [
        "acknowledged",
        "processing",
        "cancelled",
        "ready",
    ]


async def test_process_context_loading_failure_creates_a_diagnostic_report(
    tmp_path: Path,
) -> None:
    store = DiagnosticStore()
    repository = DiagnosticReportRepository(
        error_report_dir=tmp_path / "reports",
        trace_report_dir=tmp_path / "traces",
        error_history_size=8,
    )
    service = DiagnosticReportService(store, repository)
    tracer = RecordingTracer(LocalTracer(), service)
    entity_queries = _FailingEntityQueries()
    event_emitter = EventBuffer(entity_queries.agent)

    engine = CompassEngine(
        logger=TraceLogBuffer(tracer, store),
        tracer=tracer,
        meter=cast(Meter, object()),
        matcher=cast(Any, object()),
        responder=cast(Any, object()),
        compacter=cast(Any, object()),
        entity_queries=cast(Any, entity_queries),
        entity_commands=cast(Any, object()),
        usage_reporter=UsageReporter(tracer),
        session_budget_service=cast(Any, object()),
        cost_control_policy=NullCostControlPolicy(),
        health_reporter=_health_reporter(),
        dependencies=CompassDependencies(CompassOptimizationPolicy()),
    )

    succeeded = await engine.process(
        Context(session_id=entity_queries.session.id, agent_id=entity_queries.agent.id),
        event_emitter,
    )

    assert succeeded is False
    status_event = next(event for event in event_emitter.events if event.kind == EventKind.STATUS)
    assert cast(dict[str, Any], status_event.data)["status"] == "error"
    assert status_event.metadata is not None
    report_uri = metadata_utils.internal_value(
        status_event.metadata, (DIAGNOSTIC_REPORT_URI_METADATA_KEY,)
    )
    assert isinstance(report_uri, str)
    report_path = repository.resolve_error(status_event.trace_id)
    assert report_path is not None
    with zipfile.ZipFile(report_path) as archive:
        assert "compass-context-load-sentinel" in archive.read("traceback.txt").decode()


async def test_process_context_loading_failure_omits_report_metadata_when_disabled() -> None:
    tracer = LocalTracer()
    entity_queries = _FailingEntityQueries()
    event_emitter = EventBuffer(entity_queries.agent)

    engine = CompassEngine(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        meter=cast(Meter, object()),
        matcher=cast(Any, object()),
        responder=cast(Any, object()),
        compacter=cast(Any, object()),
        entity_queries=cast(Any, entity_queries),
        entity_commands=cast(Any, object()),
        usage_reporter=UsageReporter(tracer),
        session_budget_service=cast(Any, object()),
        cost_control_policy=NullCostControlPolicy(),
        health_reporter=_health_reporter(),
        dependencies=CompassDependencies(CompassOptimizationPolicy()),
    )

    succeeded = await engine.process(
        Context(session_id=entity_queries.session.id, agent_id=entity_queries.agent.id),
        event_emitter,
    )

    assert succeeded is False
    status_event = next(event for event in event_emitter.events if event.kind == EventKind.STATUS)
    assert (
        metadata_utils.internal_value(
            status_event.metadata,
            (DIAGNOSTIC_REPORT_URI_METADATA_KEY,),
        )
        is None
    )


def test_missing_input_checkpoint_event_marks_and_clears_adaptive_state() -> None:
    tracer = LocalTracer()
    entity_queries = _FakeEntityQueries()
    metadata = metadata_utils.set_internal_value(
        entity_queries.session.metadata,
        ("compass", "input_checkpoint"),
        {
            "event_id": "deleted-event",
            "offset": 4,
            "modified_utc": "2026-07-28T00:00:00+00:00",
            "content_sha256": "old",
        },
    )
    metadata = metadata_utils.set_internal_value(
        metadata,
        ("compass", "discovered_rule_ids"),
        ["rule-1"],
    )
    metadata = metadata_utils.set_internal_value(
        metadata,
        ("compass", "session_terms"),
        {"term-1": 4},
    )
    session = replace(entity_queries.session, metadata=metadata)
    workspace = SessionMetadataWorkspace(session, cast(Any, entity_queries))
    engine = object.__new__(CompassEngine)
    engine._logger = StdoutLogger(tracer)

    engine._prepare_workspace_recovery(
        workspace,
        [create_event_message(5, EventSource.CUSTOMER, "surviving")],
    )

    assert workspace.get_internal(("compass", "needs_rebuild")) is True
    assert workspace.get_internal(("compass", "input_checkpoint")) is None
    assert workspace.get_internal(("compass", "discovered_rule_ids")) is None
    assert workspace.get_internal(("compass", "session_terms")) is None


async def test_that_a_turn_logs_info_lifecycle_lines() -> None:
    """A processed turn logs 'Turn started' at INFO once past the manual-mode
    check, and 'Turn completed' at INFO when the response becomes visible."""

    info_lines: list[str] = []

    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hello")])
    preloaded_context = context

    class _LifecycleProbeEngine(CompassEngine):
        async def _load_context(
            self,
            context: Context,
            event_emitter: EventEmitter,
            load_interaction: bool = True,
        ) -> EngineContext:
            return preloaded_context

        async def _wait_before_processing(self, engine_context: EngineContext) -> None:
            pass

        async def _call_rule_handlers(self, engine_context: EngineContext) -> None:
            pass

        async def _stop_if_session_became_manual(self, engine_context: EngineContext) -> bool:
            return False

        async def _gate_turn_with_cost_control(self, engine_context: EngineContext) -> bool:
            return True

        def _stage_input_checkpoint(self, engine_context: EngineContext) -> None:
            pass

        async def _stamp_session_labels(self, engine_context: EngineContext) -> None:
            pass

        def _report_turn_health(
            self, duration_seconds: float, *, success: bool, error: BaseException | None
        ) -> None:
            pass

        async def _refresh_interaction_history(self, engine_context: EngineContext) -> None:
            pass

        async def _compact_if_needed(self, engine_context: EngineContext) -> None:
            pass

        async def _gate_background(self, engine_context: EngineContext) -> bool:
            return False

        async def _update_usage(self, engine_context: EngineContext) -> None:
            pass

        async def _commit_workspace(self, engine_context: EngineContext) -> None:
            pass

    class _RecordingLifecycleLogger(StdoutLogger):
        def info(self, message: str) -> None:
            info_lines.append(message)

    class _NoopMatcher:
        async def match_early(self, engine_context: EngineContext) -> None:
            pass

        async def preload(self, engine_context: EngineContext) -> None:
            pass

        async def fill(self, engine_context: EngineContext) -> None:
            pass

    class _NoopResponder:
        async def respond(self, engine_context: EngineContext, refresh_state: Any) -> None:
            pass

    engine = object.__new__(_LifecycleProbeEngine)
    engine._logger = _RecordingLifecycleLogger(LocalTracer())
    engine._tracer = context.tracer
    engine._compass_tracer = CompassTracer(context.tracer)
    engine._matcher = cast(Any, _NoopMatcher())
    engine._responder = cast(Any, _NoopResponder())

    result = await engine.process(context.info, context.session_event_emitter)

    assert result is True
    started = [line for line in info_lines if line.startswith("Turn started")]
    completed = [line for line in info_lines if line.startswith("Turn completed")]
    assert len(started) == 1 and context.session.id in started[0]
    assert len(completed) == 1 and context.session.id in completed[0]


class _ActivityRecordingTracer(LocalTracer):
    def __init__(self) -> None:
        super().__init__()
        self.recorded_events: list[tuple[str, dict[str, Any]]] = []
        self.recorded_attributes: dict[str, Any] = {}

    def add_event(self, name: str, attributes: Any = {}) -> None:
        self.recorded_events.append((name, dict(attributes)))

    def set_attribute(self, name: str, value: Any) -> None:
        self.recorded_attributes[name] = value
        super().set_attribute(name, value)


class _NoopBudgetService:
    def acknowledge_trace(self, trace_id: str) -> None: ...


def _make_process_engine(
    tracer: LocalTracer,
    entity_queries: Any,
    *,
    matcher: Any = None,
    hooks: CompassHooks | None = None,
) -> CompassEngine:
    dependencies = (
        CompassDependencies(CompassOptimizationPolicy(), hooks=hooks)
        if hooks is not None
        else CompassDependencies(CompassOptimizationPolicy())
    )

    return CompassEngine(
        logger=StdoutLogger(tracer),
        tracer=tracer,
        meter=cast(Meter, object()),
        matcher=cast(Any, matcher if matcher is not None else object()),
        responder=cast(Any, object()),
        compacter=cast(Any, object()),
        entity_queries=cast(Any, entity_queries),
        entity_commands=cast(Any, object()),
        usage_reporter=UsageReporter(tracer),
        session_budget_service=cast(Any, _NoopBudgetService()),
        cost_control_policy=NullCostControlPolicy(),
        health_reporter=_health_reporter(),
        dependencies=dependencies,
    )


async def test_that_a_context_loading_failure_emits_a_terminal_process_failed_event() -> None:
    tracer = _ActivityRecordingTracer()
    entity_queries = _FailingEntityQueries()
    event_emitter = EventBuffer(entity_queries.agent)

    engine = _make_process_engine(tracer, entity_queries)

    succeeded = await engine.process(
        Context(session_id=entity_queries.session.id, agent_id=entity_queries.agent.id),
        event_emitter,
    )

    assert succeeded is False
    failures = [a for n, a in tracer.recorded_events if n == "process.failed"]
    assert len(failures) == 1
    assert failures[0]["error_type"] == "RuntimeError"
    assert failures[0]["stage"] == "loading_context"
    assert failures[0]["reply_emitted"] is False
    assert failures[0]["terminal"] is True


async def test_that_a_hook_bail_resolves_the_process_outcome_as_stopped() -> None:
    tracer = _ActivityRecordingTracer()
    entity_queries = _FakeEntityQueries()
    event_emitter = EventBuffer(entity_queries.agent)

    async def bail(
        hook_context: EngineContext,
        payload: Any,
        exc: Exception | None,
    ) -> CompassHookResult:
        return CompassHookResult.BAIL

    hooks = CompassHooks()
    hooks.on_acknowledging.append(bail)

    engine = _make_process_engine(tracer, entity_queries, hooks=hooks)

    succeeded = await engine.process(
        Context(session_id=entity_queries.session.id, agent_id=entity_queries.agent.id),
        event_emitter,
    )

    assert succeeded is False
    assert tracer.recorded_attributes["outcome"] == "stopped"
    assert tracer.recorded_attributes["stop_reason"] == "hook_bail"
    assert tracer.recorded_attributes["reply_emitted"] is False


async def test_that_a_manual_mode_session_resolves_the_process_outcome_as_stopped() -> None:
    tracer = _ActivityRecordingTracer()
    entity_queries = _FakeEntityQueries()
    entity_queries.session = replace(entity_queries.session, mode="manual")
    event_emitter = EventBuffer(entity_queries.agent)

    engine = _make_process_engine(tracer, entity_queries)

    succeeded = await engine.process(
        Context(session_id=entity_queries.session.id, agent_id=entity_queries.agent.id),
        event_emitter,
    )

    assert succeeded is True
    assert tracer.recorded_attributes["outcome"] == "stopped"
    assert tracer.recorded_attributes["stop_reason"] == "manual_mode"


async def test_that_a_cancelled_turn_emits_a_turn_interrupted_event() -> None:
    tracer = _ActivityRecordingTracer()
    entity_queries = _FakeEntityQueries()
    matcher = _CancellableMatcher()
    event_emitter = EventBuffer(entity_queries.agent)

    engine = _make_process_engine(tracer, entity_queries, matcher=matcher)

    task = asyncio.create_task(
        engine.process(
            Context(session_id=entity_queries.session.id, agent_id=entity_queries.agent.id),
            event_emitter,
        )
    )

    await asyncio.wait_for(matcher.started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    interruptions = [a for n, a in tracer.recorded_events if n == "turn.interrupted"]
    assert len(interruptions) == 1
    assert interruptions[0]["cause"] == "cancelled"
    assert interruptions[0]["session_id"] == str(entity_queries.session.id)


async def test_that_stamping_session_labels_emits_an_add_label_event() -> None:
    tracer = _ActivityRecordingTracer()

    context = create_engine_context(conversation=[])
    context.state = ResponseState()
    rule = replace(create_rule("Customer asks for a human"), labels={"handoff"})
    context.state.ordinary_rule_matches.append(RuleEvaluation(rule=rule, reasoning="matched"))

    upserts: list[set[str]] = []

    class _RecordingEntityCommands:
        async def upsert_session_labels(self, session_id: Any, labels: set[str]) -> None:
            upserts.append(labels)

    engine = object.__new__(CompassEngine)
    engine._entity_commands = cast(Any, _RecordingEntityCommands())
    engine._compass_tracer = CompassTracer(tracer)

    await engine._stamp_session_labels(context)

    assert upserts == [{"handoff"}]
    label_events = [a for n, a in tracer.recorded_events if n == "action.add_label"]
    assert len(label_events) == 1
    assert label_events[0]["labels"] == ["handoff"]
    assert label_events[0]["rule_ids"] == [str(rule.id)]


async def test_that_compaction_success_and_failure_emit_events() -> None:
    tracer = _ActivityRecordingTracer()

    context = create_engine_context(conversation=[(EventSource.CUSTOMER, "hi")])
    context.state = ResponseState()

    class _FakeCompacter:
        def __init__(self, fail: bool) -> None:
            self._fail = fail

        async def needs_compaction(self, _: Any) -> bool:
            return True

        async def estimate_session_token_count(self, _: Any) -> int:
            return 1234

        async def compact(self, _: Any) -> Any:
            if self._fail:
                raise RuntimeError("compaction blew up")
            return CompactionResult(
                summary="A concise summary.",
                generation_info=GenerationInfo(
                    schema_name="CompactionSchema",
                    model="fake-model",
                    duration=0.1,
                    usage=UsageInfo(input_tokens=10, output_tokens=5),
                ),
            )

    async def run(fail: bool) -> None:
        engine = object.__new__(CompassEngine)
        engine._compacter = cast(Any, _FakeCompacter(fail))
        engine._logger = context.logger
        engine._tracer = tracer
        engine._compass_tracer = CompassTracer(tracer)

        async def refresh(_: Any) -> None: ...

        engine._refresh_interaction_history = refresh  # type: ignore[method-assign, assignment]
        await engine._compact_if_needed(context)

    await run(fail=False)
    compacted = [a for n, a in tracer.recorded_events if n == "compaction.compacted"]
    assert len(compacted) == 1
    assert compacted[0]["model"] == "fake-model"
    assert compacted[0]["summary_length"] == len("A concise summary.")

    await run(fail=True)
    failed = [a for n, a in tracer.recorded_events if n == "compaction.failed"]
    assert len(failed) == 1
    assert failed[0]["error_type"] == "RuntimeError"


async def test_that_a_committed_turn_flushes_staged_rule_decisions_as_events() -> None:
    tracer = _ActivityRecordingTracer()
    entity_queries = _FakeEntityQueries()
    event_emitter = EventBuffer(entity_queries.agent)

    selected_rule = create_rule("Customer asks about pricing")
    rejected_rule = create_rule("Customer asks about the weather")

    class _StagingMatcher:
        async def match_early(self, context: EngineContext) -> None:
            context.state.matcher.pending_rule_decisions.extend(
                [
                    RuleDecision(rule=selected_rule, selected=True, rationale="on point"),
                    RuleDecision(rule=rejected_rule, selected=False, rationale="off topic"),
                ]
            )

    async def bail(
        hook_context: EngineContext,
        payload: Any,
        exc: Exception | None,
    ) -> CompassHookResult:
        return CompassHookResult.BAIL

    hooks = CompassHooks()
    # on_preparing fires after match_early, so the staged decisions exist by
    # the time the bail path commits and flushes.
    hooks.on_preparing.append(bail)

    engine = _make_process_engine(tracer, entity_queries, matcher=_StagingMatcher(), hooks=hooks)

    await engine.process(
        Context(session_id=entity_queries.session.id, agent_id=entity_queries.agent.id),
        event_emitter,
    )

    selected_events = [a for n, a in tracer.recorded_events if n == "rule.selected"]
    ruled_out_events = [a for n, a in tracer.recorded_events if n == "rule.ruled_out"]

    assert len(selected_events) == 1
    assert selected_events[0]["rule_id"] == str(selected_rule.id)
    assert selected_events[0]["rationale"] == "on point"
    assert selected_events[0]["last_modified"] == selected_rule.modified_utc.isoformat()

    assert len(ruled_out_events) == 1
    assert ruled_out_events[0]["rule_id"] == str(rejected_rule.id)


async def test_that_an_abandoned_turn_never_flushes_rule_decisions() -> None:
    tracer = _ActivityRecordingTracer()
    entity_queries = _FakeEntityQueries()
    event_emitter = EventBuffer(entity_queries.agent)

    staged_rule = create_rule("Customer asks about refunds")
    matcher = _CancellableMatcher()

    original_preload = matcher.preload

    async def staging_preload(context: EngineContext) -> None:
        context.state.matcher.pending_rule_decisions.append(
            RuleDecision(rule=staged_rule, selected=True, rationale="never emitted")
        )
        await original_preload(context)

    matcher.preload = staging_preload  # type: ignore[method-assign]

    engine = _make_process_engine(tracer, entity_queries, matcher=matcher)

    task = asyncio.create_task(
        engine.process(
            Context(session_id=entity_queries.session.id, agent_id=entity_queries.agent.id),
            event_emitter,
        )
    )

    await asyncio.wait_for(matcher.started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert not any(n in ("rule.selected", "rule.ruled_out") for n, _ in tracer.recorded_events)
