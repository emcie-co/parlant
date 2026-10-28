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

import contextvars
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock
import zipfile

import pytest

from parlant.core.diagnostic_reports import (
    DEFAULT_ERROR_HISTORY_SIZE,
    DEFAULT_TRACE_HISTORY_SIZE,
    ERROR_HISTORY_SIZE_ENV_VAR,
    TRACE_HISTORY_SIZE_ENV_VAR,
    DiagnosticReportRepository,
    DiagnosticReportService,
    DiagnosticStore,
    DiagnosticTraceHistory,
    DiagnosticTraceSnapshot,
    RecordingTracer,
    TraceError,
    TraceLogRecord,
    TraceLogBuffer,
    capture_engine_exception,
    resolve_error_history_size,
    resolve_trace_history_size,
)
from parlant.core.common import JSONSerializable
from parlant.core.loggers import CompositeLogger, Logger, LogLevel, StdoutLogger
from parlant.core.tracer import LocalTracer


def _error_snapshot(
    trace_id: str, traceback_text: str = "full traceback"
) -> DiagnosticTraceSnapshot:
    return DiagnosticTraceSnapshot(
        trace_id=trace_id,
        created_utc="2026-01-01T00:00:00+00:00",
        logs=(),
        spans=(),
        error=TraceError(
            traceback=traceback_text,
            exception_type="RuntimeError",
            exception_message="sentinel",
            captured_utc="2026-01-01T00:00:01+00:00",
            session_id="session-1",
        ),
    )


def _repository(
    error_dir: Path,
    *,
    error_history_size: int = 8,
    logger: Logger | None = None,
) -> DiagnosticReportRepository:
    return DiagnosticReportRepository(
        error_report_dir=error_dir,
        trace_report_dir=error_dir / "traces",
        error_history_size=error_history_size,
        logger=logger,
    )


def _service(
    store: DiagnosticStore,
    repository: DiagnosticReportRepository,
    *,
    trace_history_size: int = 0,
    logger: Logger | None = None,
) -> DiagnosticReportService:
    return DiagnosticReportService(
        store=store,
        repository=repository,
        history=DiagnosticTraceHistory(trace_history_size),
        logger=logger,
    )


def test_that_a_follow_up_has_a_separate_diagnostic_lifetime_and_restores_its_parent(
    tmp_path: Path,
) -> None:
    store = DiagnosticStore()
    history = DiagnosticTraceHistory(4)
    service = DiagnosticReportService(
        store=store, repository=_repository(tmp_path), history=history
    )
    tracer = RecordingTracer(LocalTracer(), service)
    with tracer.span("origin"):
        origin = tracer.trace_id
        parent_span = tracer.current_diagnostic_span_id
        service.mark_engine_turn(origin, "session")
        with tracer.fresh_trace("follow_up"):
            follow = tracer.trace_id
            service.mark_engine_turn(follow, "session")
            assert follow != origin
            assert tracer.current_diagnostic_span_id != parent_span
        assert tracer.trace_id == origin
        assert tracer.current_diagnostic_span_id == parent_span
        snapshot = history.get(follow)
        assert snapshot is not None
        assert len(snapshot.snapshot.spans) == 1
        assert snapshot.snapshot.spans[0].parent_id is None
    assert history.get(origin) is not None
    assert tracer.current_diagnostic_span_id is None


def test_that_report_contains_full_trace_logs_spans_and_manifest(tmp_path: Path) -> None:
    writer = _repository(tmp_path / "reports")
    store = DiagnosticStore()
    service = _service(store, writer)
    tracer = RecordingTracer(LocalTracer(), service)
    logger = TraceLogBuffer(tracer, store)
    long_message = "log-" + ("x" * 100_000)

    with tracer.span("http.request", {"http.request.operation": "create_event"}):
        trace_id = tracer.trace_id
        logger.trace("api entry")
        with logger.scope("request"):
            logger.info("scoped api log")
        with tracer.span("process", {"session_id": "session-1"}):
            tracer.set_attribute("phase", "engine")
            tracer.add_event("engine.event", {"attempt": 1})
            logger.debug(long_message)
            try:
                raise RuntimeError("diagnostic-report-sentinel")
            except RuntimeError as exc:
                report_uri = capture_engine_exception(
                    tracer=tracer,
                    session_id="session-1",
                    exception=exc,
                )

        assert report_uri == f"/debug/reports/{trace_id}"
        initial_report_path = writer.resolve_error(trace_id)
        assert initial_report_path is not None
        with zipfile.ZipFile(initial_report_path) as archive:
            assert json.loads(archive.read("manifest.json"))["complete"] is False

    report_path = writer.resolve_error(trace_id)
    assert report_path is not None
    with zipfile.ZipFile(report_path) as archive:
        assert set(archive.namelist()) == {
            "traceback.txt",
            "logs.jsonl",
            "trace.json",
            "manifest.json",
        }
        traceback_text = archive.read("traceback.txt").decode()
        logs = [json.loads(line) for line in archive.read("logs.jsonl").decode().splitlines()]
        trace = json.loads(archive.read("trace.json"))
        manifest = json.loads(archive.read("manifest.json"))

    assert "RuntimeError: diagnostic-report-sentinel" in traceback_text
    assert logs[0]["message"] == "api entry"
    assert logs[0]["level"] == "TRACE"
    assert logs[1]["message"] == "scoped api log"
    assert logs[1]["scope"] == "[request]"
    assert logs[2]["message"] == long_message
    assert len(logs[2]["message"]) == len(long_message)
    assert [span["name"] for span in trace["spans"]] == ["http.request", "process"]
    assert trace["spans"][1]["parent_id"] == trace["spans"][0]["id"]
    assert trace["spans"][1]["attributes"]["phase"] == "engine"
    assert trace["spans"][1]["events"][0]["name"] == "engine.event"
    assert trace["spans"][1]["status"] == "error"
    assert manifest["complete"] is True
    assert manifest["trace_id"] == trace_id
    assert not store.contains(trace_id)


def test_that_retained_background_trace_outlives_http_span(tmp_path: Path) -> None:
    writer = _repository(tmp_path / "reports")
    store = DiagnosticStore()
    service = _service(store, writer)
    tracer = RecordingTracer(LocalTracer(), service)
    logger = TraceLogBuffer(tracer, store)

    with tracer.span("http.request"):
        trace_id = tracer.trace_id
        assert service.retain(trace_id)
        background_context = contextvars.copy_context()

    assert store.contains(trace_id)

    def run_background() -> None:
        with tracer.span("process"):
            logger.info("background-started")
            try:
                raise ValueError("background-sentinel")
            except ValueError as exc:
                capture_engine_exception(tracer, "session-1", exc)
            logger.warning("after-initial-snapshot")
        service.release(trace_id)

    background_context.run(run_background)

    assert not store.contains(trace_id)
    report_path = writer.resolve_error(trace_id)
    assert report_path is not None
    with zipfile.ZipFile(report_path) as archive:
        logs = archive.read("logs.jsonl").decode()
        trace = json.loads(archive.read("trace.json"))
        manifest = json.loads(archive.read("manifest.json"))

    assert "background-started" in logs
    assert "after-initial-snapshot" in logs
    assert [span["name"] for span in trace["spans"]] == ["http.request", "process"]
    assert manifest["complete"] is True


def test_that_concurrent_traces_keep_their_logs_isolated(tmp_path: Path) -> None:
    writer = _repository(tmp_path / "reports")
    store = DiagnosticStore()
    service = _service(store, writer)
    tracer = RecordingTracer(LocalTracer(), service)
    logger = TraceLogBuffer(tracer, store)

    def fail_trace(sentinel: str) -> str:
        with tracer.span("http.request"):
            trace_id = tracer.trace_id
            with logger.scope(sentinel):
                logger.warning(f"log-{sentinel}")
                try:
                    raise RuntimeError(sentinel)
                except RuntimeError as exc:
                    capture_engine_exception(tracer, sentinel, exc)
        return trace_id

    with ThreadPoolExecutor(max_workers=2) as executor:
        trace_ids = list(executor.map(fail_trace, ("first", "second")))

    for trace_id, own, other in zip(
        trace_ids,
        ("first", "second"),
        ("second", "first"),
        strict=True,
    ):
        report_path = writer.resolve_error(trace_id)
        assert report_path is not None
        with zipfile.ZipFile(report_path) as archive:
            logs = archive.read("logs.jsonl").decode()
        assert f"log-{own}" in logs
        assert f"[{own}]" in logs
        assert f"log-{other}" not in logs


def test_that_a_full_chained_traceback_is_preserved(tmp_path: Path) -> None:
    writer = _repository(tmp_path / "reports")
    store = DiagnosticStore()
    tracer = RecordingTracer(LocalTracer(), _service(store, writer))

    with tracer.span("process"):
        trace_id = tracer.trace_id
        try:
            try:
                raise ValueError("inner-sentinel")
            except ValueError as inner:
                raise RuntimeError("outer-sentinel") from inner
        except RuntimeError as exc:
            capture_engine_exception(tracer, "session-1", exc)

    report_path = writer.resolve_error(trace_id)
    assert report_path is not None
    with zipfile.ZipFile(report_path) as archive:
        traceback_text = archive.read("traceback.txt").decode()

    assert "ValueError: inner-sentinel" in traceback_text
    assert "RuntimeError: outer-sentinel" in traceback_text
    assert "The above exception was the direct cause" in traceback_text


def test_that_a_failed_final_rewrite_preserves_the_initial_snapshot(
    tmp_path: Path,
) -> None:
    class FinalizationFailingRepository(DiagnosticReportRepository):
        def write_error(
            self,
            snapshot: DiagnosticTraceSnapshot,
            *,
            complete: bool,
            **kwargs: Any,
        ) -> Path:
            if complete:
                raise OSError("finalization-sentinel")
            return super().write_error(snapshot, complete=complete, **kwargs)

    writer = FinalizationFailingRepository(
        error_report_dir=tmp_path / "reports",
        trace_report_dir=tmp_path / "traces",
        error_history_size=8,
    )
    store = DiagnosticStore()
    service_logger = Mock(spec=Logger)
    tracer = RecordingTracer(
        LocalTracer(),
        _service(store, writer, logger=cast(Logger, service_logger)),
    )

    with tracer.span("process"):
        trace_id = tracer.trace_id
        try:
            raise RuntimeError("initial-snapshot-sentinel")
        except RuntimeError as exc:
            capture_engine_exception(tracer, "session-1", exc)

    report_path = writer.resolve_error(trace_id)
    assert report_path is not None
    with zipfile.ZipFile(report_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        traceback_text = archive.read("traceback.txt").decode()

    assert manifest["complete"] is False
    assert "initial-snapshot-sentinel" in traceback_text
    assert not store.contains(trace_id)
    assert any(
        "failed to finalize diagnostic report" in call.args[0]
        and trace_id in call.args[0]
        and "finalization-sentinel" in call.args[0]
        for call in service_logger.error.call_args_list
    )


def test_that_service_logs_info_after_successfully_finalizing_a_report(
    tmp_path: Path,
) -> None:
    writer = _repository(tmp_path / "reports")
    store = DiagnosticStore()
    service_logger = Mock(spec=Logger)
    tracer = RecordingTracer(
        LocalTracer(),
        _service(store, writer, logger=cast(Logger, service_logger)),
    )

    with tracer.span("process"):
        trace_id = tracer.trace_id
        try:
            raise RuntimeError("successful-finalization-sentinel")
        except RuntimeError as exc:
            capture_engine_exception(tracer, "session-1", exc)

    service_logger.info.assert_called_once()
    completion_message = service_logger.info.call_args.args[0]
    assert "successfully finalized" in completion_message
    assert trace_id in completion_message
    assert str(writer.resolve_error(trace_id)) in completion_message
    service_logger.error.assert_not_called()


def test_that_service_logs_error_when_initial_report_creation_fails(
    tmp_path: Path,
) -> None:
    class FailingRepository(DiagnosticReportRepository):
        def write_error(
            self,
            snapshot: DiagnosticTraceSnapshot,
            *,
            complete: bool,
            **kwargs: Any,
        ) -> Path:
            raise OSError("initial-write-sentinel")

    store = DiagnosticStore()
    service_logger = Mock(spec=Logger)
    service = DiagnosticReportService(
        store,
        FailingRepository(
            error_report_dir=tmp_path / "reports",
            trace_report_dir=tmp_path / "traces",
            error_history_size=8,
        ),
        logger=cast(Logger, service_logger),
    )
    trace_id = "initial-write-failure"
    store.start(trace_id)

    report_uri = service.capture_exception(
        trace_id=trace_id,
        session_id="session-1",
        exception=RuntimeError("engine-sentinel"),
        formatted_traceback="full traceback",
        span_id=None,
    )

    assert report_uri is None
    service_logger.error.assert_called_once()
    failure_message = service_logger.error.call_args.args[0]
    assert "failed to create initial diagnostic report" in failure_message
    assert trace_id in failure_message
    assert "initial-write-sentinel" in failure_message


def test_that_successful_trace_is_discarded_without_a_report(tmp_path: Path) -> None:
    writer = _repository(tmp_path / "reports")
    store = DiagnosticStore()
    tracer = RecordingTracer(LocalTracer(), _service(store, writer))

    with tracer.span("http.request"):
        trace_id = tracer.trace_id

    assert not store.contains(trace_id)
    assert writer.resolve_error(trace_id) is None


def test_that_zero_trace_and_error_history_disables_report_persistence(tmp_path: Path) -> None:
    repository = _repository(
        tmp_path / "reports",
        error_history_size=0,
    )
    store = DiagnosticStore()
    service = _service(store, repository)
    tracer = RecordingTracer(LocalTracer(), service)

    with tracer.span("process"):
        trace_id = tracer.trace_id
        try:
            raise RuntimeError("zero-history-sentinel")
        except RuntimeError as exc:
            report_uri = capture_engine_exception(tracer, "session-1", exc)

    assert service.enabled is False
    assert report_uri is None
    assert repository.resolve_error(trace_id) is None
    assert not store.contains(trace_id)


def test_that_successful_engine_turn_is_materialized_only_on_request(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "reports")
    store = DiagnosticStore()
    service = _service(store, repository, trace_history_size=8)
    tracer = RecordingTracer(LocalTracer(), service)
    logger = TraceLogBuffer(tracer, store)

    with tracer.span("http.request"):
        trace_id = tracer.trace_id
        assert service.mark_engine_turn(trace_id, "session-1")
        logger.info("successful-turn-sentinel")
        with tracer.span("process", {"session_id": "session-1"}):
            pass

    assert not repository.trace_report_dir.exists()
    reports = service.list_reports(outcome="success")
    assert [report.trace_id for report in reports] == [trace_id]
    assert reports[0].size_bytes is None

    prepared = service.prepare_report(trace_id)
    assert prepared is not None
    assert prepared.delete_after_stream is True
    with zipfile.ZipFile(prepared.path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        logs = [json.loads(line) for line in archive.read("logs.jsonl").decode().splitlines()]
        trace = json.loads(archive.read("trace.json"))

        assert "traceback.txt" not in archive.namelist()
    prepared.path.unlink()

    assert manifest["complete"] is True
    assert manifest["outcome"] == "success"
    assert manifest["exception"] is None
    assert manifest["session_id"] == "session-1"
    assert [record["message"] for record in logs] == ["successful-turn-sentinel"]
    assert [span["name"] for span in trace["spans"]] == ["http.request", "process"]


def test_that_diagnostic_logs_ignore_the_server_output_level(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "reports")
    store = DiagnosticStore()
    service = _service(store, repository, trace_history_size=1)
    tracer = RecordingTracer(LocalTracer(), service)
    logger = CompositeLogger([TraceLogBuffer(tracer, store)])
    logger.set_level(LogLevel.ERROR)

    with tracer.span("http.request"):
        trace_id = tracer.trace_id
        assert service.mark_engine_turn(trace_id, "session-1")
        logger.trace("trace-sentinel")
        logger.debug("debug-sentinel")
        logger.info("info-sentinel")
        logger.warning("warning-sentinel")
        logger.error("error-sentinel")
        logger.critical("critical-sentinel")

    prepared = service.prepare_report(trace_id)
    assert prepared is not None
    with zipfile.ZipFile(prepared.path) as archive:
        logs = [json.loads(line) for line in archive.read("logs.jsonl").decode().splitlines()]
    prepared.path.unlink()

    assert [(record["level"], record["message"]) for record in logs] == [
        ("TRACE", "trace-sentinel"),
        ("DEBUG", "debug-sentinel"),
        ("INFO", "info-sentinel"),
        ("WARNING", "warning-sentinel"),
        ("ERROR", "error-sentinel"),
        ("CRITICAL", "critical-sentinel"),
    ]


def test_that_ordinary_http_trace_does_not_enter_trace_history(tmp_path: Path) -> None:
    repository = _repository(tmp_path / "reports")
    store = DiagnosticStore()
    tracer = RecordingTracer(
        LocalTracer(),
        _service(store, repository, trace_history_size=8),
    )

    with tracer.span("http.request"):
        trace_id = tracer.trace_id

    assert repository.resolve_error(trace_id) is None
    assert tracer.diagnostic_report_service.list_reports() == ()


def test_that_retained_trace_is_exported_only_after_background_work_finishes(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "reports")
    store = DiagnosticStore()
    service = _service(store, repository, trace_history_size=8)
    tracer = RecordingTracer(LocalTracer(), service)

    with tracer.span("http.request"):
        trace_id = tracer.trace_id
        service.mark_engine_turn(trace_id, "session-1")
        assert service.retain(trace_id)

    assert service.prepare_report(trace_id) is None
    store.add_log(
        TraceLogRecord(
            timestamp="2026-01-01T00:00:02+00:00",
            level="INFO",
            trace_id=trace_id,
            span_id="background",
            scope="",
            message="late-background-sentinel",
        )
    )

    service.release(trace_id)

    assert not repository.trace_report_dir.exists()
    prepared = service.prepare_report(trace_id)
    assert prepared is not None
    with zipfile.ZipFile(prepared.path) as archive:
        logs = [json.loads(line) for line in archive.read("logs.jsonl").decode().splitlines()]
    prepared.path.unlink()
    assert [record["message"] for record in logs] == ["late-background-sentinel"]


def test_that_lazy_materialization_failure_does_not_fail_a_successful_turn(
    tmp_path: Path,
) -> None:
    class FailingRepository(DiagnosticReportRepository):
        def materialize(
            self,
            snapshot: DiagnosticTraceSnapshot,
            finalized_utc: str,
        ) -> Path:
            raise RuntimeError("trace-write-sentinel")

    store = DiagnosticStore()
    service_logger = Mock(spec=Logger)
    service = DiagnosticReportService(
        store=store,
        repository=FailingRepository(
            error_report_dir=tmp_path / "reports",
            trace_report_dir=tmp_path / "traces",
            error_history_size=8,
        ),
        history=DiagnosticTraceHistory(8),
        logger=cast(Logger, service_logger),
    )
    tracer = RecordingTracer(LocalTracer(), service)

    with tracer.span("http.request"):
        trace_id = tracer.trace_id
        service.mark_engine_turn(trace_id, "session-1")

    service_logger.error.assert_not_called()
    assert service.prepare_report(trace_id) is None
    service_logger.error.assert_called_once()
    message = service_logger.error.call_args.args[0]
    assert "materialize diagnostic report" in message
    assert trace_id in message
    assert "trace-write-sentinel" in message


def test_that_trace_history_retention_does_not_evict_error_reports(tmp_path: Path) -> None:
    repository = _repository(
        tmp_path / "reports",
        error_history_size=8,
    )
    store = DiagnosticStore()
    service = _service(store, repository, trace_history_size=2)
    tracer = RecordingTracer(LocalTracer(), service)

    error_trace_id = ""
    for index in range(3):
        with tracer.span("http.request"):
            trace_id = tracer.trace_id
            service.mark_engine_turn(trace_id, f"session-{index}")
            if index == 0:
                error_trace_id = trace_id
                try:
                    raise RuntimeError("retained-failure-sentinel")
                except RuntimeError as exc:
                    capture_engine_exception(tracer, f"session-{index}", exc)

        if index == 0:
            assert service.list_reports()[0].outcome == "error"

    assert repository.resolve_error(error_trace_id) is not None
    assert [item.session_id for item in service.list_reports(outcome="success")] == [
        "session-2",
        "session-1",
    ]
    assert [item.session_id for item in service.list_reports(outcome="error")] == ["session-0"]
    assert not (tmp_path / "reports" / "traces" / f"{error_trace_id}.zip").exists()


def test_that_trace_history_retains_errors_when_error_history_is_zero(tmp_path: Path) -> None:
    repository = _repository(
        tmp_path / "reports",
        error_history_size=0,
    )
    store = DiagnosticStore()
    service = _service(store, repository, trace_history_size=2)
    tracer = RecordingTracer(LocalTracer(), service)

    with tracer.span("http.request"):
        trace_id = tracer.trace_id
        service.mark_engine_turn(trace_id, "session-1")
        try:
            raise RuntimeError("trace-window-error-sentinel")
        except RuntimeError as exc:
            report_uri = capture_engine_exception(tracer, "session-1", exc)

    assert report_uri == f"/debug/reports/{trace_id}"
    assert repository.resolve_error(trace_id) is not None
    assert [report.outcome for report in service.list_reports()] == ["error"]


def test_that_an_error_is_pruned_after_leaving_a_trace_only_history(
    tmp_path: Path,
) -> None:
    repository = _repository(tmp_path / "reports", error_history_size=0)
    store = DiagnosticStore()
    service = _service(store, repository, trace_history_size=2)
    tracer = RecordingTracer(LocalTracer(), service)

    with tracer.span("http.request"):
        error_trace_id = tracer.trace_id
        service.mark_engine_turn(error_trace_id, "session-error")
        try:
            raise RuntimeError("eventually-evicted-error")
        except RuntimeError as exc:
            capture_engine_exception(tracer, "session-error", exc)

    assert repository.resolve_error(error_trace_id) is not None

    for index in range(2):
        with tracer.span("http.request"):
            service.mark_engine_turn(tracer.trace_id, f"session-success-{index}")

    assert repository.resolve_error(error_trace_id) is None
    assert [report.session_id for report in service.list_reports()] == [
        "session-success-1",
        "session-success-0",
    ]


def test_that_writer_keeps_only_latest_k_reports_and_preserves_unrelated_files(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "reports"
    writer = _repository(report_dir, error_history_size=2)
    unrelated = report_dir / "notes.txt"
    suspicious = report_dir / "bad.zip"
    report_dir.mkdir()
    unrelated.write_text("keep")
    suspicious.mkdir()

    first = writer.write_error(_error_snapshot("trace-1"), complete=True)
    second = writer.write_error(_error_snapshot("trace-2"), complete=True)
    third = writer.write_error(_error_snapshot("trace-3"), complete=True)

    assert not first.exists()
    assert second.exists()
    assert third.exists()
    assert unrelated.read_text() == "keep"
    assert suspicious.is_dir()

    writer.write_error(
        replace(_error_snapshot("trace-3"), created_utc="later"),
        complete=True,
    )
    assert sorted(path.name for path in report_dir.glob("trace-*.zip")) == [
        "trace-2.zip",
        "trace-3.zip",
    ]


def test_that_report_resolution_rejects_invalid_ids_and_symlinks(tmp_path: Path) -> None:
    report_dir = tmp_path / "reports"
    writer = _repository(report_dir)
    outside = tmp_path / "outside.zip"
    outside.write_bytes(b"outside")
    report_dir.mkdir()
    symlink = report_dir / "linked.zip"

    try:
        symlink.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are not supported on this platform")

    assert writer.resolve_error("../outside") is None
    assert writer.resolve_error("linked") is None
    assert writer.resolve_error("missing") is None


def test_that_a_symlinked_report_directory_is_neither_written_nor_served(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    report_dir = tmp_path / "reports"

    try:
        report_dir.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are not supported on this platform")

    writer = _repository(report_dir)
    writer.prune()

    assert writer.resolve_error("trace-1") is None
    with pytest.raises(RuntimeError, match="symbolic link"):
        writer.write_error(_error_snapshot("trace-1"), complete=True)
    assert not list(outside.iterdir())


def test_that_error_history_size_defaults_and_honors_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(ERROR_HISTORY_SIZE_ENV_VAR, raising=False)
    assert resolve_error_history_size() == DEFAULT_ERROR_HISTORY_SIZE == 8

    monkeypatch.setenv(ERROR_HISTORY_SIZE_ENV_VAR, "3")
    assert resolve_error_history_size() == 3
    monkeypatch.setenv(ERROR_HISTORY_SIZE_ENV_VAR, "0")
    assert resolve_error_history_size() == 0

    for invalid in ("-1", "not-a-number"):
        monkeypatch.setenv(ERROR_HISTORY_SIZE_ENV_VAR, invalid)
        with pytest.raises(ValueError, match=ERROR_HISTORY_SIZE_ENV_VAR):
            resolve_error_history_size()


def test_that_trace_history_size_defaults_to_32_and_honors_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(TRACE_HISTORY_SIZE_ENV_VAR, raising=False)
    assert resolve_trace_history_size() == DEFAULT_TRACE_HISTORY_SIZE == 32

    monkeypatch.setenv(TRACE_HISTORY_SIZE_ENV_VAR, "3")
    assert resolve_trace_history_size() == 3
    monkeypatch.setenv(TRACE_HISTORY_SIZE_ENV_VAR, "0")
    assert resolve_trace_history_size() == 0

    for invalid in ("-1", "not-a-number"):
        monkeypatch.setenv(TRACE_HISTORY_SIZE_ENV_VAR, invalid)
        with pytest.raises(ValueError, match=TRACE_HISTORY_SIZE_ENV_VAR):
            resolve_trace_history_size()


def test_that_repository_rejects_a_negative_error_history_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        DiagnosticReportRepository(
            error_report_dir=tmp_path / "errors",
            trace_report_dir=tmp_path / "traces",
            error_history_size=-1,
        )


def test_that_trace_history_rejects_a_negative_capacity() -> None:
    with pytest.raises(ValueError):
        DiagnosticTraceHistory(-1)


def test_that_archives_are_owner_readable_and_writable_only(tmp_path: Path) -> None:
    writer = _repository(tmp_path / "reports")
    report = writer.write_error(_error_snapshot("trace-permissions"), complete=True)

    if os.name == "posix":
        assert report.stat().st_mode & 0o777 == 0o600


def test_that_attached_session_data_is_exported_only_in_the_complete_report(
    tmp_path: Path,
) -> None:
    writer = _repository(tmp_path / "reports")
    store = DiagnosticStore()
    service = _service(store, writer)
    tracer = RecordingTracer(LocalTracer(), service)

    session_export: JSONSerializable = {
        "id": "session-1",
        "metadata": {"discovered_rule_ids": ["rule-1"]},
    }
    event_exports: list[JSONSerializable] = [
        {"offset": 0, "kind": "message", "source": "customer", "data": {"message": "hi"}},
        {"offset": 1, "kind": "status", "source": "ai_agent", "data": {"status": "error"}},
    ]

    with tracer.span("http.request"):
        trace_id = tracer.trace_id
        with tracer.span("process", {"session_id": "session-1"}):
            try:
                raise RuntimeError("session-export-sentinel")
            except RuntimeError as exc:
                capture_engine_exception(tracer=tracer, session_id="session-1", exception=exc)

        # The initial (incomplete) report was written at capture time,
        # before any session data could be attached.
        initial_path = writer.resolve_error(trace_id)
        assert initial_path is not None
        with zipfile.ZipFile(initial_path) as archive:
            assert "session.json" not in archive.namelist()
            assert "events.jsonl" not in archive.namelist()

        # The dispatcher attaches the store's view after processing ends.
        service.attach_session_data(trace_id, session=session_export, events=event_exports)

    report_path = writer.resolve_error(trace_id)
    assert report_path is not None
    with zipfile.ZipFile(report_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        session_doc = json.loads(archive.read("session.json"))
        events = [json.loads(line) for line in archive.read("events.jsonl").decode().splitlines()]

    assert manifest["complete"] is True
    assert "session.json" in manifest["files"]
    assert "events.jsonl" in manifest["files"]
    assert session_doc == session_export
    assert events == event_exports


def test_that_writer_prune_failures_log_through_the_container_logger(
    tmp_path: Path,
) -> None:
    # The writer logs cleanup failures through the injected Logger — the
    # builtin logging module is not used anywhere in diagnostic reporting.
    recorded: list[str] = []

    class _RecordingLogger(StdoutLogger):
        def error(self, message: str, *args: Any, **kwargs: Any) -> None:
            recorded.append(message)

    tracer = LocalTracer()
    report_dir = tmp_path / "reports"
    writer = _repository(report_dir, logger=_RecordingLogger(tracer))
    writer.write_error(_error_snapshot("trace-a"), complete=True)
    writer.write_error(_error_snapshot("trace-b"), complete=True)

    # Tighten the limit, then make the directory read-only so the over-limit
    # archive's unlink fails with EACCES.
    writer.error_history_size = 1
    report_dir.chmod(0o500)
    try:
        writer.prune()
    finally:
        report_dir.chmod(0o700)

    assert any("prune" in message.lower() for message in recorded)


def test_that_service_diagnostics_do_not_feed_the_trace_buffer(tmp_path: Path) -> None:
    # The service's own failure messages must not loop back into
    # the diagnostic store through TraceLogBuffer — server wiring hands the
    # service a logger view that excludes the buffer.
    store = DiagnosticStore()

    class _ExplodingRepository(DiagnosticReportRepository):
        def write_error(
            self,
            snapshot: DiagnosticTraceSnapshot,
            *,
            complete: bool,
            **kwargs: Any,
        ) -> Path:
            raise RuntimeError("writer-exploded")

    tracer_sink = LocalTracer()
    file_logger = StdoutLogger(tracer_sink)
    writer = _ExplodingRepository(
        error_report_dir=tmp_path / "reports",
        trace_report_dir=tmp_path / "traces",
        error_history_size=8,
        logger=file_logger,
    )

    service = DiagnosticReportService(store=store, repository=writer, logger=file_logger)
    tracer = RecordingTracer(LocalTracer(), service)
    buffer = TraceLogBuffer(tracer=tracer, store=store)
    composite = CompositeLogger([file_logger, buffer])

    # The wiring seam under test: a composite view without the trace buffer.
    quiet_view = composite.excluding(buffer)
    service._logger = quiet_view

    with tracer.span("http.request"):
        trace_id = tracer.trace_id
        with tracer.span("process"):
            try:
                raise RuntimeError("feedback-sentinel")
            except RuntimeError as exc:
                capture_engine_exception(tracer=tracer, session_id="session-1", exception=exc)

        # The failed capture logged an error through the service's logger;
        # it must NOT have been recorded into the trace's own diagnostics.
        snapshot = store.release(trace_id)

    assert snapshot is not None
    assert not any("writer-exploded" in record.message for record in snapshot.logs)
