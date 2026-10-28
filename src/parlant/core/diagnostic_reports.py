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

"""Capture, retain, and export trace-scoped diagnostic reports.

Completed engine turns are retained as bounded in-memory snapshots and are
materialized as temporary ZIP archives only when requested through the API.
Error reports are different: they are written immediately at the engine error
boundary and atomically finalized when the trace ends. Persistent error
retention is the union of errors still in the recent-turn window and the newest
configured number of errors.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Generator
from contextlib import contextmanager, AbstractContextManager
import contextvars
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import threading
import traceback
from typing import IO, Mapping, Sequence
from uuid import uuid4
import zipfile

from typing_extensions import override

from parlant.core.common import JSONSerializable
from parlant.core.loggers import LogLevel, Logger
from parlant.core.tracer import AttributeValue, Tracer, TraceContext
from parlant.core.version import VERSION


DEFAULT_ERROR_HISTORY_SIZE = 8
ERROR_HISTORY_SIZE_ENV_VAR = "PARLANT_ERROR_HISTORY_SIZE"
DEFAULT_TRACE_HISTORY_SIZE = 32
TRACE_HISTORY_SIZE_ENV_VAR = "PARLANT_TRACE_HISTORY_SIZE"
DIAGNOSTIC_REPORT_URI_METADATA_KEY = "diagnostic_report_uri"

_TRACE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_REPORT_FILE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}\.zip$")


def resolve_error_history_size() -> int:
    raw_value = os.environ.get(ERROR_HISTORY_SIZE_ENV_VAR)
    if raw_value is None:
        return DEFAULT_ERROR_HISTORY_SIZE

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{ERROR_HISTORY_SIZE_ENV_VAR} must be a non-negative integer") from exc

    if value < 0:
        raise ValueError(f"{ERROR_HISTORY_SIZE_ENV_VAR} must be a non-negative integer")

    return value


def resolve_trace_history_size() -> int:
    """Resolve the number of recent completed engine turns to retain.

    The default is 32. Zero disables rolling trace retention while leaving
    error-report retention unchanged.
    """

    raw_value = os.environ.get(TRACE_HISTORY_SIZE_ENV_VAR)
    if raw_value is None:
        return DEFAULT_TRACE_HISTORY_SIZE

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{TRACE_HISTORY_SIZE_ENV_VAR} must be a non-negative integer") from exc

    if value < 0:
        raise ValueError(f"{TRACE_HISTORY_SIZE_ENV_VAR} must be a non-negative integer")

    return value


def is_valid_trace_id(trace_id: str) -> bool:
    return bool(_TRACE_ID_PATTERN.fullmatch(trace_id))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TraceLogRecord:
    """One unfiltered logger call captured for a live root trace.

    :class:`TraceLogBuffer` creates these records with the current trace, span
    path, and logger scope. :class:`DiagnosticStore` keeps them in call order,
    and :class:`DiagnosticReportRepository` serializes them as ``logs.jsonl``.
    """

    timestamp: str
    level: str
    trace_id: str
    span_id: str
    scope: str
    message: str


@dataclass
class TraceSpanEvent:
    """A timestamped event attached to a local diagnostic span.

    :class:`RecordingTracer` creates this record when application code calls
    ``Tracer.add_event()``. The store attaches it to a
    :class:`TraceSpanRecord`, which the writer later emits in ``trace.json``.
    """

    timestamp: str
    name: str
    attributes: Mapping[str, AttributeValue]


@dataclass
class TraceSpanRecord:
    """The module's exporter-independent representation of one traced span.

    :class:`RecordingTracer` starts and ends these records while still
    delegating the real span to the configured tracer. Parent IDs reconstruct
    nesting, mutable attributes and events mirror tracer updates, and exception
    fields describe errors observed while exiting the span. The diagnostic
    store owns live records; the writer serializes their snapshots.
    """

    id: str
    parent_id: str | None
    name: str
    start_utc: str
    end_utc: str | None
    attributes: dict[str, AttributeValue]
    events: list[TraceSpanEvent] = field(default_factory=list)
    status: str = "ok"
    exception_type: str | None = None
    exception_message: str | None = None


@dataclass
class TraceError:
    """The engine error that makes a trace eligible for a report.

    :class:`DiagnosticReportService.capture_exception` asks the store to
    create this record at the engine's top-level catch boundary. It preserves
    the full formatted traceback, exception identity, session, and capture
    time for ``traceback.txt`` and ``manifest.json``.
    """

    traceback: str
    exception_type: str
    exception_message: str
    captured_utc: str
    session_id: str


@dataclass(frozen=True)
class SessionExport:
    """Persisted session state for ``session.json`` and ``events.jsonl``.

    ``SessionModule`` builds this from the session store AFTER processing ends
    (so the stream includes the turn's own error event) and attaches it via
    :meth:`DiagnosticReportService.attach_session_data`. The payload must be detached,
    JSON-serializable data. The live store keeps a reference; completed
    snapshots deep-copy it before entering history or archive generation.
    """

    session: JSONSerializable
    events: Sequence[JSONSerializable]


@dataclass
class DiagnosticTrace:
    """Mutable in-memory state for one root trace.

    :class:`DiagnosticStore` creates this state when
    :class:`RecordingTracer` enters a root span. The recording tracer and log
    buffer append spans and logs, the report service marks an error, and
    the ``owners`` count keeps the state alive across HTTP and background-task
    lifetimes. Callers never serialize this mutable object directly.
    """

    trace_id: str
    created_utc: str
    owners: int = 0
    logs: list[TraceLogRecord] = field(default_factory=list)
    spans: list[TraceSpanRecord] = field(default_factory=list)
    error: TraceError | None = None
    session_export: SessionExport | None = None
    engine_turn: bool = False
    session_id: str | None = None


@dataclass(frozen=True)
class DiagnosticTraceSnapshot:
    """A detached view of a trace suitable for retention or archive writing.

    The store deep-copies live diagnostics into this object both when the
    engine first fails and when the last owner releases the trace. Initial and
    final error snapshots are archived eagerly; completed turn snapshots are
    retained in memory for possible export. In both cases archive creation is
    isolated from concurrent mutation.
    """

    trace_id: str
    created_utc: str
    logs: Sequence[TraceLogRecord]
    spans: Sequence[TraceSpanRecord]
    error: TraceError | None = None
    session_export: SessionExport | None = None
    engine_turn: bool = False
    session_id: str | None = None


@dataclass(frozen=True)
class DiagnosticReportMetadata:
    """Summary metadata for one finalized report available through the API."""

    trace_id: str
    session_id: str | None
    outcome: str
    created_utc: str
    finalized_utc: str
    size_bytes: int | None


@dataclass(frozen=True)
class CompletedDiagnosticTrace:
    """One finalized engine-turn snapshot retained for on-demand export."""

    snapshot: DiagnosticTraceSnapshot
    finalized_utc: str


class DiagnosticTraceHistory:
    """A concurrency-safe, bounded history of completed engine turns."""

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("capacity must be at least 0")
        self.capacity = capacity
        self._entries: OrderedDict[str, CompletedDiagnosticTrace] = OrderedDict()
        self._lock = threading.RLock()

    def add(
        self,
        snapshot: DiagnosticTraceSnapshot,
        finalized_utc: str | None = None,
    ) -> CompletedDiagnosticTrace | None:
        """Retain a finalized snapshot and return the entry evicted by it, if any."""

        if self.capacity == 0:
            return None

        entry = CompletedDiagnosticTrace(
            snapshot=snapshot,
            finalized_utc=finalized_utc or _utc_now(),
        )
        with self._lock:
            self._entries.pop(snapshot.trace_id, None)
            self._entries[snapshot.trace_id] = entry
            if len(self._entries) > self.capacity:
                return self._entries.popitem(last=False)[1]
        return None

    def get(self, trace_id: str) -> CompletedDiagnosticTrace | None:
        with self._lock:
            return self._entries.get(trace_id)

    def list(self) -> Sequence[CompletedDiagnosticTrace]:
        """Return retained entries newest first without changing recency."""

        with self._lock:
            return tuple(reversed(self._entries.values()))

    def trace_ids(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._entries)


@dataclass(frozen=True)
class PreparedDiagnosticReport:
    """An archive ready to stream, optionally requiring post-stream cleanup."""

    path: Path
    delete_after_stream: bool


class DiagnosticStore:
    """Concurrency-safe registry of live diagnostics keyed by root trace ID.

    ``start()`` creates or owns a trace, ``retain()`` extends its lifetime for
    background processing, and ``release()`` removes it when the final owner
    exits. A final release returns a snapshot for an errored trace or a trace
    explicitly selected for rolling history; ordinary successful requests
    leave neither memory nor disk artifacts.

    :class:`RecordingTracer` uses the span and ownership methods,
    :class:`TraceLogBuffer` appends log records, and
    :class:`DiagnosticReportService` marks errors and consumes final snapshots. An
    ``RLock`` protects these callers across async tasks and worker threads.
    """

    def __init__(self) -> None:
        """Create an empty registry guarded by one process-local reentrant lock."""

        self._traces: dict[str, DiagnosticTrace] = {}
        self._lock = threading.RLock()

    def start(self, trace_id: str) -> None:
        """Create a trace if needed and acquire one ownership lease.

        ``RecordingTracer.span()`` calls this when it enters an outermost span.
        Invalid trace IDs are ignored because they cannot be exported safely.
        """

        if not is_valid_trace_id(trace_id):
            return

        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is None:
                trace = DiagnosticTrace(trace_id=trace_id, created_utc=_utc_now())
                self._traces[trace_id] = trace
            trace.owners += 1

    def retain(self, trace_id: str) -> bool:
        """Acquire another lease for an existing trace.

        ``SessionModule`` uses this before scheduling background ``process()``
        work. ``False`` means no live trace was available to retain.
        """

        with self._lock:
            if trace := self._traces.get(trace_id):
                trace.owners += 1
                return True
            return False

    def release(self, trace_id: str) -> DiagnosticTraceSnapshot | None:
        """Release one lease and dispose of a trace after its final owner.

        Intermediate releases return ``None``. A final release also returns
        ``None`` for an ordinary successful trace, but returns a detached
        snapshot for errors and turns selected for trace-history export.
        """

        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is None:
                return None

            trace.owners = max(0, trace.owners - 1)
            if trace.owners:
                return None

            self._traces.pop(trace_id, None)
            if trace.error is None and not trace.engine_turn:
                return None

            return self._snapshot(trace)

    def mark_engine_turn(self, trace_id: str, session_id: str) -> bool:
        """Mark a live trace as an engine turn eligible for history retention."""

        with self._lock:
            if trace := self._traces.get(trace_id):
                trace.engine_turn = True
                trace.session_id = session_id
                return True
            return False

    def add_log(self, record: TraceLogRecord) -> None:
        """Append a log if its trace is still live.

        ``TraceLogBuffer`` calls this for every logger invocation. A record
        arriving after final release is intentionally ignored.
        """

        with self._lock:
            if trace := self._traces.get(record.trace_id):
                trace.logs.append(record)

    def start_span(
        self,
        trace_id: str,
        span_id: str,
        parent_id: str | None,
        name: str,
        attributes: Mapping[str, AttributeValue],
    ) -> None:
        """Append an open diagnostic span to a live trace.

        ``RecordingTracer`` supplies a locally generated ID and its current
        diagnostic parent whenever application code enters a tracer span.
        """

        with self._lock:
            if trace := self._traces.get(trace_id):
                trace.spans.append(
                    TraceSpanRecord(
                        id=span_id,
                        parent_id=parent_id,
                        name=name,
                        start_utc=_utc_now(),
                        end_utc=None,
                        attributes=dict(attributes),
                    )
                )

    def end_span(
        self,
        trace_id: str,
        span_id: str,
        exception: BaseException | None,
    ) -> None:
        """Close a diagnostic span and record an escaping exception, if any.

        ``RecordingTracer`` invokes this from its span context manager's
        ``finally`` block before releasing a root owner.
        """

        with self._lock:
            if span := self._find_span(trace_id, span_id):
                span.end_utc = _utc_now()
                if exception is not None:
                    span.status = "error"
                    span.exception_type = type(exception).__name__
                    span.exception_message = str(exception)

    def update_span_attributes(
        self,
        trace_id: str,
        span_id: str | None,
        attributes: Mapping[str, AttributeValue],
    ) -> None:
        """Merge tracer attributes into the current diagnostic span.

        Calls outside a recorded span have no diagnostic span ID and are
        ignored. ``RecordingTracer.attributes()`` and ``set_attribute()`` are
        the normal callers.
        """

        if span_id is None:
            return

        with self._lock:
            if span := self._find_span(trace_id, span_id):
                span.attributes.update(attributes)

    def add_span_event(
        self,
        trace_id: str,
        span_id: str | None,
        name: str,
        attributes: Mapping[str, AttributeValue],
    ) -> None:
        """Attach a timestamped tracer event to the current diagnostic span.

        ``RecordingTracer.add_event()`` delegates the event to the configured
        tracer and mirrors it here. Calls without a current span are ignored.
        """

        if span_id is None:
            return

        with self._lock:
            if span := self._find_span(trace_id, span_id):
                span.events.append(
                    TraceSpanEvent(
                        timestamp=_utc_now(),
                        name=name,
                        attributes=dict(attributes),
                    )
                )

    def mark_error(
        self,
        trace_id: str,
        session_id: str,
        exception: Exception,
        formatted_traceback: str,
        span_id: str | None,
    ) -> DiagnosticTraceSnapshot | None:
        """Mark a live trace errored and return its initial archive snapshot.

        ``DiagnosticReportService.capture_exception()`` calls this at the engine catch
        boundary. The current span is also marked errored when supplied.
        ``None`` means the trace was no longer live.
        """

        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is None:
                return None

            trace.error = TraceError(
                traceback=formatted_traceback,
                exception_type=type(exception).__name__,
                exception_message=str(exception),
                captured_utc=_utc_now(),
                session_id=session_id,
            )
            trace.engine_turn = True
            trace.session_id = session_id

            if span_id and (span := self._find_span(trace_id, span_id)):
                span.status = "error"
                span.exception_type = type(exception).__name__
                span.exception_message = str(exception)

            return self._snapshot(trace)

    def attach_session_export(self, trace_id: str, export: SessionExport) -> None:
        """Attach persisted session state to a live diagnostic trace.

        A no-op for traces that are gone, have no error, and were not selected
        for trace-history export.
        """

        with self._lock:
            trace = self._traces.get(trace_id)
            if trace is not None and (trace.error is not None or trace.engine_turn):
                trace.session_export = export

    def should_attach_session_data(self, trace_id: str) -> bool:
        """Whether a pending error/history export should include session data."""

        with self._lock:
            trace = self._traces.get(trace_id)
            return trace is not None and (trace.error is not None or trace.engine_turn)

    def contains(self, trace_id: str) -> bool:
        """Return whether diagnostics are currently accepting data for a trace.

        ``TraceLogBuffer`` uses this as a fast guard before constructing a log
        record; ``add_log()`` checks again under its own critical section.
        """

        with self._lock:
            return trace_id in self._traces

    def _find_span(self, trace_id: str, span_id: str) -> TraceSpanRecord | None:
        """Find a span from newest to oldest while the caller holds the lock."""

        trace = self._traces.get(trace_id)
        if trace is None:
            return None
        return next((span for span in reversed(trace.spans) if span.id == span_id), None)

    def _snapshot(self, trace: DiagnosticTrace) -> DiagnosticTraceSnapshot:
        """Deep-copy an exportable live trace while the caller holds the store lock."""

        return DiagnosticTraceSnapshot(
            trace_id=trace.trace_id,
            created_utc=trace.created_utc,
            logs=tuple(TraceLogRecord(**asdict(record)) for record in trace.logs),
            spans=tuple(
                TraceSpanRecord(
                    id=span.id,
                    parent_id=span.parent_id,
                    name=span.name,
                    start_utc=span.start_utc,
                    end_utc=span.end_utc,
                    attributes=dict(span.attributes),
                    events=[
                        TraceSpanEvent(
                            timestamp=event.timestamp,
                            name=event.name,
                            attributes=dict(event.attributes),
                        )
                        for event in span.events
                    ],
                    status=span.status,
                    exception_type=span.exception_type,
                    exception_message=span.exception_message,
                )
                for span in trace.spans
            ),
            error=TraceError(**asdict(trace.error)) if trace.error else None,
            session_export=deepcopy(trace.session_export),
            engine_turn=trace.engine_turn,
            session_id=trace.session_id or (trace.error.session_id if trace.error else None),
        )


@dataclass(frozen=True)
class _StoredReport:
    path: Path
    trace_id: str
    session_id: str | None
    outcome: str
    created_utc: str
    finalized_utc: str | None
    complete: bool
    size_bytes: int
    mtime_ns: int


class DiagnosticReportRepository:
    """Persist error reports and materialize temporary on-demand exports."""

    def __init__(
        self,
        error_report_dir: Path,
        trace_report_dir: Path,
        error_history_size: int,
        logger: Logger | None = None,
    ) -> None:
        if error_history_size < 0:
            raise ValueError("error_history_size must be at least 0")

        self.error_report_dir = error_report_dir
        self.trace_report_dir = trace_report_dir
        self.error_history_size = error_history_size
        self._logger = logger
        self._lock = threading.RLock()

    def write_error(
        self,
        snapshot: DiagnosticTraceSnapshot,
        *,
        complete: bool,
        finalized_utc: str | None = None,
        protected_trace_ids: frozenset[str] = frozenset(),
    ) -> Path:
        """Atomically write an error archive and return its persistent path."""

        if not is_valid_trace_id(snapshot.trace_id):
            raise ValueError("Invalid trace ID")
        if snapshot.error is None:
            raise ValueError("Persistent diagnostic reports require an error")

        with self._lock:
            self._ensure_report_dir(self.error_report_dir)
            target = self.error_report_dir / f"{snapshot.trace_id}.zip"
            temporary_path: Path | None = None

            try:
                with tempfile.NamedTemporaryFile(
                    mode="w+b",
                    prefix=f".{snapshot.trace_id}.",
                    suffix=".tmp",
                    dir=self.error_report_dir,
                    delete=False,
                ) as temporary:
                    temporary_path = Path(temporary.name)
                    self._write_archive(
                        temporary.file,
                        snapshot,
                        complete=complete,
                        finalized_utc=finalized_utc or (_utc_now() if complete else None),
                    )
                    temporary.flush()
                    os.fsync(temporary.fileno())

                try:
                    os.chmod(temporary_path, stat.S_IRUSR | stat.S_IWUSR)
                except OSError:
                    pass
                os.replace(temporary_path, target)
                try:
                    os.chmod(target, stat.S_IRUSR | stat.S_IWUSR)
                except OSError:
                    pass

                if complete:
                    self._prune_locked(
                        protected_trace_ids,
                        retain_incomplete=True,
                    )
                return target
            except BaseException:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
                raise

    def materialize(
        self,
        snapshot: DiagnosticTraceSnapshot,
        finalized_utc: str,
    ) -> Path:
        """Create a temporary complete archive for one retained trace snapshot."""

        if not is_valid_trace_id(snapshot.trace_id):
            raise ValueError("Invalid trace ID")

        with self._lock:
            self._ensure_report_dir(self.trace_report_dir)

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{snapshot.trace_id}.",
                suffix=".zip",
                dir=self.trace_report_dir,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                self._write_archive(
                    temporary.file,
                    snapshot,
                    complete=True,
                    finalized_utc=finalized_utc,
                )
                temporary.flush()
                os.fsync(temporary.fileno())

            try:
                os.chmod(temporary_path, stat.S_IRUSR | stat.S_IWUSR)
            except OSError:
                pass
            return temporary_path
        except BaseException:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

    def list_error_reports(
        self,
        limit: int | None = None,
    ) -> Sequence[DiagnosticReportMetadata]:
        """List finalized persistent error reports, newest first."""

        if limit is not None and limit < 1:
            raise ValueError("limit must be at least 1")

        with self._lock:
            stored = [report for report in self._scan_locked() if report.complete]
            stored.sort(
                key=lambda report: (report.finalized_utc or "", report.trace_id),
                reverse=True,
            )
            if limit is not None:
                stored = stored[:limit]
            return tuple(
                DiagnosticReportMetadata(
                    trace_id=report.trace_id,
                    session_id=report.session_id,
                    outcome=report.outcome,
                    created_utc=report.created_utc,
                    finalized_utc=report.finalized_utc or "",
                    size_bytes=report.size_bytes,
                )
                for report in stored
            )

    def resolve_error(self, trace_id: str) -> Path | None:
        """Resolve a persistent error archive without allowing path escape."""

        if not is_valid_trace_id(trace_id):
            return None

        with self._lock:
            return self._resolve_in_dir(self.error_report_dir, trace_id)

    def prune(
        self,
        protected_trace_ids: frozenset[str] = frozenset(),
        *,
        retain_incomplete: bool = False,
    ) -> None:
        """Keep protected recent-turn errors and the newest configured errors."""

        with self._lock:
            self._prune_locked(
                protected_trace_ids,
                retain_incomplete=retain_incomplete,
            )

    def _write_archive(
        self,
        temporary: IO[bytes],
        snapshot: DiagnosticTraceSnapshot,
        *,
        complete: bool,
        finalized_utc: str | None,
    ) -> None:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            allowZip64=True,
        ) as archive:
            if snapshot.error is not None:
                archive.writestr("traceback.txt", snapshot.error.traceback)
            archive.writestr(
                "logs.jsonl",
                "".join(
                    f"{json.dumps(asdict(record), ensure_ascii=False)}\n"
                    for record in snapshot.logs
                ),
            )
            archive.writestr(
                "trace.json",
                json.dumps(
                    {
                        "trace_id": snapshot.trace_id,
                        "spans": [asdict(span) for span in snapshot.spans],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            if snapshot.session_export is not None:
                archive.writestr(
                    "session.json",
                    json.dumps(snapshot.session_export.session, ensure_ascii=False, indent=2),
                )
                archive.writestr(
                    "events.jsonl",
                    "".join(
                        f"{json.dumps(event, ensure_ascii=False)}\n"
                        for event in snapshot.session_export.events
                    ),
                )

            files = ["logs.jsonl", "trace.json", "manifest.json"]
            if snapshot.error is not None:
                files.insert(0, "traceback.txt")
            if snapshot.session_export is not None:
                files.extend(["session.json", "events.jsonl"])

            archive.writestr(
                "manifest.json",
                json.dumps(
                    {
                        "schema_version": 1,
                        "trace_id": snapshot.trace_id,
                        "session_id": snapshot.session_id
                        or (snapshot.error.session_id if snapshot.error else None),
                        "parlant_version": VERSION,
                        "created_utc": snapshot.created_utc,
                        "captured_utc": (snapshot.error.captured_utc if snapshot.error else None),
                        "finalized_utc": finalized_utc if complete else None,
                        "complete": complete,
                        "outcome": "error" if snapshot.error else "success",
                        "exception": (
                            {
                                "type": snapshot.error.exception_type,
                                "message": snapshot.error.exception_message,
                            }
                            if snapshot.error
                            else None
                        ),
                        "files": files,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )

    def _scan_locked(self) -> list[_StoredReport]:
        reports: list[_StoredReport] = []
        report_dir = self.error_report_dir
        if not report_dir.exists() or report_dir.is_symlink():
            return reports
        for path in report_dir.iterdir():
            if path.is_symlink() or not _REPORT_FILE_PATTERN.fullmatch(path.name):
                continue
            try:
                file_stat = path.stat(follow_symlinks=False)
                if not stat.S_ISREG(file_stat.st_mode):
                    continue
                with zipfile.ZipFile(path) as archive:
                    manifest = json.loads(archive.read("manifest.json"))
                trace_id = path.stem
                outcome = manifest.get("outcome")
                complete = manifest.get("complete")
                finalized_utc = manifest.get("finalized_utc")
                if (
                    manifest.get("trace_id") != trace_id
                    or outcome != "error"
                    or not isinstance(complete, bool)
                    or (complete and not isinstance(finalized_utc, str))
                ):
                    continue
                reports.append(
                    _StoredReport(
                        path=path,
                        trace_id=trace_id,
                        session_id=(
                            str(manifest["session_id"])
                            if manifest.get("session_id") is not None
                            else None
                        ),
                        outcome=outcome,
                        created_utc=str(manifest["created_utc"]),
                        finalized_utc=finalized_utc,
                        complete=complete,
                        size_bytes=file_stat.st_size,
                        mtime_ns=file_stat.st_mtime_ns,
                    )
                )
            except (
                KeyError,
                OSError,
                ValueError,
                zipfile.BadZipFile,
                json.JSONDecodeError,
            ):
                continue
        return reports

    def _prune_locked(
        self,
        protected_trace_ids: frozenset[str],
        *,
        retain_incomplete: bool,
    ) -> None:
        reports = self._scan_locked()
        reports.sort(
            key=lambda report: (report.finalized_utc or "", report.mtime_ns, report.trace_id),
            reverse=True,
        )
        retained_paths = {
            report.path for report in reports if report.trace_id in protected_trace_ids
        }
        if retain_incomplete:
            retained_paths.update(report.path for report in reports if not report.complete)
        retained_paths.update(report.path for report in reports[: self.error_history_size])

        for report in reports:
            if report.path in retained_paths:
                continue
            try:
                report.path.unlink()
            except OSError as exc:
                if self._logger:
                    self._logger.error(f"Failed to prune diagnostic archive {report.path}: {exc}")

    def _resolve_in_dir(self, report_dir: Path, trace_id: str) -> Path | None:
        if not report_dir.exists() or report_dir.is_symlink():
            return None
        try:
            root = report_dir.resolve(strict=True)
            candidate = report_dir / f"{trace_id}.zip"
            if candidate.is_symlink():
                return None
            resolved = candidate.resolve(strict=True)
            if resolved.parent != root or resolved.suffix != ".zip" or not resolved.is_file():
                return None
            return resolved
        except (FileNotFoundError, OSError, RuntimeError):
            return None

    def _ensure_report_dir(self, report_dir: Path) -> None:
        if report_dir.is_symlink():
            raise RuntimeError("Diagnostic report directory must not be a symbolic link")
        report_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        if report_dir.is_symlink():
            raise RuntimeError("Diagnostic report directory must not be a symbolic link")
        try:
            os.chmod(report_dir, stat.S_IRWXU)
        except OSError:
            pass


class DiagnosticReportService:
    """Coordinate live capture, memory retention, persistence, and export."""

    def __init__(
        self,
        store: DiagnosticStore | None = None,
        repository: DiagnosticReportRepository | None = None,
        history: DiagnosticTraceHistory | None = None,
        logger: Logger | None = None,
    ) -> None:
        self._store = store
        self._repository = repository
        self._history = history
        self._logger = logger

    @property
    def enabled(self) -> bool:
        return (
            self._store is not None
            and self._repository is not None
            and (
                (self._history is not None and self._history.capacity > 0)
                or self._repository.error_history_size > 0
            )
        )

    def start(self, trace_id: str) -> None:
        if not self._store:
            return
        try:
            self._store.start(trace_id)
        except Exception as exc:
            self._log_error(trace_id, "start diagnostic capture", exc)

    def retain(self, trace_id: str) -> bool:
        if not self._store:
            return False
        try:
            return self._store.retain(trace_id)
        except Exception as exc:
            self._log_error(trace_id, "retain diagnostic capture", exc)
            return False

    def mark_engine_turn(self, trace_id: str, session_id: str) -> bool:
        if not self._store or not self._history or self._history.capacity == 0:
            return False
        try:
            return self._store.mark_engine_turn(trace_id, session_id)
        except Exception as exc:
            self._log_error(trace_id, "mark engine turn", exc)
            return False

    def release(self, trace_id: str) -> None:
        if not self._store or not self._repository:
            return
        try:
            snapshot = self._store.release(trace_id)
            if snapshot is None:
                return
        except Exception as exc:
            self._log_error(trace_id, "release diagnostic capture", exc)
            return

        finalized_utc = _utc_now()
        evicted: CompletedDiagnosticTrace | None = None
        if self._history and self._history.capacity > 0 and snapshot.engine_turn:
            evicted = self._history.add(snapshot, finalized_utc)

        protected_trace_ids = self._history.trace_ids() if self._history else frozenset()
        if snapshot.error is not None:
            try:
                report_path = self._repository.write_error(
                    snapshot,
                    complete=True,
                    finalized_utc=finalized_utc,
                    protected_trace_ids=protected_trace_ids,
                )
            except Exception as exc:
                self._log_error(trace_id, "finalize diagnostic report", exc)
            else:
                self._log_info(
                    f"Diagnostic report successfully finalized for trace {trace_id}: {report_path}"
                )
        elif evicted is not None and evicted.snapshot.error is not None:
            try:
                self._repository.prune(
                    protected_trace_ids,
                    retain_incomplete=True,
                )
            except Exception as exc:
                self._log_error(trace_id, "prune diagnostic reports", exc)

    def should_attach_session_data(self, trace_id: str) -> bool:
        if not self._store:
            return False
        try:
            return self._store.should_attach_session_data(trace_id)
        except Exception as exc:
            self._log_error(trace_id, "query diagnostic report state", exc)
            return False

    def attach_session_data(
        self,
        trace_id: str,
        session: JSONSerializable,
        events: Sequence[JSONSerializable],
    ) -> None:
        if not self._store:
            return
        try:
            self._store.attach_session_export(
                trace_id,
                SessionExport(session=session, events=tuple(events)),
            )
        except Exception as exc:
            self._log_error(trace_id, "attach session data", exc)

    def capture_exception(
        self,
        trace_id: str,
        session_id: str,
        exception: Exception,
        formatted_traceback: str,
        span_id: str | None,
    ) -> str | None:
        """Persist an immediately downloadable initial error report."""

        if not self._store or not self._repository:
            return None
        if (
            self._history is None or self._history.capacity == 0
        ) and self._repository.error_history_size == 0:
            return None
        try:
            snapshot = self._store.mark_error(
                trace_id=trace_id,
                session_id=session_id,
                exception=exception,
                formatted_traceback=formatted_traceback,
                span_id=span_id,
            )
            if snapshot is None:
                return None
            self._repository.write_error(snapshot, complete=False)
        except Exception as exc:
            self._log_error(trace_id, "create initial diagnostic report", exc)
            return None
        return f"/debug/reports/{trace_id}"

    def prepare_report(self, trace_id: str) -> PreparedDiagnosticReport | None:
        """Resolve an error archive or materialize a retained trace for streaming."""

        if not self._repository or not is_valid_trace_id(trace_id):
            return None
        if error_path := self._repository.resolve_error(trace_id):
            return PreparedDiagnosticReport(error_path, delete_after_stream=False)
        if not self._history or not (entry := self._history.get(trace_id)):
            return None

        try:
            path = self._repository.materialize(
                entry.snapshot,
                finalized_utc=entry.finalized_utc,
            )
        except Exception as exc:
            self._log_error(trace_id, "materialize diagnostic report", exc)
            return None
        return PreparedDiagnosticReport(path, delete_after_stream=True)

    def list_reports(
        self,
        limit: int | None = None,
        outcome: str | None = None,
    ) -> Sequence[DiagnosticReportMetadata]:
        if limit is not None and limit < 1:
            raise ValueError("limit must be at least 1")
        if outcome not in (None, "success", "error"):
            raise ValueError("outcome must be 'success' or 'error'")
        if not self._repository:
            return ()

        reports: list[DiagnosticReportMetadata] = []
        if outcome in (None, "error"):
            reports.extend(self._repository.list_error_reports())
        if outcome in (None, "success") and self._history:
            reports.extend(
                DiagnosticReportMetadata(
                    trace_id=entry.snapshot.trace_id,
                    session_id=entry.snapshot.session_id,
                    outcome="success",
                    created_utc=entry.snapshot.created_utc,
                    finalized_utc=entry.finalized_utc,
                    size_bytes=None,
                )
                for entry in self._history.list()
                if entry.snapshot.error is None
            )

        reports.sort(
            key=lambda report: (report.finalized_utc, report.trace_id),
            reverse=True,
        )
        if limit is not None:
            reports = reports[:limit]
        return tuple(reports)

    def _log_info(self, message: str) -> None:
        try:
            if self._logger:
                self._logger.info(message)
        except Exception:
            pass

    def _log_error(self, trace_id: str, operation: str, exception: Exception) -> None:
        message = (
            f"Diagnostic report service failed to {operation} for trace {trace_id}: "
            f"{''.join(traceback.format_exception(exception))}"
        )
        try:
            if self._logger:
                self._logger.error(message)
        except Exception:
            pass


NOOP_DIAGNOSTIC_REPORT_SERVICE = DiagnosticReportService()


class RecordingTracer(Tracer):
    """Tracer decorator that records a local span timeline and trace lifetime.

    Every tracer operation is delegated to ``inner`` first, preserving the
    behavior of :class:`LocalTracer`, OpenTelemetry, or a custom tracer. In
    parallel, a context-local diagnostic span stack supplies parentage for
    :class:`TraceSpanRecord` objects. Entering an outermost span starts a trace
    owner; exiting it closes the span and releases that owner.

    Server startup installs this object as the container's ``Tracer``, so the
    existing API middleware, session module, engines, and application services
    use it without constructor changes. :class:`TraceLogBuffer` reads its
    current trace/span identity, and engine helpers retrieve its associated
    :class:`DiagnosticReportService`.
    """

    def __init__(self, inner: Tracer, service: DiagnosticReportService) -> None:
        """Wrap an existing tracer and bind it to one diagnostic-report service.

        Server startup constructs this after module/custom tracer
        configuration, ensuring diagnostics supplement rather than replace the
        selected tracing implementation.
        """

        self._inner = inner
        self.diagnostic_report_service = service
        self._store = service._store
        self._span_stack = contextvars.ContextVar[tuple[str, ...]](
            f"diagnostic_report_span_stack_{id(self)}",
            default=(),
        )

    @property
    def current_diagnostic_span_id(self) -> str | None:
        """Return the innermost local span ID for engine error association."""

        stack = self._span_stack.get()
        return stack[-1] if stack else None

    @contextmanager
    @override
    def fresh_trace(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue] = {},
        *,
        links: Sequence[TraceContext] = (),
        trace_id: str | None = None,
    ) -> Generator[None]:
        # The inner fresh root owns the native span; this wrapper still records
        # a separate diagnostic lifetime, never a child of the closed origin.
        reset = self._span_stack.set(())
        try:
            with self._record_span(
                span_id,
                attributes,
                self._inner.fresh_trace(span_id, attributes, links=links, trace_id=trace_id),
            ):
                yield
        finally:
            self._span_stack.reset(reset)

    @property
    @override
    def current_span_context(self) -> TraceContext | None:
        return self._inner.current_span_context

    @contextmanager
    @override
    def span(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> Generator[None]:
        """Delegate a span while mirroring its lifecycle into diagnostics.

        An empty diagnostic stack identifies a root span: the service acquires
        ownership on entry and releases it after the local span is closed.
        Nested calls record parentage without changing ownership.
        """

        with self._record_span(span_id, attributes, self._inner.span(span_id, attributes)):
            yield

    @contextmanager
    def _record_span(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue],
        scope: AbstractContextManager[None],
    ) -> Generator[None]:
        parent_stack = self._span_stack.get()
        diagnostic_span_id = uuid4().hex

        with scope:
            trace_id = self._inner.trace_id
            is_root = not parent_stack

            if is_root:
                self.diagnostic_report_service.start(trace_id)

            if self._store:
                self._store.start_span(
                    trace_id=trace_id,
                    span_id=diagnostic_span_id,
                    parent_id=parent_stack[-1] if parent_stack else None,
                    name=span_id,
                    attributes=attributes,
                )

            reset_token = self._span_stack.set((*parent_stack, diagnostic_span_id))
            caught: BaseException | None = None

            try:
                yield
            except BaseException as exc:
                caught = exc
                raise
            finally:
                if self._store:
                    self._store.end_span(trace_id, diagnostic_span_id, caught)
                self._span_stack.reset(reset_token)
                if is_root:
                    self.diagnostic_report_service.release(trace_id)

    @contextmanager
    @override
    def attributes(
        self,
        attributes: Mapping[str, AttributeValue],
    ) -> Generator[None]:
        """Delegate an attribute context and mirror values to the current span."""

        with self._inner.attributes(attributes):
            if self._store:
                self._store.update_span_attributes(
                    self.trace_id,
                    self.current_diagnostic_span_id,
                    attributes,
                )
            yield

    @property
    @override
    def trace_id(self) -> str:
        """Expose the configured tracer's current root trace ID."""

        return self._inner.trace_id

    @property
    @override
    def span_id(self) -> str:
        """Expose the configured tracer's current span path or identifier."""

        return self._inner.span_id

    @override
    def get_attribute(self, name: str) -> AttributeValue | None:
        """Read an attribute from the configured tracer."""

        return self._inner.get_attribute(name)

    @override
    def set_attribute(self, name: str, value: AttributeValue) -> None:
        """Set an inner-tracer attribute and mirror it to the diagnostic span."""

        self._inner.set_attribute(name, value)
        if self._store:
            self._store.update_span_attributes(
                self.trace_id,
                self.current_diagnostic_span_id,
                {name: value},
            )

    @override
    def add_event(
        self,
        name: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> None:
        """Add an inner-tracer event and mirror it to the diagnostic span."""

        self._inner.add_event(name, attributes)
        if self._store:
            self._store.add_span_event(
                self.trace_id,
                self.current_diagnostic_span_id,
                name,
                attributes,
            )

    @override
    def flush(self) -> None:
        """Flush the configured tracer; local diagnostics need no separate flush."""

        self._inner.flush()


def get_diagnostic_report_service(tracer: Tracer) -> DiagnosticReportService:
    if isinstance(tracer, RecordingTracer):
        return tracer.diagnostic_report_service
    return NOOP_DIAGNOSTIC_REPORT_SERVICE


def capture_engine_exception(
    tracer: Tracer,
    session_id: str,
    exception: Exception,
    formatted_traceback: str | None = None,
) -> str | None:
    service = get_diagnostic_report_service(tracer)
    span_id = tracer.current_diagnostic_span_id if isinstance(tracer, RecordingTracer) else None
    return service.capture_exception(
        trace_id=tracer.trace_id,
        session_id=session_id,
        exception=exception,
        formatted_traceback=formatted_traceback
        or "".join(
            traceback.format_exception(
                type(exception),
                exception,
                exception.__traceback__,
            )
        ),
        span_id=span_id,
    )


class TraceLogBuffer(Logger):
    """Logger sink that buffers every log level for the active root trace.

    Server startup appends this sink to the existing
    :class:`~parlant.core.loggers.CompositeLogger`; normal file, WebSocket, or
    OpenTelemetry logging therefore continues unchanged. Each logger method
    creates a :class:`TraceLogRecord` only while the store contains the current
    trace. ``set_level()`` intentionally does nothing so TRACE and DEBUG calls
    remain available in a diagnostic report even when other sinks filter
    them.

    Logger scopes are held in a ``ContextVar`` for async isolation. The
    diagnostic store consumes the records, and final trace release discards
    the entire in-memory buffer after it has either been snapshotted or deemed
    ineligible for retention.
    """

    def __init__(self, tracer: Tracer, store: DiagnosticStore) -> None:
        """Bind log records to the current tracer and shared diagnostic store."""

        self._tracer = tracer
        self._store = store
        self._scopes = contextvars.ContextVar[str](
            f"diagnostic_report_log_scopes_{id(self)}",
            default="",
        )

    @override
    def set_level(self, log_level: LogLevel) -> None:
        """Ignore output thresholds so reports retain every invoked log level."""

        pass

    @override
    def trace(self, message: str) -> None:
        """Capture a TRACE call for the current live trace."""

        self._record("TRACE", message)

    @override
    def debug(self, message: str) -> None:
        """Capture a DEBUG call for the current live trace."""

        self._record("DEBUG", message)

    @override
    def info(self, message: str) -> None:
        """Capture an INFO call for the current live trace."""

        self._record("INFO", message)

    @override
    def warning(self, message: str) -> None:
        """Capture a WARNING call for the current live trace."""

        self._record("WARNING", message)

    @override
    def error(self, message: str) -> None:
        """Capture an ERROR call for the current live trace."""

        self._record("ERROR", message)

    @override
    def critical(self, message: str) -> None:
        """Capture a CRITICAL call for the current live trace."""

        self._record("CRITICAL", message)

    @contextmanager
    @override
    def scope(self, scope_id: str) -> Generator[None, None, None]:
        """Append a context-local logger scope for records inside the block."""

        current = self._scopes.get()
        reset_token = self._scopes.set(f"{current}[{scope_id}]" if current else f"[{scope_id}]")
        try:
            yield
        finally:
            self._scopes.reset(reset_token)

    def _record(self, level: str, message: str) -> None:
        """Build and append a structured record when the current trace is live."""

        trace_id = self._tracer.trace_id
        if not self._store.contains(trace_id):
            return

        self._store.add_log(
            TraceLogRecord(
                timestamp=_utc_now(),
                level=level,
                trace_id=trace_id,
                span_id=self._tracer.span_id,
                scope=self._scopes.get(),
                message=message,
            )
        )


def format_exception(exception: Exception) -> str:
    return "".join(
        traceback.format_exception(
            type(exception),
            exception,
            exception.__traceback__,
        )
    )
