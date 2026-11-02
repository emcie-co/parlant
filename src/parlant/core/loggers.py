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
from abc import ABC, abstractmethod
from contextlib import ExitStack, contextmanager
from collections.abc import Generator
import contextvars
from enum import Enum, auto
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import structlog
from typing import Sequence
from typing_extensions import override

from parlant.core.common import generate_id
from parlant.core.tracer import Tracer


class LogLevel(Enum):
    """Enumeration of log levels with comparison and conversion methods."""

    TRACE = auto()
    """Trace level for detailed debugging information."""

    DEBUG = auto()
    """Debug level for general debugging information."""

    INFO = auto()
    """Info level for general informational messages."""

    WARNING = auto()
    """Warning level for potential issues that do not require immediate attention."""

    ERROR = auto()
    """Error level for errors that do not stop the program."""

    CRITICAL = auto()
    """Critical level for severe errors that may cause the program to stop."""

    def __lt__(self, other: LogLevel) -> bool:
        return self.to_int() < other.to_int()

    def __le__(self, other: LogLevel) -> bool:
        return self.to_int() <= other.to_int()

    def __gt__(self, other: LogLevel) -> bool:
        return self.to_int() > other.to_int()

    def __ge__(self, other: LogLevel) -> bool:
        return self.to_int() >= other.to_int()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LogLevel):
            return NotImplemented
        return self.to_int() == other.to_int()

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, LogLevel):
            return NotImplemented
        return self.to_int() != other.to_int()

    def __hash__(self) -> int:
        return super().__hash__()

    def to_logging_level(self) -> int:
        """Convert the log level to a logging module level."""

        return {
            LogLevel.TRACE: logging.DEBUG,
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }[self]

    def to_int(self) -> int:
        """Convert the log level to an integer for comparison."""

        return {
            LogLevel.TRACE: 0,
            LogLevel.DEBUG: 1,
            LogLevel.INFO: 2,
            LogLevel.WARNING: 3,
            LogLevel.ERROR: 4,
            LogLevel.CRITICAL: 5,
        }[self]


class Logger(ABC):
    """An abstract base class for logging operations.

    Level conventions (enforced by review, not tooling):

    - TRACE: bulky payloads — full prompts, turn instructions, streaming
      internals. Only emitted when the level is exactly TRACE.
    - DEBUG: structured phase results — discovery/evaluation tables, usage
      blocks, resolution dumps.
    - INFO: sparse operational lifecycle notes.
    - WARNING, two families:
      * Degradations that fail open — suffix the message with the behavior
        taken, e.g. "... failed (continuing)", "... (failing open)",
        "... (keeping the rule)". A warning without such a suffix implies no
        degradation occurred.
      * Behavioral anomalies — unexpected but handled conditions (unknown
        tool requested, transient retry, policy denial).
    - ERROR: an operation failed and its result is lost or replaced (a turn
      aborted, a tool call failed, accounting could not be finalized).
    - CRITICAL: reserved; currently unused.

    Scope conventions: components identify themselves via logging scopes, not
    message-text prefixes — flow components push ``with logger.scope("X")`` at
    their entry points; service/adapter components bind ``ScopedLogger(logger,
    "X")`` once at construction. Scope ids are bare component names (class
    names), with no engine or product prefix.
    """

    @abstractmethod
    def set_level(self, log_level: LogLevel) -> None:
        """Set the logging level for the logger."""
        ...

    @abstractmethod
    def trace(self, message: str) -> None:
        """Log a message at the TRACE level."""
        ...

    @abstractmethod
    def debug(self, message: str) -> None:
        """Log a message at the DEBUG level."""
        ...

    @abstractmethod
    def info(self, message: str) -> None:
        """Log a message at the INFO level."""
        ...

    @abstractmethod
    def warning(self, message: str) -> None:
        """Log a message at the WARNING level."""
        ...

    @abstractmethod
    def error(self, message: str) -> None:
        """Log a message at the ERROR level."""
        ...

    @abstractmethod
    def critical(self, message: str) -> None:
        """Log a message at the CRITICAL level."""
        ...

    @abstractmethod
    @contextmanager
    def scope(self, scope_id: str) -> Generator[None, None, None]:
        """Create a new logging scope."""
        ...


class TracingLogger(Logger):
    """A logger that supports trace IDs for structured logging."""

    def __init__(
        self,
        tracer: Tracer,
        log_level: LogLevel = LogLevel.DEBUG,
        logger_id: str | None = None,
    ) -> None:
        self._tracer = tracer
        self.raw_logger = logging.getLogger(logger_id or "parlant")
        self.raw_logger.setLevel(log_level.to_logging_level())
        self.log_level = log_level

        # Wrap it with structlog configuration
        self._logger = structlog.wrap_logger(
            self.raw_logger,
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.stdlib.add_log_level,
                structlog.stdlib.filter_by_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.dev.ConsoleRenderer(colors=True),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(0),
        )

        # Scope support using contextvars
        self._instance_id = generate_id()

        self._scopes = contextvars.ContextVar[str](
            f"logger_{self._instance_id}_scopes",
            default="",
        )

    @override
    def set_level(self, log_level: LogLevel) -> None:
        self.raw_logger.setLevel(log_level.to_logging_level())
        self.log_level = log_level

    @override
    def trace(self, message: str) -> None:
        # Unlike the other levels (filtered by the underlying stdlib logger),
        # TRACE self-filters: stdlib logging has no TRACE severity, so trace
        # lines ride DEBUG severity with a textual marker and would otherwise
        # leak at DEBUG level.
        if self.log_level > LogLevel.TRACE:
            return

        self._logger.debug(
            f"TRACE {self._add_trace_id_and_scopes(message)}",
        )

    @override
    def debug(self, message: str) -> None:
        self._logger.debug(self._add_trace_id_and_scopes(message))

    @override
    def info(self, message: str) -> None:
        self._logger.info(self._add_trace_id_and_scopes(message))

    @override
    def warning(self, message: str) -> None:
        self._logger.warning(self._add_trace_id_and_scopes(message))

    @override
    def error(self, message: str) -> None:
        self._logger.error(self._add_trace_id_and_scopes(message))

    @override
    def critical(self, message: str) -> None:
        self._logger.critical(self._add_trace_id_and_scopes(message))

    @override
    @contextmanager
    def scope(self, scope_id: str) -> Generator[None, None, None]:
        current_scopes = self._scopes.get()

        if current_scopes:
            new_scopes = current_scopes + f"[{scope_id}]"
        else:
            new_scopes = f"[{scope_id}]"

        reset_token = self._scopes.set(new_scopes)

        yield

        self._scopes.reset(reset_token)

    @property
    def current_scope(self) -> str:
        return self._get_scopes()

    def _add_trace_id_and_scopes(self, message: str) -> str:
        # Render only the trace id's tail: W3C/OTel guarantee randomness in the
        # rightmost bits (the left half may be zero-padded), so the last chars
        # are the safe abbreviation. The full id stays on the tracer and in
        # structured sinks; `grep <last-10-of-full-id>` finds these lines.
        return f"[{self._tracer.trace_id[-10:]}]{self.current_scope} {message}"

    def _get_scopes(self) -> str:
        if scopes := self._scopes.get():
            return scopes
        return ""


class StdoutLogger(TracingLogger):
    """A logger that outputs to standard output."""

    def __init__(
        self,
        tracer: Tracer,
        log_level: LogLevel = LogLevel.DEBUG,
        logger_id: str | None = None,
    ) -> None:
        super().__init__(tracer, log_level, logger_id)
        self.raw_logger.addHandler(logging.StreamHandler())


class _OwnerOnlyRotatingFileHandler(RotatingFileHandler):
    """Rotating handler that reapplies owner-only mode to each new active file."""

    @override
    def doRollover(self) -> None:
        super().doRollover()
        try:
            os.chmod(self.baseFilename, 0o600)
        except OSError:
            pass


class FileLogger(TracingLogger):
    """A logger that outputs to a file."""

    def __init__(
        self,
        log_file_path: Path,
        tracer: Tracer,
        log_level: LogLevel = LogLevel.DEBUG,
        logger_id: str | None = None,
        max_bytes: int = 0,
        backup_count: int = 0,
    ) -> None:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        if backup_count < 0:
            raise ValueError("backup_count must be non-negative")
        if max_bytes > 0 and backup_count < 1:
            raise ValueError("backup_count must be at least 1 when rotation is enabled")

        super().__init__(tracer, log_level, logger_id)

        directory_existed = log_file_path.parent.exists()
        log_file_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not directory_existed:
            try:
                os.chmod(log_file_path.parent, 0o700)
            except OSError:
                pass

        file_handler = _OwnerOnlyRotatingFileHandler(
            log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        try:
            os.chmod(log_file_path, 0o600)
        except OSError:
            pass

        handlers: list[logging.Handler] = [file_handler, logging.StreamHandler()]

        for handler in handlers:
            self.raw_logger.addHandler(handler)


class ScopedLogger(Logger):
    """A view of a logger that emits every call inside a fixed scope.

    Components bind their telemetry-indexing scope once at construction
    (``self._logger = ScopedLogger(logger, "QdrantVectorStore")``) instead of
    wrapping each call site in ``with logger.scope(...)``. Scopes entered on
    the view nest under the fixed scope; the underlying logger is untouched
    outside the view's own calls.
    """

    def __init__(self, logger: Logger, scope_id: str) -> None:
        self._logger = logger
        self._scope_id = scope_id
        self._entered: contextvars.ContextVar[bool] = contextvars.ContextVar(
            f"scoped_logger_{scope_id}", default=False
        )

    @contextmanager
    def _fixed_scope(self) -> Generator[None, None, None]:
        # Idempotent: a call made inside the view's own scope() block must not
        # push the fixed scope a second time (or out of order).
        if self._entered.get():
            yield
            return
        reset_token = self._entered.set(True)
        try:
            with self._logger.scope(self._scope_id):
                yield
        finally:
            self._entered.reset(reset_token)

    @override
    def set_level(self, log_level: LogLevel) -> None:
        self._logger.set_level(log_level)

    @override
    def trace(self, message: str) -> None:
        with self._fixed_scope():
            self._logger.trace(message)

    @override
    def debug(self, message: str) -> None:
        with self._fixed_scope():
            self._logger.debug(message)

    @override
    def info(self, message: str) -> None:
        with self._fixed_scope():
            self._logger.info(message)

    @override
    def warning(self, message: str) -> None:
        with self._fixed_scope():
            self._logger.warning(message)

    @override
    def error(self, message: str) -> None:
        with self._fixed_scope():
            self._logger.error(message)

    @override
    def critical(self, message: str) -> None:
        with self._fixed_scope():
            self._logger.critical(message)

    @override
    @contextmanager
    def scope(self, scope_id: str) -> Generator[None, None, None]:
        with self._fixed_scope(), self._logger.scope(scope_id):
            yield


class CompositeLogger(Logger):
    """A logger that combines multiple loggers into one."""

    def __init__(self, loggers: Sequence[Logger]) -> None:
        self._loggers = list(loggers)

    def append(self, logger: Logger) -> None:
        self._loggers.append(logger)

    def excluding(self, member: Logger) -> CompositeLogger:
        """A view of this composite without one member.

        Components that sit INSIDE the composite (e.g. the diagnostic trace
        buffer) use this for their own diagnostics, so their failure messages
        don't feed back into the machinery being diagnosed.
        """

        return CompositeLogger([logger for logger in self._loggers if logger is not member])

    @override
    def set_level(self, log_level: LogLevel) -> None:
        for logger in self._loggers:
            logger.set_level(log_level)

    @override
    def trace(self, message: str) -> None:
        for logger in self._loggers:
            logger.trace(message)

    @override
    def debug(self, message: str) -> None:
        for logger in self._loggers:
            logger.debug(message)

    @override
    def info(self, message: str) -> None:
        for logger in self._loggers:
            logger.info(message)

    @override
    def warning(self, message: str) -> None:
        for logger in self._loggers:
            logger.warning(message)

    @override
    def error(self, message: str) -> None:
        for logger in self._loggers:
            logger.error(message)

    @override
    def critical(self, message: str) -> None:
        for logger in self._loggers:
            logger.critical(message)

    @override
    @contextmanager
    def scope(self, scope_id: str) -> Generator[None, None, None]:
        with ExitStack() as stack:
            for context in [logger.scope(scope_id) for logger in self._loggers]:
                stack.enter_context(context)
            yield
