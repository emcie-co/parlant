# Copyright 2024 Emcie Co Ltd.
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

from abc import ABC, abstractmethod
import asyncio
from contextlib import ExitStack, contextmanager
from enum import Enum, auto
import logging
from pathlib import Path
import structlog
import time
import traceback
from typing import Any, Iterator, Sequence
from typing_extensions import override

from parlant.core.contextual_correlator import ContextualCorrelator


class LogLevel(Enum):
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

    def to_logging_level(self) -> int:
        return {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }[self]


class Logger(ABC):
    @abstractmethod
    def set_level(self, log_level: LogLevel) -> None: ...

    @abstractmethod
    def debug(self, message: str) -> None: ...

    @abstractmethod
    def info(self, message: str) -> None: ...

    @abstractmethod
    def warning(self, message: str) -> None: ...

    @abstractmethod
    def error(self, message: str) -> None: ...

    @abstractmethod
    def critical(self, message: str) -> None: ...

    @abstractmethod
    @contextmanager
    def operation(self, name: str, props: dict[str, Any] = {}) -> Iterator[None]: ...


class CorrelationalLogger(Logger):
    def __init__(
        self,
        correlator: ContextualCorrelator,
        log_level: LogLevel = LogLevel.DEBUG,
        logger_id: str | None = None,
    ) -> None:
        self._correlator = correlator
        self.raw_logger = logging.getLogger(logger_id or "parlant")
        self.raw_logger.setLevel(log_level.to_logging_level())

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
            wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        )

    @override
    def set_level(self, log_level: LogLevel) -> None:
        self.raw_logger.setLevel(log_level.to_logging_level())

    @override
    def debug(self, message: str) -> None:
        self._logger.debug(self._add_correlation_id(message))

    @override
    def info(self, message: str) -> None:
        self._logger.info(self._add_correlation_id(message))

    @override
    def warning(self, message: str) -> None:
        self._logger.warning(self._add_correlation_id(message))

    @override
    def error(self, message: str) -> None:
        self._logger.error(self._add_correlation_id(message))

    @override
    def critical(self, message: str) -> None:
        self._logger.critical(self._add_correlation_id(message))

    @override
    @contextmanager
    def operation(self, name: str, props: dict[str, Any] = {}) -> Iterator[None]:
        t_start = time.time()
        try:
            if props:
                self.info(f"{name} [{props}] started")
            else:
                self.info(f"{name} started")

            yield

            t_end = time.time()

            if props:
                self.info(f"{name} [{props}] finished in {t_end - t_start}s")
            else:
                self.info(f"{name} finished in {round(t_end - t_start, 3)} seconds")
        except asyncio.CancelledError:
            self.warning(f"{name} cancelled after {round(time.time() - t_start, 3)} seconds")
            raise
        except Exception as exc:
            self.error(f"{name} failed")
            self.error(" ".join(traceback.format_exception(exc)))
            raise
        except BaseException as exc:
            self.error(f"{name} failed with critical error")
            self.critical(" ".join(traceback.format_exception(exc)))
            raise

    def _add_correlation_id(self, message: str) -> str:
        return f"[{self._correlator.correlation_id}] {message}"


class StdoutLogger(CorrelationalLogger):
    def __init__(
        self,
        correlator: ContextualCorrelator,
        log_level: LogLevel = LogLevel.DEBUG,
        logger_id: str | None = None,
    ) -> None:
        super().__init__(correlator, log_level, logger_id)


class FileLogger(CorrelationalLogger):
    def __init__(
        self,
        log_file_path: Path,
        correlator: ContextualCorrelator,
        log_level: LogLevel = LogLevel.DEBUG,
        logger_id: str | None = None,
    ) -> None:
        super().__init__(correlator, log_level, logger_id)

        handlers: list[logging.Handler] = [
            logging.FileHandler(log_file_path),
            logging.StreamHandler(),
        ]

        for handler in handlers:
            self.raw_logger.addHandler(handler)


class CompositeLogger(Logger):
    def __init__(self, loggers: Sequence[Logger]) -> None:
        self._loggers = list(loggers)

    def append(self, logger: Logger) -> None:
        self._loggers.append(logger)

    @override
    def set_level(self, log_level: LogLevel) -> None:
        for logger in self._loggers:
            logger.set_level(log_level)

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
    def operation(self, name: str, props: dict[str, Any] = {}) -> Iterator[None]:
        with ExitStack() as stack:
            for context in [logger.operation(name, props) for logger in self._loggers]:
                stack.enter_context(context)
            yield
