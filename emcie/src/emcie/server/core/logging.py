from abc import ABC, abstractmethod
import asyncio
from contextlib import contextmanager
from pathlib import Path
import time
import traceback
from typing import Any, Iterator
import coloredlogs  # type: ignore
import logging
import logging.handlers

from emcie.server.core.contextual_correlator import ContextualCorrelator


class CustomFormatter(logging.Formatter, ABC):
    def __init__(self) -> None:
        super().__init__(
            "%(asctime)s %(name)s[P=%(process)d; T=%(thread)d] %(levelname)s %(message)s"
        )


class Logger(ABC):
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
    def __init__(self, correlator: ContextualCorrelator) -> None:
        self._correlator = correlator
        self.logger = logging.getLogger("emcie")
        self.logger.setLevel(logging.DEBUG)
        self._formatter = CustomFormatter()

    def debug(self, message: str) -> None:
        self.logger.debug(self._add_correlation_id(message))

    def info(self, message: str) -> None:
        self.logger.info(self._add_correlation_id(message))

    def warning(self, message: str) -> None:
        self.logger.warning(self._add_correlation_id(message))

    def error(self, message: str) -> None:
        self.logger.error(self._add_correlation_id(message))

    def critical(self, message: str) -> None:
        self.logger.critical(self._add_correlation_id(message))

    @contextmanager
    def operation(self, name: str, props: dict[str, Any] = {}) -> Iterator[None]:
        try:
            t_start = time.time()
            self.info(f"OPERATION {name} [{props}] started")
            yield
            t_end = time.time()
            if props:
                self.info(f"OPERATION {name} [{props}] finished in {t_end - t_start}s")
            else:
                self.info(f"OPERATION {name} finished in {round(t_end - t_start, 3)} seconds")
        except asyncio.CancelledError:
            self.error(f"OPERATION {name} cancelled")
            raise
        except Exception as exc:
            self.error(f"OPERATION {name} failed")
            self.error(" ".join(traceback.format_exception(exc)))
            raise
        except BaseException as exc:
            self.error(f"OPERATION {name} failed with critical error")
            self.critical(" ".join(traceback.format_exception(exc)))
            raise

    def _add_correlation_id(self, message: str) -> str:
        return f"[{self._correlator.correlation_id}] {message}"


class StdoutLogger(CorrelationalLogger):
    def __init__(self, correlator: ContextualCorrelator) -> None:
        super().__init__(correlator)
        coloredlogs.install(level="DEBUG", logger=self.logger)


class FileLogger(CorrelationalLogger):
    def __init__(self, log_file_path: Path, correlator: ContextualCorrelator) -> None:
        super().__init__(correlator)

        handlers: list[logging.Handler] = [
            logging.FileHandler(log_file_path),
            logging.StreamHandler(),
        ]

        for handler in handlers:
            handler.setFormatter(self._formatter)
            self.logger.addHandler(handler)

        coloredlogs.install(level="DEBUG", logger=self.logger)
