from abc import ABC
from contextlib import contextmanager
from pathlib import Path
import time
import traceback
from typing import Any, Iterator
import coloredlogs  # type: ignore
import contextvars
import logging
import logging.handlers

from emcie.server.core.common import generate_id


class CustomFormatter(logging.Formatter, ABC):
    def __init__(self) -> None:
        super().__init__(
            "%(asctime)s %(name)s[P=%(process)d; T=%(thread)d] %(levelname)s %(message)s"
        )


class Logger(ABC):
    def __init__(self) -> None:
        self.logger = logging.getLogger("emcie")
        self.logger.setLevel(logging.DEBUG)
        self.formatter = CustomFormatter()
        self.scopes = contextvars.ContextVar[list[str]]("scopes", default=[])

    def debug(self, message: str) -> None:
        self.logger.debug(self._add_scopes(message))

    def info(self, message: str) -> None:
        self.logger.info(self._add_scopes(message))

    def warning(self, message: str) -> None:
        self.logger.warning(self._add_scopes(message))

    def error(self, message: str) -> None:
        self.logger.error(self._add_scopes(message))

    def critical(self, message: str) -> None:
        self.logger.critical(self._add_scopes(message))

    @contextmanager
    def scope(self) -> Iterator[None]:
        self.scopes.get().append(generate_id())
        yield
        self.scopes.get().pop()

    @contextmanager
    def operation(self, name: str, props: dict[str, Any] = {}) -> Iterator[None]:
        try:
            t_start = time.time()
            yield
            t_end = time.time()
            if props:
                self.info(f"OPERATION {name} [{props}] finished in {t_end - t_start}s")
            else:
                self.info(f"OPERATION {name} finished in {round(t_end - t_start, 3)} seconds")
        except Exception as exc:
            self.error(f"OPERATION {name} failed")
            self.error(" ".join(traceback.format_exception(exc)))
            raise
        except BaseException as exc:
            self.error(f"OPERATION {name} failed with critical error")
            self.critical(" ".join(traceback.format_exception(exc)))
            raise

    def _add_scopes(self, message: str) -> str:
        if scopes := self.scopes.get():
            chained_scopes = ".".join(scopes)
            return f"[{chained_scopes}] {message}"
        else:
            return message


class StdoutLogger(Logger):
    def __init__(self) -> None:
        super().__init__()
        coloredlogs.install(level="DEBUG", logger=self.logger)


class FileLogger(Logger):
    def __init__(self, log_file_path: Path) -> None:
        super().__init__()

        handlers: list[logging.Handler] = [
            logging.FileHandler(log_file_path),
            logging.StreamHandler(),
        ]

        for handler in handlers:
            handler.setFormatter(self.formatter)
            self.logger.addHandler(handler)

        coloredlogs.install(level="DEBUG", logger=self.logger)
