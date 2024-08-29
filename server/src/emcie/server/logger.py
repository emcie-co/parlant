from abc import ABC
import logging.handlers
from pathlib import Path
import coloredlogs  # type: ignore
import logging


class Logger(ABC):
    def __init__(self) -> None:
        self.logger = logging.getLogger("emcie")
        self.logger.setLevel(logging.DEBUG)

    def debug(self, message: str) -> None:
        self.logger.debug(message)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def critical(self, message: str) -> None:
        self.logger.critical(message)


class StdoutLogger(Logger):
    def __init__(self) -> None:
        super().__init__()
        coloredlogs.install(level="DEBUG", logger=self.logger)


class FileLogger(Logger):
    def __init__(self, log_file_path: Path) -> None:
        super().__init__()
        formatter = logging.Formatter(
            "%(asctime)s %(name)s[P=%(process)d; T=%(thread)d] %(levelname)s  %(message)s"
        )

        handlers: list[logging.Handler] = [
            logging.FileHandler(log_file_path),
            logging.StreamHandler(),
        ]

        for handler in handlers:
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
