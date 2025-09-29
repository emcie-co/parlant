import os
from typing import Any, MutableMapping
import structlog
from types import TracebackType
from typing_extensions import Self, override

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter

from parlant.core.loggers import LogLevel, TracingLogger
from parlant.core.tracer import Tracer


class OtelLogger(TracingLogger):
    """TracingLogger with OpenTelemetry log export via OTLP gRPC."""

    @staticmethod
    def is_environment_set() -> bool:
        return bool(os.environ.get("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"))

    def __init__(
        self,
        tracer: Tracer,
        log_level: LogLevel = LogLevel.DEBUG,
        logger_id: str | None = None,
    ) -> None:
        super().__init__(tracer=tracer, log_level=log_level, logger_id=logger_id)

        self._service_name = os.getenv("OTEL_SERVICE_NAME", "parlant")

        self._logger_provider: LoggerProvider
        self._log_exporter: OTLPLogExporter
        self._log_processor: BatchLogRecordProcessor
        self._logging_handler: LoggingHandler

    async def __aenter__(self) -> Self:
        resource = Resource.create({"service.name": self._service_name})

        endpoint = os.environ["OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"]
        insecure = os.getenv("OTEL_EXPORTER_OTLP_INSECURE", "false").lower() == "true"

        self._log_exporter = OTLPLogExporter(endpoint=endpoint, insecure=insecure)
        self._logger_provider = LoggerProvider(resource=resource)
        self._log_processor = BatchLogRecordProcessor(
            exporter=self._log_exporter,
            schedule_delay_millis=2000,
        )
        self._logger_provider.add_log_record_processor(self._log_processor)

        self._logging_handler = LoggingHandler(
            level=self.log_level.to_logging_level(),
            logger_provider=self._logger_provider,
        )

        self.raw_logger.addHandler(self._logging_handler)

        self._inject_structlog_processors()

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self._logger_provider.shutdown()  # type: ignore
        self.raw_logger.removeHandler(self._logging_handler)

        return False

    @override
    def set_level(self, log_level: LogLevel) -> None:
        super().set_level(log_level)
        if self._logging_handler is not None:
            self._logging_handler.setLevel(log_level.to_logging_level())

    def _inject_structlog_processors(self) -> None:
        """Add trace_id/scopes as structured fields (OTEL attributes)."""

        def _add_attributes(
            _: Any,  # logger
            __: str,  # method name (e.g., "info", "error")
            event_dict: MutableMapping[str, Any],
        ) -> MutableMapping[str, Any]:
            attributes = event_dict.get("attributes", {})

            attributes["trace_id"] = str(self._tracer.trace_id)

            scopes = self.current_scope
            if scopes:
                attributes["scopes"] = scopes
            event_dict["attributes"] = attributes

            return event_dict

        self._logger = structlog.wrap_logger(
            self.raw_logger,
            processors=[
                _add_attributes,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.stdlib.render_to_log_kwargs,
            ],
            wrapper_class=structlog.make_filtering_bound_logger(
                0
            ),  # Avoids doing the level check twice.
        )

    @override
    def trace(self, message: str) -> None:
        if self.log_level != LogLevel.TRACE:
            return

        self._logger.debug(message)

    @override
    def debug(self, message: str) -> None:
        self._logger.debug(message)

    @override
    def info(self, message: str) -> None:
        self._logger.info(message)

    @override
    def warning(self, message: str) -> None:
        self._logger.warning(message)

    @override
    def error(self, message: str) -> None:
        self._logger.error(message)

    @override
    def critical(self, message: str) -> None:
        self._logger.critical(message)
