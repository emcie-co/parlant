# Copyright 2026 Parlant (Emcie Co Ltd.)
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

"""ParlantCloudLogger — OTLP-backed logger for Parlant Cloud."""

import logging
import os
from types import TracebackType
from typing import Any, MutableMapping

import structlog
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from typing_extensions import Self, override

from parlant.core.loggers import LogLevel, TracingLogger
from parlant.core.tracer import Tracer

from .config import _get_cloud_otel_url

_logger = logging.getLogger(__name__)


class ParlantCloudLogger(TracingLogger):
    """A logger that sends logs to Parlant Cloud backend using OpenTelemetry OTLP."""

    def __init__(
        self,
        tracer: Tracer,
        project_id: str = "",
        log_level: LogLevel = LogLevel.DEBUG,
        logger_id: str | None = None,
    ) -> None:
        super().__init__(tracer=tracer, log_level=LogLevel.TRACE, logger_id=logger_id)

        self._project_id = project_id
        self._endpoint = f"{_get_cloud_otel_url()}/v1/logs"
        self._project_token = os.getenv("PARLANT_CLOUD_PROJECT_TOKEN", "")

        self._logger_provider: LoggerProvider | None = None
        self._log_exporter: OTLPLogExporter | None = None
        self._log_processor: BatchLogRecordProcessor | None = None
        self._logging_handler: LoggingHandler | None = None

    async def __aenter__(self) -> Self:
        resource_attributes: dict[str, str] = {
            "service.name": "parlant-cloud-logger",
        }
        if self._project_id:
            resource_attributes["project_id"] = self._project_id

        resource = Resource.create(resource_attributes)

        headers = {}
        if self._project_token:
            headers["authorization"] = f"Bearer {self._project_token}"

        self._log_exporter = OTLPLogExporter(
            endpoint=self._endpoint,
            headers=headers,
        )

        self._logger_provider = LoggerProvider(resource=resource)
        self._log_processor = BatchLogRecordProcessor(
            exporter=self._log_exporter,
            schedule_delay_millis=1000,
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
        if self._log_processor:
            try:
                self._log_processor.force_flush()
                self._log_processor.shutdown()  # type: ignore[no-untyped-call]
            except Exception as e:
                _logger.warning(f"Error during ParlantCloudLogger shutdown: {e}")

        if self._logging_handler:
            self.raw_logger.removeHandler(self._logging_handler)

        return False

    @override
    def set_level(self, log_level: LogLevel) -> None:
        super().set_level(LogLevel.TRACE)
        if self._logging_handler is not None:
            self._logging_handler.setLevel(LogLevel.TRACE.to_logging_level())

    def _inject_structlog_processors(self) -> None:
        def _add_attributes(
            _: Any,
            method: str,
            event_dict: MutableMapping[str, Any],
        ) -> MutableMapping[str, Any]:
            level = event_dict.get("actual_level", event_dict.get("level", method))
            event_dict.pop("actual_level", None)
            event_dict.pop("level", None)

            event_dict["severity_text"] = str(level).upper()
            event_dict["trace_id"] = self._tracer.trace_id
            event_dict["span_id"] = self._tracer.span_id

            if self._project_id:
                event_dict["project_id"] = self._project_id

            if scope := self.current_scope:
                event_dict["scope"] = scope

            return event_dict

        self._logger = structlog.wrap_logger(
            self.raw_logger,
            processors=[
                structlog.stdlib.add_log_level,
                _add_attributes,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.stdlib.render_to_log_kwargs,
            ],
            wrapper_class=structlog.make_filtering_bound_logger(0),
        )

    @override
    def trace(self, message: str) -> None:
        self._logger.debug(message, actual_level="trace")

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
