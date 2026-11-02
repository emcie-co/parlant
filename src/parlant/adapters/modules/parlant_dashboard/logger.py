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

"""ParlantDashboardLogger — OTLP-backed logger for Parlant Dashboard."""

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

from parlant.adapters.modules.parlant_dashboard.config import _get_dashboard_otel_url


class ParlantDashboardLogger(TracingLogger):
    """A logger that sends logs to Parlant Dashboard backend using OpenTelemetry OTLP."""

    def __init__(
        self,
        tracer: Tracer,
        project_id: str = "",
        log_level: LogLevel = LogLevel.DEBUG,
        logger_id: str | None = None,
    ) -> None:
        # A dedicated channel, never the console logger's "parlant": this
        # logger sits in the same CompositeLogger as the console FileLogger,
        # and sharing the underlying stdlib logger would print every record
        # twice — once console-rendered, once as a bare duplicate when our
        # structured record hits the console handler. propagate=False keeps
        # pre-__aenter__ records off stdlib's last-resort stderr handler too.
        super().__init__(
            tracer=tracer,
            log_level=LogLevel.TRACE,
            logger_id=logger_id or "parlant-dashboard",
        )
        self.raw_logger.propagate = False

        self._project_id = project_id
        self._endpoint = f"{_get_dashboard_otel_url()}/v1/logs"
        self._project_token = os.getenv("PARLANT_DASHBOARD_PROJECT_TOKEN", "")

        self._logger_provider: LoggerProvider | None = None
        self._log_exporter: OTLPLogExporter | None = None
        self._log_processor: BatchLogRecordProcessor | None = None
        self._logging_handler: LoggingHandler | None = None

    async def __aenter__(self) -> Self:
        resource_attributes: dict[str, str] = {
            "service.name": "parlant-dashboard-logger",
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
                # Log through our own local handlers: the OTLP handler is what
                # failed, and it is removed in the finally below.
                self.warning(f"Error during ParlantDashboardLogger shutdown: {e}")

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
