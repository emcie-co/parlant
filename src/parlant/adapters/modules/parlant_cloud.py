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

"""Parlant Cloud module.

Auto-loaded by the Server when PARLANT_CLOUD_PROJECT_TOKEN is set.
Validates the project token, resolves project context, and sets up
ParlantCloudTracer / ParlantCloudLogger / ParlantCloudMeter.

Tunnel and telemetry derive their base URL from:
  PARLANT_CLOUD_BASE_URL  (preferred)
  PARLANT_CLOUD_OTEL_URL  (backward compat, same meaning)
  default: https://api.parlant.cloud

PARLANT_CLOUD_API_KEY and PARLANT_CLOUD_API_URL are used only by the NLP
service adapter.
"""

import asyncio
import contextvars
import hmac
import logging
import os
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager
from types import TracebackType
from typing import Any, AsyncGenerator, Iterator, Mapping, MutableMapping

import httpx
import structlog
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from lagom import Container
from opentelemetry import context, trace
from opentelemetry.exporter.otlp.proto.http._log_exporter import (
    OTLPLogExporter,
)
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HTTPOTLPSpanExporter,
)
from opentelemetry.metrics import Counter as OTELCounter
from opentelemetry.metrics import Histogram as OTELHistogram
from opentelemetry.metrics import Meter as OTELMeter
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Span, set_tracer_provider
from typing_extensions import Self, override

import json as _json_mod
import websockets
from parlant.api.authorization import AuthorizationPolicy, Operation, ProductionAuthorizationPolicy
from parlant.core.app_modules.agents import AgentModule
from parlant.core.app_modules.customers import CustomerModule
from parlant.core.app_modules.sessions import SessionModule
from parlant.core.app_modules.tags import TagModule
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.loggers import CompositeLogger, LogLevel, Logger, TracingLogger
from parlant.core.meter import Counter, DurationHistogram, Histogram, Meter
from parlant.core.tracer import AttributeValue, CompositeTracer, Tracer
from parlant.core.tunnels import (
    TunnelRequest,
    TunnelRequestDispatcher,
    TunnelResponse,
    TunnelService,
)

_logger = logging.getLogger(__name__)

_exit_stack = AsyncExitStack()

PROJECT_TOKEN_HEADER = "X-Parlant-Cloud-Project-Token"

_DEFAULT_BASE_URL = "https://api.parlant.cloud"


def _get_cloud_base_url() -> str:
    """Resolve the Parlant Cloud base URL from environment.

    Priority: PARLANT_CLOUD_BASE_URL > PARLANT_CLOUD_OTEL_URL > default.
    """
    return (
        os.getenv("PARLANT_CLOUD_BASE_URL")
        or os.getenv("PARLANT_CLOUD_OTEL_URL")
        or _DEFAULT_BASE_URL
    ).rstrip("/")


class ParlantCloudAuthorizationPolicy(AuthorizationPolicy):
    def __init__(self, project_token: str) -> None:
        self._project_token = project_token
        self._production_policy = ProductionAuthorizationPolicy()

    @property
    @override
    def name(self) -> str:
        return "parlant-cloud"

    @override
    async def configure_app(self, app: FastAPI) -> FastAPI:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        return app

    def _is_trusted(self, headers: Mapping[str, str]) -> bool:
        token = headers.get(PROJECT_TOKEN_HEADER, "")
        return bool(token) and hmac.compare_digest(token, self._project_token)

    @override
    async def check_permission(self, request: Request, operation: Operation) -> bool:
        if self._is_trusted(request.headers):
            return True
        if operation == Operation.ACCESS_INTEGRATED_UI:
            host = request.headers.get("host", "")
            if host.startswith("localhost") or host.startswith("127.0.0.1"):
                return True
        return await self._production_policy.check_permission(request, operation)

    @override
    async def check_rate_limit(self, request: Request, operation: Operation) -> bool:
        if self._is_trusted(request.headers):
            return True
        return await self._production_policy.check_rate_limit(request, operation)

    @override
    async def check_websocket_permission(
        self,
        websocket: WebSocket,
        operation: Operation,
    ) -> bool:
        if self._is_trusted(websocket.headers):
            return True
        return await self._production_policy.check_websocket_permission(websocket, operation)


# ---------------------------------------------------------------------------
# ParlantCloudTracer
# ---------------------------------------------------------------------------


class ParlantCloudTracer(Tracer):
    def __init__(self, project_id: str = "") -> None:
        self._project_id = project_id

        self._endpoint = os.getenv(
            "PARLANT_CLOUD_OTEL_ENDPOINT",
            f"{_get_cloud_base_url()}/v1/traces",
        )

        self._project_token = os.getenv("PARLANT_CLOUD_PROJECT_TOKEN", "")

        self._spans = contextvars.ContextVar[str](
            "tracer_spans",
            default="",
        )

        self._attributes = contextvars.ContextVar[Mapping[str, AttributeValue]](
            "tracer_attributes",
            default={},
        )

        self._trace_id = contextvars.ContextVar[str](
            "tracer_trace_id",
            default="",
        )

        self._current_span = contextvars.ContextVar[Span | None](
            "tracer_current_span",
            default=None,
        )

        self._otel_context = contextvars.ContextVar[context.Context | None](
            "parlant_cloud_tracer_otel_context",
            default=None,
        )

    async def __aenter__(self) -> Self:
        headers = {}
        if self._project_token:
            headers["authorization"] = f"Bearer {self._project_token}"
        else:
            _logger.info(
                "Parlant Cloud tracing is not configured. Learn more at https://parlant.io/cloud"
            )

        span_exporter = HTTPOTLPSpanExporter(
            endpoint=self._endpoint,
            headers=headers,
        )

        processor = BatchSpanProcessor(
            span_exporter=span_exporter,
            schedule_delay_millis=1000,
            max_queue_size=1000,
            max_export_batch_size=100,
        )

        original_on_end = processor.on_end

        def filtered_on_end(span: ReadableSpan) -> None:
            attributes = dict(span.attributes) if span.attributes else {}
            if attributes.get("http.request.operation") == "create_event":
                original_on_end(span)

        setattr(processor, "on_end", filtered_on_end)

        resource_attributes: dict[str, str] = {
            "service.name": "parlant-cloud-tracer",
        }
        if self._project_id:
            resource_attributes["project_id"] = self._project_id

        resource = Resource.create(resource_attributes)
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(processor)
        set_tracer_provider(provider)
        self._tracer_provider = provider
        self._otel_tracer = provider.get_tracer(__name__)
        self._processor = processor
        self._initialized = True

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if self._processor:
            try:
                self._processor.force_flush()
                self._processor.shutdown()  # type: ignore[no-untyped-call]
            except Exception as e:
                _logger.warning(f"Error during ParlantCloudTracer shutdown: {e}")

        return False

    @contextmanager
    @override
    def span(
        self,
        span_id: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> Iterator[None]:
        current_span_chain = self._spans.get()
        current_attributes = self._attributes.get()
        new_attributes = {**current_attributes, **attributes}

        if not current_span_chain:
            new_span_chain = span_id
            # Use existing trace_id if set by CompositeTracer, otherwise generate
            existing = self._trace_id.get()
            if existing:
                custom_trace_id = existing
            else:
                custom_trace_id = self._generate_trace_id()
            trace_id_reset_token = self._trace_id.set(custom_trace_id)

            # Inject Parlant trace_id into the OTEL context so the SDK uses
            # the same ID (UUID hex → 128-bit int). Must use is_remote=True
            # and a valid span_id so the SDK treats it as a real parent.
            import random

            otel_trace_id = int(custom_trace_id.replace("-", ""), 16)
            span_ctx = trace.SpanContext(
                trace_id=otel_trace_id,
                span_id=random.getrandbits(64),
                is_remote=True,
                trace_flags=trace.TraceFlags(trace.TraceFlags.SAMPLED),
            )
            seeded_span = trace.NonRecordingSpan(span_ctx)
            isolated_ctx = trace.set_span_in_context(seeded_span, context.Context())
            otel_context_reset_token = self._otel_context.set(isolated_ctx)
        else:
            new_span_chain = current_span_chain + f"::{span_id}"
            trace_id_reset_token = None
            stored_ctx = self._otel_context.get()
            if stored_ctx is None:
                isolated_ctx = context.Context()
            else:
                isolated_ctx = stored_ctx
            otel_context_reset_token = None

        spans_reset_token = self._spans.set(new_span_chain)
        attributes_reset_token = self._attributes.set(new_attributes)

        otel_tracer = self._otel_tracer

        otel_span = otel_tracer.start_span(
            name=span_id,
            context=isolated_ctx,
            attributes=dict(new_attributes),
        )

        new_ctx = trace.set_span_in_context(otel_span, isolated_ctx)
        ctx_token = self._otel_context.set(new_ctx)

        span_reset_token = self._current_span.set(otel_span)

        try:
            with trace.use_span(otel_span, end_on_exit=True):
                yield
        except Exception as e:
            otel_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            otel_span.record_exception(e)
            raise
        finally:
            self._spans.reset(spans_reset_token)
            self._attributes.reset(attributes_reset_token)
            self._current_span.reset(span_reset_token)
            self._otel_context.reset(ctx_token)
            if trace_id_reset_token is not None:
                self._trace_id.reset(trace_id_reset_token)
            if otel_context_reset_token is not None:
                self._otel_context.reset(otel_context_reset_token)

    @contextmanager
    @override
    def attributes(
        self,
        attributes: Mapping[str, AttributeValue],
    ) -> Iterator[None]:
        current_attributes = self._attributes.get()
        new_attributes = {**current_attributes, **attributes}

        attributes_reset_token = self._attributes.set(new_attributes)

        current_span = self._current_span.get()
        if current_span and current_span.is_recording():
            for key, value in attributes.items():
                current_span.set_attribute(key, value)

        try:
            yield
        finally:
            self._attributes.reset(attributes_reset_token)

    @property
    @override
    def trace_id(self) -> str:
        if trace_id := self._trace_id.get():
            return trace_id
        return "<main>"

    @property
    @override
    def span_id(self) -> str:
        if spans := self._spans.get():
            return spans
        return "<main>"

    @override
    def get_attribute(
        self,
        name: str,
    ) -> AttributeValue | None:
        attributes = self._attributes.get()
        return attributes.get(name, None)

    @override
    def set_attribute(
        self,
        name: str,
        value: AttributeValue,
    ) -> None:
        current_attributes = self._attributes.get()
        new_attributes = {**current_attributes, name: value}
        self._attributes.set(new_attributes)

        current_span = self._current_span.get()
        if current_span and current_span.is_recording():
            current_span.set_attribute(name, value)

    @override
    def add_event(
        self,
        name: str,
        attributes: Mapping[str, AttributeValue] = {},
    ) -> None:
        transformed_attributes = dict(attributes)

        current_span = self._current_span.get()
        if current_span and current_span.is_recording():
            current_span.add_event(name, transformed_attributes)

    @override
    def flush(self) -> None:
        if hasattr(self, "_processor") and self._processor:
            try:
                self._processor.force_flush()
            except Exception as e:
                _logger.warning(f"Failed to flush spans: {e}")


# ---------------------------------------------------------------------------
# ParlantCloudLogger
# ---------------------------------------------------------------------------


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
        self._endpoint = f"{_get_cloud_base_url()}/v1/logs"
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


# ---------------------------------------------------------------------------
# ParlantCloudMeter (Counter, Histogram, DurationHistogram)
# ---------------------------------------------------------------------------


class ParlantCloudCounter(Counter):
    def __init__(self, otel_counter: OTELCounter) -> None:
        self._otel_counter = otel_counter

    @override
    async def increment(
        self,
        value: int,
        attributes: Mapping[str, str] | None = None,
    ) -> None:
        self._otel_counter.add(value, attributes or {})


class ParlantCloudHistogram(Histogram):
    def __init__(self, otel_histogram: OTELHistogram) -> None:
        self._otel_histogram = otel_histogram

    @override
    async def record(
        self,
        value: float,
        attributes: Mapping[str, str] | None = None,
    ) -> None:
        self._otel_histogram.record(value, attributes or {})


class ParlantCloudDurationHistogram(DurationHistogram):
    def __init__(self, otel_histogram: OTELHistogram) -> None:
        self._otel_histogram = otel_histogram

    @override
    async def record(
        self,
        value: float,
        attributes: Mapping[str, str] | None = None,
    ) -> None:
        self._otel_histogram.record(value, attributes or {})

    @override
    @asynccontextmanager
    async def measure(
        self,
        attributes: Mapping[str, str] | None = None,
    ) -> AsyncGenerator[None, None]:
        start_time = asyncio.get_running_loop().time()
        try:
            yield
        finally:
            duration = asyncio.get_running_loop().time() - start_time
            await self.record(duration, attributes)


class ParlantCloudMeter(Meter):
    def __init__(self, project_id: str = "") -> None:
        self._project_id = project_id
        self._endpoint = f"{_get_cloud_base_url()}/v1/metrics"
        self._project_token = os.getenv("PARLANT_CLOUD_PROJECT_TOKEN", "")

        self._meter_provider: MeterProvider | None = None
        self._metric_exporter: OTLPMetricExporter | None = None
        self._metric_reader: PeriodicExportingMetricReader | None = None
        self._otel_meter: OTELMeter | None = None

    async def __aenter__(self) -> Self:
        resource_attributes: dict[str, str] = {
            "service.name": "parlant-cloud-meter",
        }
        if self._project_id:
            resource_attributes["project_id"] = self._project_id

        resource = Resource.create(resource_attributes)

        headers = {}
        if self._project_token:
            headers["authorization"] = f"Bearer {self._project_token}"

        self._metric_exporter = OTLPMetricExporter(
            endpoint=self._endpoint,
            headers=headers,
        )

        self._metric_reader = PeriodicExportingMetricReader(
            exporter=self._metric_exporter,
            export_interval_millis=1000,
        )

        self._meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[self._metric_reader],
        )

        self._otel_meter = self._meter_provider.get_meter(__name__)

        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if self._metric_reader:
            try:
                self._metric_reader.force_flush()
                self._metric_reader.shutdown()  # type: ignore[no-untyped-call]
            except Exception as e:
                _logger.warning(f"Error during ParlantCloudMeter shutdown: {e}")

        return False

    @override
    def create_counter(
        self,
        name: str,
        description: str,
    ) -> Counter:
        if not self._otel_meter:
            raise RuntimeError("ParlantCloudMeter must be used as an async context manager")

        otel_counter = self._otel_meter.create_counter(
            name=name,
            description=description,
        )
        return ParlantCloudCounter(otel_counter)

    @override
    def create_custom_histogram(
        self,
        name: str,
        description: str,
        unit: str,
    ) -> Histogram:
        if not self._otel_meter:
            raise RuntimeError("ParlantCloudMeter must be used as an async context manager")

        otel_histogram = self._otel_meter.create_histogram(
            name=name,
            description=description,
            unit=unit,
        )
        return ParlantCloudHistogram(otel_histogram)

    @override
    def create_duration_histogram(
        self,
        name: str,
        description: str,
    ) -> DurationHistogram:
        if not self._otel_meter:
            raise RuntimeError("ParlantCloudMeter must be used as an async context manager")

        otel_histogram = self._otel_meter.create_histogram(
            name=name,
            description=description,
            unit="s",
        )
        return ParlantCloudDurationHistogram(otel_histogram)


# ---------------------------------------------------------------------------
# Tunnel wiring
# ---------------------------------------------------------------------------

_MAX_RECONNECT_DELAY = 60.0


class WebSocketTunnelService(TunnelService):
    """Tunnel that connects to the platform via WebSocket."""

    def __init__(
        self,
        url: str,
        token: str,
        dispatcher: TunnelRequestDispatcher,
        initial_reconnect_delay: float = 1.0,
    ) -> None:
        self._url = url
        self._token = token
        self._dispatcher = dispatcher
        self._initial_reconnect_delay = initial_reconnect_delay
        self._running = False
        self._stop_event: asyncio.Event | None = None
        self._websocket: Any | None = None

    async def start(self) -> None:
        if not self._token:
            raise ValueError(
                "PARLANT_CLOUD_PROJECT_TOKEN is required to start the tunnel. "
                "Set it in your environment to connect to Parlant Cloud."
            )

        self._running = True
        self._stop_event = asyncio.Event()
        reconnect_delay = self._initial_reconnect_delay

        while self._running:
            try:
                await self._connect_and_listen()
                reconnect_delay = self._initial_reconnect_delay
            except asyncio.CancelledError:
                await self.stop()
                return
            except Exception as e:
                if not self._running:
                    return
                _logger.warning(
                    f"Tunnel connection failed: {e}. Reconnecting in {reconnect_delay:.1f}s..."
                )
                await self._wait_for_reconnect_or_stop(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, _MAX_RECONNECT_DELAY)

    async def stop(self) -> None:
        self._running = False
        if self._stop_event:
            self._stop_event.set()

        if self._websocket is not None:
            await self._websocket.close()

    async def _wait_for_reconnect_or_stop(self, delay: float) -> None:
        if not self._stop_event:
            await asyncio.sleep(delay)
            return

        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=delay)
        except TimeoutError:
            pass

    async def _connect_and_listen(self) -> None:
        headers = {"Authorization": f"Bearer {self._token}"}

        async with websockets.connect(self._url, additional_headers=headers) as ws:
            self._websocket = ws
            _logger.info(f"Tunnel connected to {self._url}")

            try:
                async for raw_message in ws:
                    if not self._running:
                        break

                    try:
                        message: dict[str, Any] = _json_mod.loads(raw_message)
                        request = TunnelRequest(
                            request_id=message["request_id"],
                            method=message["method"],
                            params=message.get("params", {}),
                        )

                        response = await self._dispatcher.dispatch(request)
                        await ws.send(_json_mod.dumps(response.to_dict()))

                    except Exception as e:
                        _logger.error(f"Error processing tunnel message: {e}")
                        request_id = (
                            message.get("request_id", "unknown")
                            if isinstance(message, dict)
                            else "unknown"
                        )
                        try:
                            error_resp = TunnelResponse(
                                request_id=request_id,
                                error=str(e),
                            )
                            await ws.send(_json_mod.dumps(error_resp.to_dict()))
                        except Exception:
                            pass
            finally:
                self._websocket = None


def _create_tunnel_service(
    session_module: SessionModule,
    agent_module: AgentModule,
    customer_module: CustomerModule,
    tag_module: TagModule,
    background_task_service: BackgroundTaskService,
    logger: Logger | None = None,
) -> WebSocketTunnelService | None:
    """Create a tunnel service if PARLANT_CLOUD_PROJECT_TOKEN is set."""
    token = os.environ.get("PARLANT_CLOUD_PROJECT_TOKEN", "")
    if not token:
        return None

    base_url = _get_cloud_base_url()
    ws_url = base_url.replace("https://", "wss://").replace("http://", "ws://") + "/cloud"

    dispatcher = TunnelRequestDispatcher(
        session_module=session_module,
        agent_module=agent_module,
        customer_module=customer_module,
        tag_module=tag_module,
        logger=logger,
    )

    return WebSocketTunnelService(
        url=ws_url,
        token=token,
        dispatcher=dispatcher,
    )


# ---------------------------------------------------------------------------
# Module entry point — configure_container
# ---------------------------------------------------------------------------


async def configure_container(container: Container) -> Container:
    project_token = os.environ.get("PARLANT_CLOUD_PROJECT_TOKEN", "")
    if project_token:
        container[AuthorizationPolicy] = ParlantCloudAuthorizationPolicy(project_token)
    else:
        return container

    logger = container[Logger]
    base_url = _get_cloud_base_url()

    auth_url = f"{base_url}/v1/auth/project-token"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                auth_url,
                headers={"Authorization": f"Bearer {project_token}"},
            )
            resp.raise_for_status()
            auth_data = resp.json()
            project_id: str = auth_data.get("project_id", "")
    except Exception:
        logger.warning("Parlant Cloud project token validation failed; observability disabled")
        return container

    if not project_id:
        logger.warning("Parlant Cloud auth response missing project_id; observability disabled")
        return container

    tracer = container[Tracer]
    cloud_tracer = await _exit_stack.enter_async_context(ParlantCloudTracer(project_id=project_id))
    if isinstance(tracer, CompositeTracer):
        tracer.append(cloud_tracer)
    else:
        container.define(Tracer, CompositeTracer([tracer, cloud_tracer]))

    existing_logger = container[Logger]
    cloud_logger = await _exit_stack.enter_async_context(
        ParlantCloudLogger(tracer=tracer, project_id=project_id)
    )
    if isinstance(existing_logger, CompositeLogger):
        existing_logger.append(cloud_logger)
    else:
        container.define(Logger, CompositeLogger([existing_logger, cloud_logger]))

    try:
        _ = container[Meter]
    except Exception:
        cloud_meter = await _exit_stack.enter_async_context(
            ParlantCloudMeter(project_id=project_id)
        )
        container[Meter] = cloud_meter

    return container


async def initialize_container(container: Container) -> None:
    """Start the tunnel after core application modules are available."""
    logger = container[Logger]

    try:
        tunnel = _create_tunnel_service(
            session_module=container[SessionModule],
            agent_module=container[AgentModule],
            customer_module=container[CustomerModule],
            tag_module=container[TagModule],
            background_task_service=container[BackgroundTaskService],
            logger=container[Logger],
        )

        if tunnel:
            container[TunnelService] = tunnel
            await container[BackgroundTaskService].start(
                tunnel.start(),
                tag="parlant-cloud-tunnel",
            )
    except Exception as e:
        logger.warning(f"Failed to start Parlant Cloud tunnel: {e}")
