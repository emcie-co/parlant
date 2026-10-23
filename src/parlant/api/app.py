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

import asyncio
from contextvars import ContextVar
import os
import traceback
from typing import Any, Awaitable, Callable, Iterable, Mapping, Protocol, TypeAlias, cast

import mimetypes

from fastapi import APIRouter, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi import routing as fastapi_routing
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from starlette.types import Receive, Scope, Send
from starlette.routing import BaseRoute, Match


from parlant.core.container import Container

from parlant.adapters.loggers.websocket import WebSocketLogger
from parlant.api.health import configure_healthz
from parlant.core.health import HealthReporter
from parlant.core.nlp.model_routing import ModelConfigurationError
from parlant.api import agents
from parlant.api import sessions
from parlant.api import groups
from parlant.api import customers
from parlant.api import logs
from parlant.api import debug
from parlant.api.authentication import AuthenticationError
from parlant.api.authorization import (
    AuthorizationException,
    AuthorizationPolicy,
    Operation,
    RateLimitExceededException,
)
from parlant.core.app_modules.request_context import RequestContext
from parlant.core.version import VERSION
from parlant.core.meter import Meter
from parlant.core.tracer import Tracer
from parlant.core.common import ItemNotFoundError, generate_id
from parlant.core.loggers import Logger
from parlant.core.application import Application
from parlant.core.diagnostic_reports import (
    DiagnosticReportService,
    NOOP_DIAGNOSTIC_REPORT_SERVICE,
)


mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("image/svg+xml", ".svg")


ASGIApplication: TypeAlias = Callable[
    [
        Scope,
        Receive,
        Send,
    ],
    Awaitable[None],
]


class AppWrapper:
    def __init__(self, app: FastAPI) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """FastAPI's built-in exception handling doesn't catch BaseExceptions
        such as asyncio.CancelledError. This causes the server process to terminate
        with an ugly traceback. This wrapper addresses that by specifically allowing
        asyncio.CancelledError to gracefully exit.
        """
        try:
            return await self.app(scope, receive, send)
        except asyncio.CancelledError:
            pass


RECORDED_FLAG = "_otel_metrics_recorded"


def _recent_diagnostic_report_links(container: Container) -> list[dict[str, Any]] | None:
    """The most recent diagnostic reports as /debug/reports/{trace_id} links.

    The count comes from PARLANT_HEALTHZ_REPORT_LINKS (default 3; 0 disables
    the section). Returns None when disabled or when diagnostic reports are
    not configured, so /healthz omits the key entirely.
    """
    try:
        limit = int(os.environ.get("PARLANT_HEALTHZ_REPORT_LINKS", "3"))
    except ValueError:
        limit = 3
    if limit <= 0:
        return None
    if DiagnosticReportService not in container.defined_types:
        return None

    return [
        {
            "trace_id": report.trace_id,
            "session_id": report.session_id,
            "outcome": report.outcome,
            "created_utc": report.created_utc,
            "report_uri": f"/debug/reports/{report.trace_id}",
        }
        for report in container[DiagnosticReportService].list_reports(limit=limit)
    ]


class _RouteContext(Protocol):
    @property
    def original_route(self) -> BaseRoute: ...

    def matches(self, scope: Scope) -> tuple[Match, Scope]: ...


def _resolve_operation_id(request: Request) -> str | None:
    route = request.scope.get("route")
    if isinstance(route, APIRoute):
        return route.operation_id

    routes = request.app.router.routes
    # Newer FastAPI versions retain included routers instead of flattening
    # APIRoutes. Its context iterator supplies the effective paths (including
    # nested inclusion prefixes) while retaining the original operation IDs.
    iter_contexts = getattr(fastapi_routing, "iter_route_contexts", None)
    if callable(iter_contexts):
        contexts = cast(Callable[[Iterable[BaseRoute]], Iterable[_RouteContext]], iter_contexts)
        for context in contexts(routes):
            if context.matches(request.scope)[0] == Match.FULL:
                original = context.original_route
                return original.operation_id if isinstance(original, APIRoute) else None
        return None

    # Older FastAPI versions expose the effective routes directly.
    for candidate in routes:
        if candidate.matches(request.scope)[0] == Match.FULL:
            return candidate.operation_id if isinstance(candidate, APIRoute) else None
    return None


def _app_path(request: Request) -> str:
    """The request path relative to where the API is served. Behind a reverse
    proxy with a prefix (ROOT_PATH, e.g. /ask3Sn2jas), the URL path carries the
    prefix; routes and these checks are relative to it."""
    path = request.url.path
    root_path = str(request.scope.get("root_path") or "")
    if root_path and path.startswith(root_path):
        return path[len(root_path) :] or "/"
    return path


def _is_api_docs_path(path: str) -> bool:
    """FastAPI's own docs pages, matched as whole paths: an agent's path such as
    /agents/docs merely contains "/docs" and is an ordinary API route."""
    return path in ("/docs", "/redoc", "/openapi.json") or path.startswith("/docs/")


def _is_integrated_ui_path(path: str) -> bool:
    """The integrated chat UI, mounted at /chat."""
    return path == "/chat" or path.startswith("/chat/")


async def create_api_app(
    container: Container,
    configure: Callable[[FastAPI], Awaitable[FastAPI | None]] | None = None,
    contextvar_propagation: Mapping[ContextVar[Any], Any] = {},
) -> ASGIApplication:
    logger = container[Logger]
    websocket_logger = container[WebSocketLogger]
    tracer = container[Tracer]
    authorization_policy = container[AuthorizationPolicy]
    application = container[Application]
    request_context = container[RequestContext]

    meter = container[Meter]
    _hist_http_request_duration = meter.create_duration_histogram(
        name="httpreq",
        description="HTTP Request Duration",
    )

    configure_healthz(container)
    health_reporter = container[HealthReporter]

    api_app = FastAPI(
        title="Parlant API",
        description="API documentation for the Parlant server.",
        version=VERSION,
    )

    api_app = await authorization_policy.configure_app(api_app)

    @api_app.middleware("http")
    async def propagate_contextvars_into_request_task(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        for var, value in contextvar_propagation.items():
            var.set(value)
        return await call_next(request)

    @api_app.middleware("http")
    async def add_origin(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        origin = request.headers.get("X-Parlant-Origin")
        if origin:
            request_context.set_origin(origin)
        return await call_next(request)

    @api_app.middleware("http")
    async def handle_cancellation(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        try:
            return await call_next(request)
        except asyncio.CancelledError:
            return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

    @api_app.middleware("http")
    async def add_trace_id(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = _app_path(request)

        # Exceptions raised inside a middleware bypass the app's exception
        # handlers, so the authorization outcomes for these statically-served
        # paths must be translated to responses right here — otherwise an
        # anonymous browser hit surfaces as a 500 instead of a 401/403.
        if _is_api_docs_path(path):
            try:
                await authorization_policy.authorize(
                    request=request,
                    operation=Operation.ACCESS_API_DOCS,
                )
            except AuthenticationError as exc:
                return JSONResponse(status_code=401, content={"detail": str(exc)})
            except AuthorizationException:
                return JSONResponse(status_code=403, content={"detail": "Forbidden"})
            return await call_next(request)

        if _is_integrated_ui_path(path):
            try:
                await authorization_policy.authorize(
                    request=request,
                    operation=Operation.ACCESS_INTEGRATED_UI,
                )
            except AuthenticationError as exc:
                return JSONResponse(status_code=401, content={"detail": str(exc)})
            except AuthorizationException:
                return JSONResponse(status_code=403, content={"detail": "Forbidden"})

            return await call_next(request)

        operation_id = _resolve_operation_id(request)

        if operation_id is None:
            return await call_next(request)

        request_id = generate_id()
        with tracer.span(
            "http.request",
            {
                "http.request.id": request_id,
                "http.request.operation": operation_id,
                "http.request.method": request.method,
                **request.path_params,
            },
        ):
            async with _hist_http_request_duration.measure(
                {
                    "http.request.operation": operation_id,
                    "http.request.method": request.method,
                },
            ):
                return await call_next(request)

    @api_app.exception_handler(RateLimitExceededException)
    async def rate_limit_exceeded_handler(
        request: Request, exc: RateLimitExceededException
    ) -> HTTPException:
        logger.trace(f"Rate limit exceeded: {exc}")

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        )

    @api_app.exception_handler(AuthenticationError)
    async def authentication_error_handler(
        request: Request, exc: AuthenticationError
    ) -> HTTPException:
        logger.trace(f"Authentication error: {exc}")

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        )

    @api_app.exception_handler(AuthorizationException)
    async def authorization_error_handler(
        request: Request, exc: AuthorizationException
    ) -> HTTPException:
        # The full message (including the request headers) is useful for
        # operators but must not be echoed to the caller: it reflects
        # proxy-injected headers back to an unauthorized client.
        logger.trace(f"Authorization error: {exc}")

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden",
        )

    @api_app.exception_handler(ItemNotFoundError)
    async def item_not_found_error_handler(
        request: Request, exc: ItemNotFoundError
    ) -> HTTPException:
        # Client-side misuse, not an operational event: keep it off the
        # user-visible INFO narrative.
        logger.debug(str(exc))

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )

    @api_app.exception_handler(ModelConfigurationError)
    async def model_configuration_error_handler(
        request: Request,
        exc: ModelConfigurationError,
    ) -> JSONResponse:
        logger.debug(str(exc))
        return JSONResponse(
            status_code=exc.http_status,
            content={"detail": exc.diagnostic()},
        )

    @api_app.exception_handler(Exception)
    async def server_error_handler(request: Request, exc: ItemNotFoundError) -> HTTPException:
        # Log the full detail and traceback for operators, but return a generic
        # body: raw exception text leaks driver/provider internals to clients.
        logger.error(str(exc))
        logger.error(str(traceback.format_exception(exc)))

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )

    static_dir = os.path.join(os.path.dirname(__file__), "chat/dist")
    api_app.mount("/chat", StaticFiles(directory=static_dir, html=True), name="static")

    @api_app.get("/", include_in_schema=False)
    async def root() -> Response:
        return RedirectResponse("/chat")

    @api_app.get("/healthz")
    async def health_check(request: Request) -> dict[str, Any]:
        snapshot = health_reporter.snapshot()

        # The full body (component checks, latencies, error classes, report
        # links) is operational detail; unauthorized callers — load balancers,
        # uptime probes — get only the top-level status, still with HTTP 200.
        # Policies that raise on missing credentials count as "not permitted"
        # here: /healthz never turns callers away.
        try:
            permitted = await authorization_policy.check_permission(
                request, Operation.READ_HEALTH_REPORT
            )
        except (AuthenticationError, AuthorizationException):
            permitted = False
        if not permitted:
            return {"status": snapshot["status"]}

        snapshot = await health_reporter.admin_snapshot()
        report_links = _recent_diagnostic_report_links(container)
        if report_links is not None:
            return {**snapshot, "diagnostic_reports": report_links}
        return snapshot

    agent_router = APIRouter(prefix="/agents")

    api_app.include_router(
        router=agents.create_router(
            policy=authorization_policy,
            app=application,
        ),
        prefix="/agents",
    )

    api_app.include_router(
        router=agent_router,
    )

    api_app.include_router(
        prefix="/sessions",
        router=sessions.create_router(
            authorization_policy=authorization_policy,
            app=application,
        ),
    )

    api_app.include_router(
        prefix="/groups",
        router=groups.create_router(
            authorization_policy=authorization_policy,
            app=application,
        ),
    )

    api_app.include_router(
        prefix="/customers",
        router=customers.create_router(
            authorization_policy=authorization_policy,
            app=application,
        ),
    )

    api_app.include_router(
        router=logs.create_router(
            websocket_logger=websocket_logger,
            authorization_policy=authorization_policy,
        )
    )

    api_app.include_router(
        router=debug.create_router(
            diagnostic_report_service=(
                container[DiagnosticReportService]
                if DiagnosticReportService in container.defined_types
                else NOOP_DIAGNOSTIC_REPORT_SERVICE
            ),
            authorization_policy=authorization_policy,
        )
    )

    # Call configure_api hook if provided
    if configure:
        if new_app := await configure(api_app):
            api_app = new_app

    # Store FastAPI app in container for access via Server.api property
    container[FastAPI] = api_app

    return AppWrapper(api_app)
