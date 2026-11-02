import asyncio
import os
from typing import Any, Coroutine, cast
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from lagom.exceptions import UnresolvableType

from parlant.core.container import Container

import parlant.adapters.modules.parlant_dashboard.lifecycle as lifecycle
from parlant.adapters.modules.parlant_dashboard import (
    ParlantDashboardTunnelService,
    configure_container,
    create_tunnel_service,
    initialize_container,
)
from parlant.adapters.modules.parlant_dashboard.lifecycle import DashboardProjectAuth
from parlant.adapters.modules.parlant_dashboard.logger import ParlantDashboardLogger
from parlant.adapters.modules.parlant_dashboard.tracer import ParlantDashboardTracer
from parlant.core.application import Application
from parlant.core.application_context import ApplicationContext
from parlant.core.health import HealthReporter
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.loggers import CompositeLogger, Logger
from parlant.core.meter import Meter
from parlant.core.tracer import CompositeTracer, LocalTracer, Tracer
from parlant.sdk import _should_configure_parlant_dashboard
from tests.test_utilities import _TestLogger


class FakeBackgroundTaskService(BackgroundTaskService):
    # Subclasses the real service so the container's type-validated binding
    # accepts it; only `start` is overridden to record the call.
    def __init__(self) -> None:
        super().__init__(_TestLogger())
        self.started = False
        self.tag: str | None = None

    async def start(self, f: Coroutine[Any, Any, None], /, *, tag: str) -> asyncio.Task[None]:
        self.started = True
        self.tag = tag
        f.close()

        async def _noop() -> None: ...

        task = asyncio.get_running_loop().create_task(_noop())
        await task
        return task


def _health_reporter() -> HealthReporter:
    return HealthReporter(ApplicationContext(instance_id="test-instance"))


def _fake_application() -> Application:
    # A real Application over mock modules: satisfies the container's
    # type-validated binding while keeping these wiring tests light.
    return Application(
        agent_module=AsyncMock(),
        session_module=AsyncMock(),
        service_module=AsyncMock(),
        group_module=AsyncMock(),
        customer_module=AsyncMock(),
        rule_module=AsyncMock(),
        context_variable_module=AsyncMock(),
        relationship_module=AsyncMock(),
        glossary_module=AsyncMock(),
        indexing_module=AsyncMock(),
        canned_response_module=AsyncMock(),
    )


async def test_that_sdk_dashboard_module_is_not_loaded_without_dashboard_credentials() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PARLANT_DASHBOARD_PROJECT_TOKEN", None)

        assert _should_configure_parlant_dashboard() is False


@pytest.mark.parametrize("environment", ["development", "production"])
async def test_that_sdk_dashboard_module_is_loaded_with_project_token(environment: str) -> None:
    with patch.dict(
        os.environ,
        {"PARLANT_DASHBOARD_PROJECT_TOKEN": "test-token", "PARLANT_ENV": environment},
    ):
        assert _should_configure_parlant_dashboard() is True


async def test_that_sdk_dashboard_module_is_not_loaded_in_the_test_environment_even_with_a_project_token() -> (
    None
):
    # A connected checkout's .env supplies the real project's token, and the
    # generated config loads .env; a test server must still stay local.
    with patch.dict(
        os.environ, {"PARLANT_DASHBOARD_PROJECT_TOKEN": "test-token", "PARLANT_ENV": "test"}
    ):
        assert _should_configure_parlant_dashboard() is False


async def test_that_tunnel_is_not_created_without_project_token() -> None:
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PARLANT_DASHBOARD_PROJECT_TOKEN", None)

        result = create_tunnel_service(
            application=_fake_application(),
            health_reporter=_health_reporter(),
            logger=_TestLogger(),
        )

        assert result is None


async def test_that_tunnel_is_created_with_project_token() -> None:
    with patch.dict(os.environ, {"PARLANT_DASHBOARD_PROJECT_TOKEN": "test-token"}):
        result = create_tunnel_service(
            application=_fake_application(),
            health_reporter=_health_reporter(),
            logger=_TestLogger(),
        )

        assert result is not None
        assert isinstance(result, ParlantDashboardTunnelService)


async def test_that_tunnel_uses_dashboard_base_url() -> None:
    with patch.dict(
        os.environ,
        {
            "PARLANT_DASHBOARD_PROJECT_TOKEN": "test-token",
            "PARLANT_DASHBOARD_BASE_URL": "https://api.emcie.xyz",
        },
    ):
        os.environ.pop("PARLANT_DASHBOARD_TUNNEL_URL", None)
        result = create_tunnel_service(
            application=_fake_application(),
            health_reporter=_health_reporter(),
            logger=_TestLogger(),
        )

        assert result is not None
        assert isinstance(result, ParlantDashboardTunnelService)
        assert result._url == "wss://api.emcie.xyz/cloud"


async def test_that_tunnel_uses_explicit_parlant_dashboard_tunnel_url() -> None:
    with patch.dict(
        os.environ,
        {
            "PARLANT_DASHBOARD_PROJECT_TOKEN": "test-token",
            "PARLANT_DASHBOARD_BASE_URL": "https://api.emcie.xyz",
            "PARLANT_DASHBOARD_TUNNEL_URL": "http://localhost:2500",
        },
    ):
        result = create_tunnel_service(
            application=_fake_application(),
            health_reporter=_health_reporter(),
            logger=_TestLogger(),
        )

        assert result is not None
        assert isinstance(result, ParlantDashboardTunnelService)
        assert result._url == "ws://localhost:2500/cloud"


async def test_that_tunnel_preserves_an_explicit_websocket_tunnel_path() -> None:
    with patch.dict(
        os.environ,
        {
            "PARLANT_DASHBOARD_PROJECT_TOKEN": "test-token",
            "PARLANT_DASHBOARD_TUNNEL_URL": "ws://localhost:2500/cloud",
        },
    ):
        result = create_tunnel_service(
            application=_fake_application(),
            health_reporter=_health_reporter(),
            logger=_TestLogger(),
        )

        assert result is not None
        assert isinstance(result, ParlantDashboardTunnelService)
        assert result._url == "ws://localhost:2500/cloud"


async def test_that_the_tunnel_defaults_to_the_production_dashboard_api() -> None:
    with patch.dict(
        os.environ,
        {
            "PARLANT_DASHBOARD_PROJECT_TOKEN": "test-token",
        },
    ):
        os.environ.pop("PARLANT_DASHBOARD_TUNNEL_URL", None)
        os.environ.pop("PARLANT_DASHBOARD_BASE_URL", None)
        result = create_tunnel_service(
            application=_fake_application(),
            health_reporter=_health_reporter(),
            logger=_TestLogger(),
        )

        assert result is not None
        assert isinstance(result, ParlantDashboardTunnelService)
        # dashboard.parlant.io is the web app; the tunnel is served beside the API.
        assert result._url == "wss://api.parlant.io/cloud"


async def test_that_the_collectors_default_to_the_production_dashboard_api() -> None:
    with patch.dict(os.environ, {}):
        os.environ.pop("PARLANT_DASHBOARD_OTEL_URL", None)
        os.environ.pop("PARLANT_DASHBOARD_BASE_URL", None)
        tracer = ParlantDashboardTracer()
        logger = ParlantDashboardLogger(tracer=LocalTracer())

        assert tracer._endpoint == "https://api.parlant.io/v1/traces"
        assert logger._endpoint == "https://api.parlant.io/v1/logs"


async def test_that_dashboard_otel_url_configures_logs_and_traces_collectors() -> None:
    with patch.dict(
        os.environ,
        {
            "PARLANT_DASHBOARD_OTEL_URL": "http://localhost:4318",
        },
    ):
        os.environ.pop("PARLANT_DASHBOARD_TUNNEL_URL", None)
        os.environ.pop("PARLANT_DASHBOARD_BASE_URL", None)
        tracer = ParlantDashboardTracer()
        logger = ParlantDashboardLogger(tracer=LocalTracer())

        assert tracer._endpoint == "http://localhost:4318/v1/traces"
        assert logger._endpoint == "http://localhost:4318/v1/logs"


async def test_that_the_dashboard_module_does_not_bind_a_meter() -> None:
    # The dashboard no longer receives metrics, so a server signed in to it
    # keeps whatever meter it has, and gets none from the dashboard module.
    def dashboard(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/auth/project-token"
        return httpx.Response(200, json={"project_id": "project-1"})

    real_client = httpx.AsyncClient

    with (
        patch.dict(
            os.environ,
            {
                "PARLANT_DASHBOARD_PROJECT_TOKEN": "test-token",
                "PARLANT_DASHBOARD_BASE_URL": "http://dashboard.test",
                "PARLANT_GUEST_TOKEN_SECRET": "test-secret",
            },
        ),
        patch.object(
            httpx,
            "AsyncClient",
            lambda: real_client(transport=httpx.MockTransport(dashboard)),
        ),
    ):
        os.environ.pop("PARLANT_DASHBOARD_TUNNEL_URL", None)
        os.environ.pop("PARLANT_DASHBOARD_OTEL_URL", None)
        # Composites, so the module appends its tracer and logger rather than
        # rebinding types the container has already resolved.
        container = Container()
        container[Logger] = CompositeLogger([_TestLogger()])
        container[Tracer] = CompositeTracer([LocalTracer()])

        try:
            await configure_container(container)

            with pytest.raises(UnresolvableType):
                container[Meter]
        finally:
            await lifecycle._exit_stack.aclose()
            lifecycle._dashboard_project_auth = None


async def test_that_dashboard_initializer_starts_tunnel_after_session_module_exists() -> None:
    with patch.dict(os.environ, {"PARLANT_DASHBOARD_PROJECT_TOKEN": "test-token"}):
        lifecycle._dashboard_project_auth = DashboardProjectAuth(
            project_id="project-1",
            authenticated=True,
        )
        container = Container()
        background_task_service = FakeBackgroundTaskService()

        # The container validates bindings by type; bare object() sentinels
        # no longer pass, so use spec'd mocks.
        container[Application] = _fake_application()
        container[BackgroundTaskService] = cast(BackgroundTaskService, background_task_service)
        container[Logger] = _TestLogger()
        container[Tracer] = LocalTracer()
        container[HealthReporter] = _health_reporter()

        try:
            await initialize_container(container)

            assert background_task_service.started
            assert background_task_service.tag == "parlant-dashboard-tunnel"
        finally:
            lifecycle._dashboard_project_auth = None
