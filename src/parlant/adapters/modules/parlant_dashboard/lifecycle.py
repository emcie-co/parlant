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

"""Parlant Dashboard module lifecycle entry points.

`configure_container` and `initialize_container` are called by the SDK when
PARLANT_DASHBOARD_PROJECT_TOKEN is set. They wire the Dashboard-backed authorization
policy, tracer, logger, and tunnel service into the lagom container.
"""

import os
from contextlib import AsyncExitStack
from dataclasses import dataclass

import httpx
from parlant.core.container import Container

from parlant.api.authorization import AuthorizationPolicy
from parlant.core.application import Application
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.loggers import CompositeLogger, Logger
from parlant.core.health import HealthReporter
from parlant.core.tracer import CompositeTracer, Tracer
from parlant.core.tunnels import TunnelService

from parlant.adapters.modules.parlant_dashboard.auth import (
    ParlantDashboardAuthorizationPolicy,
    build_default_inner_policy,
)
from parlant.core.customers import CustomerStore
from parlant.adapters.modules.parlant_dashboard.config import _get_dashboard_base_url
from parlant.adapters.modules.parlant_dashboard.logger import ParlantDashboardLogger
from parlant.adapters.modules.parlant_dashboard.tracer import ParlantDashboardTracer
from parlant.adapters.modules.parlant_dashboard.tunnel import create_tunnel_service

_exit_stack = AsyncExitStack()


@dataclass(frozen=True)
class DashboardProjectAuth:
    project_id: str
    authenticated: bool


_dashboard_project_auth: DashboardProjectAuth | None = None


async def configure_container(container: Container) -> Container:
    global _dashboard_project_auth

    project_token = os.environ.get("PARLANT_DASHBOARD_PROJECT_TOKEN", "")
    if project_token:
        if not os.environ.get("PARLANT_GUEST_TOKEN_SECRET"):
            container[Logger].warning(
                "PARLANT_GUEST_TOKEN_SECRET is not set; generated an ephemeral guest "
                "token secret. Guest sessions will not survive a server restart."
            )

        # Memoized: the inner composite policy holds rate-limiter state and
        # (possibly ephemeral) token secrets that must be stable for the
        # process lifetime. Deferred so CustomerStore resolves once the server
        # actually stands up.
        dashboard_policy: AuthorizationPolicy | None = None

        def _dashboard_policy(c: Container) -> AuthorizationPolicy:
            nonlocal dashboard_policy
            if dashboard_policy is None:
                dashboard_policy = ParlantDashboardAuthorizationPolicy(
                    project_token=project_token,
                    inner=build_default_inner_policy(customer_store=c[CustomerStore]),
                )
            return dashboard_policy

        container[AuthorizationPolicy] = _dashboard_policy
    else:
        _dashboard_project_auth = None
        return container

    logger = container[Logger]
    base_url = _get_dashboard_base_url()

    auth_url = f"{base_url}/v1/auth/project-token"
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                auth_url,
                headers={"Authorization": f"Bearer {project_token}"},
            )
            resp.raise_for_status()
            auth_data = resp.json()
            project_id = auth_data.get("project_id", "")
    except Exception as exc:
        _dashboard_project_auth = DashboardProjectAuth(
            project_id="",
            authenticated=False,
        )
        if isinstance(exc, httpx.HTTPStatusError):
            reason = (
                f"the Parlant Dashboard at {auth_url} rejected the project token "
                f"(HTTP {exc.response.status_code}) — check PARLANT_DASHBOARD_PROJECT_TOKEN"
            )
        elif isinstance(exc, httpx.RequestError):
            reason = (
                f"could not connect to the Parlant Dashboard at {auth_url} "
                f"({type(exc).__name__}: {exc}) — check that host:port is reachable from here "
                f"(PARLANT_DASHBOARD_BASE_URL)"
            )
        else:
            reason = f"unexpected error validating the project token against {auth_url} ({type(exc).__name__}: {exc})"
        logger.error(
            f"Parlant Dashboard project token validation failed: {reason}; observability disabled"
        )
        return container

    if not isinstance(project_id, str) or not project_id:
        _dashboard_project_auth = DashboardProjectAuth(
            project_id="",
            authenticated=False,
        )
        logger.error("Parlant Dashboard auth response missing project_id; observability disabled")
        return container

    _dashboard_project_auth = DashboardProjectAuth(
        project_id=project_id,
        authenticated=True,
    )

    tracer = container[Tracer]
    dashboard_tracer = await _exit_stack.enter_async_context(
        ParlantDashboardTracer(project_id=project_id, logger=container[Logger])
    )
    if isinstance(tracer, CompositeTracer):
        tracer.append(dashboard_tracer)
    else:
        container[Tracer] = CompositeTracer([tracer, dashboard_tracer])

    existing_logger = container[Logger]
    dashboard_logger = await _exit_stack.enter_async_context(
        ParlantDashboardLogger(tracer=tracer, project_id=project_id)
    )
    if isinstance(existing_logger, CompositeLogger):
        existing_logger.append(dashboard_logger)
    else:
        container[Logger] = CompositeLogger([existing_logger, dashboard_logger])

    return container


async def initialize_container(container: Container) -> None:
    """Start the tunnel after core application modules are available."""
    logger = container[Logger]

    try:
        if _dashboard_project_auth is not None and not _dashboard_project_auth.authenticated:
            return

        tunnel = create_tunnel_service(
            application=container[Application],
            health_reporter=container[HealthReporter],
            logger=container[Logger],
            tracer=container[Tracer],
        )

        if tunnel:
            container[TunnelService] = tunnel
            await container[BackgroundTaskService].start(
                tunnel.start(),
                tag="parlant-dashboard-tunnel",
            )
    except Exception as e:
        logger.error(f"Failed to start Parlant Dashboard tunnel: {e}")
