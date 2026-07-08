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

"""Parlant Cloud module lifecycle entry points.

`configure_container` and `initialize_container` are called by the SDK when
PARLANT_CLOUD_PROJECT_TOKEN is set. They wire the cloud-backed authorization
policy, tracer, logger, meter, and tunnel service into the lagom container.
"""

import os
from contextlib import AsyncExitStack
from dataclasses import dataclass

import httpx
from lagom import Container

from parlant.api.authorization import AuthorizationPolicy
from parlant.core.app_modules.agents import AgentModule
from parlant.core.app_modules.customers import CustomerModule
from parlant.core.app_modules.sessions import SessionModule
from parlant.core.app_modules.groups import GroupModule
from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.loggers import CompositeLogger, Logger
from parlant.core.meter import Meter
from parlant.core.tracer import CompositeTracer, Tracer
from parlant.core.tunnels import TunnelService

from parlant.adapters.modules.parlant_cloud.auth import ParlantCloudAuthorizationPolicy
from parlant.adapters.modules.parlant_cloud.config import _get_cloud_base_url
from parlant.adapters.modules.parlant_cloud.logger import ParlantCloudLogger
from parlant.adapters.modules.parlant_cloud.meter import ParlantCloudMeter
from parlant.adapters.modules.parlant_cloud.tracer import ParlantCloudTracer
from parlant.adapters.modules.parlant_cloud.tunnel import _create_tunnel_service

_exit_stack = AsyncExitStack()


@dataclass(frozen=True)
class CloudProjectAuth:
    project_id: str
    authenticated: bool


_cloud_project_auth: CloudProjectAuth | None = None


async def configure_container(container: Container) -> Container:
    global _cloud_project_auth

    project_token = os.environ.get("PARLANT_CLOUD_PROJECT_TOKEN", "")
    if project_token:
        container[AuthorizationPolicy] = ParlantCloudAuthorizationPolicy(project_token)
    else:
        _cloud_project_auth = None
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
            project_id = auth_data.get("project_id", "")
    except Exception:
        _cloud_project_auth = CloudProjectAuth(
            project_id="",
            authenticated=False,
        )
        logger.warning("Parlant Cloud project token validation failed; observability disabled")
        return container

    if not isinstance(project_id, str) or not project_id:
        _cloud_project_auth = CloudProjectAuth(
            project_id="",
            authenticated=False,
        )
        logger.warning("Parlant Cloud auth response missing project_id; observability disabled")
        return container

    _cloud_project_auth = CloudProjectAuth(
        project_id=project_id,
        authenticated=True,
    )

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
        if _cloud_project_auth is not None and not _cloud_project_auth.authenticated:
            return

        tunnel = _create_tunnel_service(
            session_module=container[SessionModule],
            agent_module=container[AgentModule],
            customer_module=container[CustomerModule],
            group_module=container[GroupModule],
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
