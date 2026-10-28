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

"""Per-domain handlers for the Dashboard tunnel's RPC surface.

Each handler class translates wire params into calls on one Application
module and serializes the results back into fixed response shapes. The
registry below is the complete tunnel contract: 62 unary methods and 2
session event streams.
"""

from typing import Any

from parlant.core.session_clients import SessionClientRecorder
from parlant.core.application import Application
from parlant.core.health import HealthReporter
from parlant.core.loggers import Logger
from parlant.core.tunnel_handlers.agents import AgentTunnelHandlers
from parlant.core.tunnel_handlers.canned_responses import CannedResponseTunnelHandlers
from parlant.core.tunnel_handlers.common import (
    StreamHandler,
    TunnelHandlerSet,
    UnaryHandler,
    merge_handler_sets,
)
from parlant.core.tunnel_handlers.context_variables import ContextVariableTunnelHandlers
from parlant.core.tunnel_handlers.customers import CustomerTunnelHandlers
from parlant.core.tunnel_handlers.glossary import GlossaryTunnelHandlers
from parlant.core.tunnel_handlers.groups import GroupTunnelHandlers
from parlant.core.tunnel_handlers.indexing import IndexingTunnelHandlers
from parlant.core.tunnel_handlers.relationships import RelationshipTunnelHandlers
from parlant.core.tunnel_handlers.rules import RuleTunnelHandlers
from parlant.core.tunnel_handlers.services import ServiceTunnelHandlers
from parlant.core.tunnel_handlers.sessions import SessionTunnelHandlers
from parlant.core.version import VERSION

__all__ = [
    "StreamHandler",
    "TunnelHandlerSet",
    "UnaryHandler",
    "create_tunnel_handler_registry",
    "merge_handler_sets",
]


async def _system_version(params: dict[str, Any]) -> dict[str, Any]:
    del params
    return {"version": VERSION}


def _make_system_health(health_reporter: HealthReporter) -> UnaryHandler:
    """Expose the reporter's snapshot verbatim.

    The shape is the same one ``/healthz`` serves, so the tunnel and the HTTP
    endpoint stay a single contract rather than two that can drift.
    """

    async def _system_health(params: dict[str, Any]) -> dict[str, Any]:
        del params
        return await health_reporter.admin_snapshot()

    return _system_health


def create_tunnel_handler_registry(
    application: Application,
    health_reporter: HealthReporter,
    logger: Logger | None = None,
    session_client_recorder: SessionClientRecorder | None = None,
) -> TunnelHandlerSet:
    del logger  # reserved for future handler-level diagnostics

    return merge_handler_sets(
        TunnelHandlerSet(
            unary={
                "system.version": _system_version,
                "system.health": _make_system_health(health_reporter),
            },
            streams={},
        ),
        SessionTunnelHandlers(application.sessions, session_client_recorder).handler_set(),
        AgentTunnelHandlers(application.agents).handler_set(),
        CustomerTunnelHandlers(application.customers).handler_set(),
        GroupTunnelHandlers(application.groups).handler_set(),
        RuleTunnelHandlers(application.rules).handler_set(),
        IndexingTunnelHandlers(application.indexing).handler_set(),
        RelationshipTunnelHandlers(application.relationships).handler_set(),
        GlossaryTunnelHandlers(application.glossary).handler_set(),
        CannedResponseTunnelHandlers(application.canned_responses).handler_set(),
        ContextVariableTunnelHandlers(
            application.variables,
            group_module=application.groups,
        ).handler_set(),
        ServiceTunnelHandlers(application.services).handler_set(),
    )
