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

"""Parlant Dashboard module.

Auto-loaded by the Server when PARLANT_DASHBOARD_PROJECT_TOKEN is set.
Validates the project token, resolves project context, and sets up
ParlantDashboardTracer / ParlantDashboardLogger.

Tunnel URL:
  PARLANT_DASHBOARD_TUNNEL_URL, or PARLANT_DASHBOARD_BASE_URL converted to WebSocket /cloud.

Logs and traces collector URL:
  PARLANT_DASHBOARD_OTEL_URL, then PARLANT_DASHBOARD_BASE_URL.
"""

from parlant.adapters.modules.parlant_dashboard.lifecycle import (
    configure_container,
    initialize_container,
)
from parlant.adapters.modules.parlant_dashboard.tunnel import (
    ParlantDashboardTunnelService,
    create_tunnel_service,
)

__all__ = [
    "ParlantDashboardTunnelService",
    "create_tunnel_service",
    "configure_container",
    "initialize_container",
]
