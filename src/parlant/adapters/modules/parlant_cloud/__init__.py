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

Tunnel URL:
  PARLANT_CLOUD_TUNNEL_URL, or PARLANT_CLOUD_BASE_URL converted to WebSocket /cloud.

Logs, traces, and metrics collector URL:
  PARLANT_CLOUD_OTEL_URL, then PARLANT_CLOUD_BASE_URL.

PARLANT_CLOUD_API_KEY and PARLANT_CLOUD_API_URL are used only by the NLP
service adapter.
"""

from parlant.adapters.modules.parlant_cloud.lifecycle import (
    configure_container,
    initialize_container,
)
from parlant.adapters.modules.parlant_cloud.tunnel import (
    ParlantCloudTunnelService,
    _create_tunnel_service,
)

__all__ = [
    "ParlantCloudTunnelService",
    "_create_tunnel_service",
    "configure_container",
    "initialize_container",
]
