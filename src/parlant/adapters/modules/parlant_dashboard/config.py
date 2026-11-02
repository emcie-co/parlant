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

"""Shared configuration and constants for the Parlant Dashboard module."""

import os


PROJECT_TOKEN_HEADER = "X-Parlant-Project-Token"

# The dashboard's API, which also serves the tunnel (/cloud) and the collector
# (/v1/traces, /v1/logs). dashboard.parlant.io is the web app: its CDN answers
# API paths with the page.
_DEFAULT_BASE_URL = "https://api.parlant.io"


def _get_dashboard_base_url() -> str:
    """Resolve the Parlant Dashboard base URL from environment.

    Used for Parlant Dashboard REST endpoints such as project-token validation.
    """
    return (os.getenv("PARLANT_DASHBOARD_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")


def _get_dashboard_otel_url() -> str:
    """Resolve the Parlant Dashboard OTLP collector base URL.

    ``PARLANT_DASHBOARD_OTEL_URL`` is injected by Parlant Dashboard when observability
    should be sent to a collector other than the API base URL.
    """
    return (os.getenv("PARLANT_DASHBOARD_OTEL_URL") or _get_dashboard_base_url()).rstrip("/")


def _get_dashboard_tunnel_url() -> str:
    """Resolve the WebSocket URL used by the Parlant Dashboard tunnel."""
    return _to_websocket_tunnel_url(
        os.getenv("PARLANT_DASHBOARD_TUNNEL_URL", _get_dashboard_base_url()).rstrip("/")
    )


def _to_websocket_tunnel_url(url: str) -> str:
    normalized_url = url.rstrip("/")

    if normalized_url.startswith("https://"):
        normalized_url = "wss://" + normalized_url.removeprefix("https://")
    elif normalized_url.startswith("http://"):
        normalized_url = "ws://" + normalized_url.removeprefix("http://")
    elif not normalized_url.startswith(("ws://", "wss://")):
        normalized_url = f"ws://{normalized_url}"

    if normalized_url.endswith("/cloud"):
        return normalized_url

    return f"{normalized_url}/cloud"
