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

"""Shared configuration and constants for the Parlant Cloud module."""

import logging
import os

_logger = logging.getLogger(__name__)

PROJECT_TOKEN_HEADER = "X-Parlant-Cloud-Project-Token"

_DEFAULT_BASE_URL = "https://api.parlant.cloud"


def _get_cloud_base_url() -> str:
    """Resolve the Parlant Cloud base URL from environment.

    Used for Parlant Cloud REST endpoints such as project-token validation.
    """
    return (os.getenv("PARLANT_CLOUD_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")


def _get_cloud_otel_url() -> str:
    """Resolve the Parlant Cloud OTLP collector base URL.

    ``PARLANT_CLOUD_CLOUD_OTEL_URL`` is the current runtime env var injected by
    Parlant Cloud. ``PARLANT_CLOUD_OTEL_URL`` remains accepted for compatibility.
    """
    return (
        os.getenv("PARLANT_CLOUD_CLOUD_OTEL_URL")
        or os.getenv("PARLANT_CLOUD_OTEL_URL")
        or _get_cloud_base_url()
    ).rstrip("/")


def _get_cloud_tunnel_url() -> str:
    """Resolve the WebSocket URL used by the Parlant Cloud tunnel."""
    configured_url = os.getenv("PARLANT_CLOUD_TUNNEL_URL")
    if configured_url:
        return _to_websocket_cloud_url(configured_url)

    return _to_websocket_cloud_url(_get_cloud_base_url())


def _to_websocket_cloud_url(url: str) -> str:
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
