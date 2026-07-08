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

from dataclasses import dataclass
from typing import Mapping

from parlant.adapters.modules.parlant_cloud.auth import ParlantCloudAuthorizationPolicy
from parlant.adapters.modules.parlant_cloud.config import PROJECT_TOKEN_HEADER
from parlant.api.authorization import Operation


@dataclass
class FakeWebSocket:
    headers: Mapping[str, str]


@dataclass
class FakeRequest:
    headers: Mapping[str, str]


async def test_that_localhost_is_trusted_for_api_operations_in_cloud_auth_mode() -> None:
    policy = ParlantCloudAuthorizationPolicy(project_token="project-token")
    request = FakeRequest(headers={"host": "localhost:2222"})

    result = await policy.check_permission(request, Operation.LIST_AGENTS)  # type: ignore[arg-type]

    assert result is True


async def test_that_localhost_can_stream_logs_in_cloud_auth_mode() -> None:
    policy = ParlantCloudAuthorizationPolicy(project_token="project-token")
    websocket = FakeWebSocket(headers={"host": "localhost:2222"})

    result = await policy.check_websocket_permission(websocket, Operation.STREAM_LOGS)  # type: ignore[arg-type]

    assert result is True


async def test_that_127_0_0_1_can_stream_logs_in_cloud_auth_mode() -> None:
    policy = ParlantCloudAuthorizationPolicy(project_token="project-token")
    websocket = FakeWebSocket(headers={"host": "127.0.0.1:2222"})

    result = await policy.check_websocket_permission(websocket, Operation.STREAM_LOGS)  # type: ignore[arg-type]

    assert result is True


async def test_that_non_localhost_cannot_stream_logs_without_project_token() -> None:
    policy = ParlantCloudAuthorizationPolicy(project_token="project-token")
    websocket = FakeWebSocket(headers={"host": "example.com"})

    result = await policy.check_websocket_permission(websocket, Operation.STREAM_LOGS)  # type: ignore[arg-type]

    assert result is False


async def test_that_project_token_can_stream_logs_in_cloud_auth_mode() -> None:
    policy = ParlantCloudAuthorizationPolicy(project_token="project-token")
    websocket = FakeWebSocket(
        headers={
            "host": "example.com",
            PROJECT_TOKEN_HEADER: "project-token",
        }
    )

    result = await policy.check_websocket_permission(websocket, Operation.STREAM_LOGS)  # type: ignore[arg-type]

    assert result is True
