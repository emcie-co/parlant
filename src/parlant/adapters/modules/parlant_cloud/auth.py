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

"""Authorization policy for the Parlant Cloud module."""

import hmac
from typing import Mapping

from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from typing_extensions import override

from limits import RateLimitItemPerMinute

from parlant.api.authorization import AuthorizationPolicy, BasicRateLimiter, Operation

from parlant.adapters.modules.parlant_cloud.config import PROJECT_TOKEN_HEADER


# Operations an untrusted (tokenless, non-localhost) caller may still perform,
# with their rate limits. Mirrors the legacy production allowlist.
_UNTRUSTED_OPERATIONS = frozenset(
    {
        Operation.READ_AGENT,
        Operation.CREATE_GUEST_SESSION,
        Operation.READ_SESSION,
        Operation.LIST_EVENTS,
        Operation.CREATE_CUSTOMER_EVENT,
    }
)


class ParlantCloudAuthorizationPolicy(AuthorizationPolicy):
    def __init__(self, project_token: str) -> None:
        self._project_token = project_token
        self._untrusted_rate_limiter = BasicRateLimiter(
            rate_limit_item_per_operation={
                Operation.READ_AGENT: RateLimitItemPerMinute(30),
                Operation.CREATE_GUEST_SESSION: RateLimitItemPerMinute(10),
                Operation.READ_SESSION: RateLimitItemPerMinute(30),
                Operation.LIST_EVENTS: RateLimitItemPerMinute(240),
                Operation.CREATE_CUSTOMER_EVENT: RateLimitItemPerMinute(30),
                Operation.CREATE_STATUS_EVENT: RateLimitItemPerMinute(60),
            }
        )

    @property
    @override
    def name(self) -> str:
        return "parlant-cloud"

    @override
    async def configure_app(self, app: FastAPI) -> FastAPI:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        return app

    def _is_trusted(self, headers: Mapping[str, str]) -> bool:
        if self._is_localhost(headers):
            return True

        token = headers.get(PROJECT_TOKEN_HEADER, "")
        return bool(token) and hmac.compare_digest(token, self._project_token)

    def _is_localhost(self, headers: Mapping[str, str]) -> bool:
        host = headers.get("host", "")
        return host.startswith("localhost") or host.startswith("127.0.0.1")

    @override
    async def check_permission(self, request: Request, operation: Operation) -> bool:
        if self._is_trusted(request.headers):
            return True
        return operation in _UNTRUSTED_OPERATIONS

    @override
    async def check_rate_limit(self, request: Request, operation: Operation) -> bool:
        if self._is_trusted(request.headers):
            return True
        return await self._untrusted_rate_limiter.check(request, operation)

    @override
    async def check_websocket_permission(
        self,
        websocket: WebSocket,
        operation: Operation,
    ) -> bool:
        return self._is_trusted(websocket.headers)
