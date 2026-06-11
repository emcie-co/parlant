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

from parlant.api.authorization import AuthorizationPolicy, Operation, ProductionAuthorizationPolicy

from parlant.adapters.modules.parlant_cloud.config import PROJECT_TOKEN_HEADER


class ParlantCloudAuthorizationPolicy(AuthorizationPolicy):
    def __init__(self, project_token: str) -> None:
        self._project_token = project_token
        self._production_policy = ProductionAuthorizationPolicy()

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
        return await self._production_policy.check_permission(request, operation)

    @override
    async def check_rate_limit(self, request: Request, operation: Operation) -> bool:
        if self._is_trusted(request.headers):
            return True
        return await self._production_policy.check_rate_limit(request, operation)

    @override
    async def check_websocket_permission(
        self,
        websocket: WebSocket,
        operation: Operation,
    ) -> bool:
        if self._is_trusted(websocket.headers):
            return True
        return await self._production_policy.check_websocket_permission(websocket, operation)
