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

"""Authorization policy for the Parlant Dashboard module.

``ParlantDashboardAuthorizationPolicy`` is a project-token trust wrapper: a
request carrying the correct ``X-Parlant-Project-Token`` header passes every
check with admin semantics (this is how the Dashboard platform manages the
server), and every other request delegates to an inner policy.

The default inner policy is the OSS composite policy — real guest tokens,
customer JWTs, session ownership, and rate limits — configured from the same
``PARLANT_*`` environment variables the production environment uses, except
that ``PARLANT_ADMIN_API_KEY`` is optional here: under Dashboard mode, admin
access legitimately arrives through the project token, so an unset admin key
simply means the project token is the only admin credential.
"""

import hmac
import os
import secrets
from typing import Mapping

from fastapi import FastAPI, Request, WebSocket
from typing_extensions import override

from parlant.api.authentication import AdminPrincipal, Principal, TokenAuthenticator
from parlant.api.authorization import (
    AuthorizationPolicy,
    CompositeAuthorizationPolicy,
    Operation,
)
from parlant.core.customers import CustomerStore
from parlant.core.sessions import Session

from parlant.adapters.modules.parlant_dashboard.config import PROJECT_TOKEN_HEADER


def build_default_inner_policy(
    customer_store: CustomerStore | None,
) -> CompositeAuthorizationPolicy:
    """The composite guest/customer policy for tokenless Dashboard callers.

    Reads the standard auth environment variables (customer JWT secret or JWKS
    URL, guest token secret, auto-provisioning); generates ephemeral secrets
    where none are configured — guest tokens then die with the process.
    """
    customer_jwt_secret = os.environ.get("PARLANT_CUSTOMER_JWT_SECRET")
    customer_jwks_url = os.environ.get("PARLANT_CUSTOMER_JWKS_URL")

    if customer_jwt_secret and customer_jwks_url:
        raise ValueError(
            "PARLANT_CUSTOMER_JWT_SECRET and PARLANT_CUSTOMER_JWKS_URL are mutually "
            "exclusive; set exactly one customer token verification method."
        )

    return CompositeAuthorizationPolicy(
        authenticator=TokenAuthenticator(
            guest_token_secret=os.environ.get("PARLANT_GUEST_TOKEN_SECRET")
            or secrets.token_hex(32),
            admin_api_key=os.environ.get("PARLANT_ADMIN_API_KEY") or secrets.token_hex(32),
            customer_jwt_secret=customer_jwt_secret,
            customer_jwks_url=customer_jwks_url,
            customer_jwt_audience=os.environ.get("PARLANT_CUSTOMER_JWT_AUDIENCE"),
            customer_jwt_issuer=os.environ.get("PARLANT_CUSTOMER_JWT_ISSUER"),
            require_customer_jwt_expiry=os.environ.get(
                "PARLANT_CUSTOMER_JWT_REQUIRE_EXPIRY", ""
            ).lower()
            in ("1", "true", "yes"),
        ),
        customer_store=customer_store,
        auto_provision_customers=os.environ.get("PARLANT_AUTH_AUTO_PROVISION_CUSTOMERS", "").lower()
        in ("1", "true", "yes"),
        trusted_proxies=frozenset(
            part.strip()
            for part in os.environ.get("PARLANT_TRUSTED_PROXIES", "").split(",")
            if part.strip()
        ),
    )


class ParlantDashboardAuthorizationPolicy(AuthorizationPolicy):
    def __init__(self, *, project_token: str, inner: AuthorizationPolicy) -> None:
        self._project_token = project_token
        self._inner = inner

    @property
    @override
    def name(self) -> str:
        return f"parlant-dashboard({self._inner.name})"

    @property
    def inner(self) -> AuthorizationPolicy:
        return self._inner

    @override
    async def configure_app(self, app: FastAPI) -> FastAPI:
        return await self._inner.configure_app(app)

    def _is_trusted(self, headers: Mapping[str, str]) -> bool:
        token = headers.get(PROJECT_TOKEN_HEADER, "")
        return bool(token) and hmac.compare_digest(token, self._project_token)

    @override
    async def check_permission(self, request: Request, operation: Operation) -> bool:
        if self._is_trusted(request.headers):
            return True
        return await self._inner.check_permission(request, operation)

    @override
    async def check_rate_limit(self, request: Request, operation: Operation) -> bool:
        if self._is_trusted(request.headers):
            return True
        return await self._inner.check_rate_limit(request, operation)

    @override
    async def check_websocket_permission(
        self,
        websocket: WebSocket,
        operation: Operation,
    ) -> bool:
        if self._is_trusted(websocket.headers):
            return True
        return await self._inner.check_websocket_permission(websocket, operation)

    @override
    async def resolve_principal(self, request: Request) -> Principal | None:
        if self._is_trusted(request.headers):
            return AdminPrincipal()
        return await self._inner.resolve_principal(request)

    @override
    async def check_session_access(self, request: Request, session: Session) -> bool:
        if self._is_trusted(request.headers):
            return True
        return await self._inner.check_session_access(request, session)

    @override
    def issue_guest_token(self, guest_instance_id: str) -> str | None:
        return self._inner.issue_guest_token(guest_instance_id)
