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

import secrets as _secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from typing_extensions import override
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from limits.storage import MemoryStorage
from limits.strategies import (
    MovingWindowRateLimiter,
    FixedWindowRateLimiter,
    SlidingWindowCounterRateLimiter,
)
from limits import RateLimitItem, RateLimitItemPerMinute

from parlant.api.authentication import (
    AdminPrincipal,
    Authenticator,
    AuthenticationError,
    CustomerPrincipal,
    GuestPrincipal,
    Principal,
    TokenAuthenticator,
)
from parlant.core.common import ItemNotFoundError
from parlant.core.customers import CustomerStore
from parlant.core.sessions import Session


# Reserved session-metadata key tagging a guest-created session with the guest
# instance that owns it. Written at session creation; read by the ownership
# authorizer to scope a guest to its own sessions.
GUEST_SESSION_METADATA_KEY = "__guest_instance_id__"

# Response header carrying a freshly minted guest token when an anonymous
# request creates its first guest session.
GUEST_TOKEN_HEADER = "X-Parlant-Guest-Token"


class Operation(Enum):
    ACCESS_INTEGRATED_UI = "access_integrated_ui"
    ACCESS_API_DOCS = "access_api_docs"

    CREATE_AGENT = "create_agent"
    READ_AGENT = "read_agent"
    READ_AGENT_DESCRIPTION = "read_agent_description"
    LIST_AGENTS = "list_agents"
    UPDATE_AGENT = "update_agent"
    DELETE_AGENT = "delete_agent"

    CREATE_CANNED_RESPONSE = "create_canned_response"
    READ_CANNED_RESPONSE = "read_canned_response"
    LIST_CANNED_RESPONSES = "list_canned_responses"
    UPDATE_CANNED_RESPONSE = "update_canned_response"
    DELETE_CANNED_RESPONSE = "delete_canned_response"

    CREATE_CAPABILITY = "create_capability"
    READ_CAPABILITY = "read_capability"
    LIST_CAPABILITIES = "list_capabilities"
    UPDATE_CAPABILITY = "update_capability"
    DELETE_CAPABILITY = "delete_capability"

    CREATE_CONTEXT_VARIABLE = "create_context_variable"
    READ_CONTEXT_VARIABLE = "read_context_variable"
    LIST_CONTEXT_VARIABLES = "list_context_variables"
    UPDATE_CONTEXT_VARIABLE = "update_context_variable"
    DELETE_CONTEXT_VARIABLE = "delete_context_variable"
    DELETE_CONTEXT_VARIABLES = "delete_context_variables"
    READ_CONTEXT_VARIABLE_VALUE = "read_context_variable_value"
    UPDATE_CONTEXT_VARIABLE_VALUE = "update_context_variable_value"
    DELETE_CONTEXT_VARIABLE_VALUE = "delete_context_variable_value"

    CREATE_CUSTOMER = "create_customer"
    READ_CUSTOMER = "read_customer"
    LIST_CUSTOMERS = "list_customers"
    UPDATE_CUSTOMER = "update_customer"
    DELETE_CUSTOMER = "delete_customer"

    CREATE_EVALUATION = "create_evaluation"
    READ_EVALUATION = "read_evaluation"

    CREATE_TRAINING = "create_training"
    READ_TRAINING = "read_training"

    CREATE_TERM = "create_term"
    READ_TERM = "read_term"
    LIST_TERMS = "list_terms"
    UPDATE_TERM = "update_term"
    DELETE_TERM = "delete_term"

    CREATE_RULE = "create_rule"
    READ_RULE = "read_rule"
    LIST_RULES = "list_rules"
    UPDATE_RULE = "update_rule"
    DELETE_RULE = "delete_rule"

    CREATE_JOURNEY = "create_journey"
    READ_JOURNEY = "read_journey"
    LIST_JOURNEYS = "list_journeys"
    UPDATE_JOURNEY = "update_journey"
    DELETE_JOURNEY = "delete_journey"

    CREATE_RELATIONSHIP = "create_relationship"
    READ_RELATIONSHIP = "read_relationship"
    LIST_RELATIONSHIPS = "list_relationships"
    DELETE_RELATIONSHIP = "delete_relationship"

    UPDATE_SERVICE = "update_service"
    READ_SERVICE = "read_service"
    LIST_SERVICES = "list_services"
    DELETE_SERVICE = "delete_service"

    CREATE_GUEST_SESSION = "create_guest_session"
    CREATE_CUSTOMER_SESSION = "create_customer_session"
    READ_SESSION = "read_session"
    LIST_SESSIONS = "list_sessions"
    UPDATE_SESSION = "update_session"
    DELETE_SESSION = "delete_session"
    DELETE_SESSIONS = "delete_sessions"
    CREATE_CUSTOMER_EVENT = "create_customer_event"
    CREATE_AGENT_EVENT = "create_agent_event"
    CREATE_HUMAN_AGENT_EVENT = "create_human_agent_event"
    CREATE_HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT_EVENT = (
        "create_human_agent_on_behalf_of_ai_agent_event"
    )
    OVERRIDE_CUSTOMER_PARTICIPANT = "override_customer_participant"
    CREATE_STATUS_EVENT = "create_status_event"
    CREATE_CUSTOM_EVENT = "create_custom_event"
    LIST_EVENTS = "list_events"
    READ_EVENT = "read_event"
    DELETE_EVENTS = "delete_events"
    UPDATE_EVENT = "update_event"

    CREATE_GROUP = "create_group"
    READ_GROUP = "read_group"
    LIST_GROUPS = "list_groups"
    UPDATE_GROUP = "update_group"
    DELETE_GROUP = "delete_group"

    STREAM_LOGS = "stream_logs"


class AuthorizationException(Exception):
    def __init__(
        self,
        request: Request | WebSocket,
        operation: Operation | None,
        message_prefix: str = "Authorization failed",
    ) -> None:
        super().__init__(
            f"{message_prefix}: OPERATION={operation.value if operation else 'GENERIC'}, HEADERS={_safe_headers(request.headers)}"
        )

        self.request = request
        self.operation = operation


def _safe_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {
        name: "<redacted>" if _is_sensitive_header(name) else value
        for name, value in headers.items()
    }


def _is_sensitive_header(name: str) -> bool:
    lower_name = name.lower()
    return (
        lower_name in {"authorization", "cookie", "set-cookie"}
        or "token" in lower_name
        or "secret" in lower_name
        or "key" in lower_name
    )


class RateLimitExceededException(AuthorizationException):
    def __init__(self, request: Request, operation: Operation | None) -> None:
        super().__init__(
            request=request,
            operation=operation,
            message_prefix="Rate limit exceeded",
        )


class AuthorizationPolicy(ABC):
    async def configure_app(self, app: FastAPI) -> FastAPI:
        return app

    @abstractmethod
    async def check_permission(self, request: Request, operation: Operation) -> bool: ...

    @abstractmethod
    async def check_rate_limit(self, request: Request, operation: Operation) -> bool: ...

    async def authorize(self, request: Request, operation: Operation) -> None:
        if not await self.check_permission(request, operation):
            raise AuthorizationException(request, operation)

        if not await self.check_rate_limit(request, operation):
            raise RateLimitExceededException(request, operation)

    @abstractmethod
    async def check_websocket_permission(
        self, websocket: WebSocket, operation: Operation
    ) -> bool: ...

    async def authorize_websocket(self, websocket: WebSocket, operation: Operation) -> None:
        if not await self.check_websocket_permission(websocket, operation):
            raise AuthorizationException(websocket, operation)

    async def resolve_principal(self, request: Request) -> Principal | None:
        """The authenticated principal behind this request, or None if this
        policy does not authenticate callers (e.g. development mode)."""
        return None

    async def check_session_access(self, request: Request, session: Session) -> bool:
        """Whether this request's caller may access the given session.

        By default there are no resource-level restrictions; principal-aware
        policies override this with an ownership check.
        """
        return True

    async def authorize_session_access(self, request: Request, session: Session) -> None:
        if not await self.check_session_access(request, session):
            raise AuthorizationException(
                request,
                None,
                message_prefix=f"Access to session '{session.id}' denied",
            )

    def issue_guest_token(self, guest_instance_id: str) -> str | None:
        """Mints a guest credential tied to the given guest instance, if this
        policy authenticates guests. Returns None otherwise."""
        return None

    @property
    @abstractmethod
    def name(self) -> str: ...


class DevelopmentAuthorizationPolicy(AuthorizationPolicy):
    async def configure_app(self, app: FastAPI) -> FastAPI:
        # Allow all origins in development
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        return app

    @override
    async def check_rate_limit(self, request: Request, operation: Operation) -> bool:
        # In development, we do not enforce rate limits
        return True

    @override
    async def check_permission(self, request: Request, operation: Operation) -> bool:
        # In development, we allow all actions
        return True

    @override
    async def check_websocket_permission(self, websocket: WebSocket, operation: Operation) -> bool:
        # In development, we allow all websocket actions
        return True

    @property
    @override
    def name(self) -> str:
        return "development"


class RateLimiter(ABC):
    @abstractmethod
    async def check(
        self,
        request: Request,
        operation: Operation,
    ) -> bool: ...


class PermissionGate(ABC):
    """Port: the coarse, per-principal-kind operation allowlist.

    Decides whether this KIND of caller may attempt an operation at all.
    Resource-level restrictions (ownership) are the Authorizer's job.
    """

    @abstractmethod
    async def check(self, principal: Principal, operation: Operation) -> bool: ...


class DefaultPermissionGate(PermissionGate):
    """OSS default gate: anonymous callers may only start a guest conversation;
    guests and customers get conversation-scoped operations; admins get all."""

    _ANONYMOUS_OPERATIONS = frozenset(
        {
            Operation.CREATE_GUEST_SESSION,
            Operation.READ_AGENT,
            Operation.READ_AGENT_DESCRIPTION,
        }
    )

    _GUEST_OPERATIONS = frozenset(
        {
            Operation.CREATE_GUEST_SESSION,
            Operation.READ_AGENT,
            Operation.READ_AGENT_DESCRIPTION,
            Operation.READ_SESSION,
            Operation.LIST_EVENTS,
            Operation.READ_EVENT,
            Operation.CREATE_CUSTOMER_EVENT,
            Operation.CREATE_STATUS_EVENT,
        }
    )

    _CUSTOMER_OPERATIONS = frozenset(
        {
            Operation.CREATE_CUSTOMER_SESSION,
            Operation.READ_AGENT,
            Operation.READ_AGENT_DESCRIPTION,
            Operation.READ_SESSION,
            Operation.LIST_SESSIONS,
            Operation.LIST_EVENTS,
            Operation.READ_EVENT,
            Operation.CREATE_CUSTOMER_EVENT,
            Operation.CREATE_STATUS_EVENT,
        }
    )

    @override
    async def check(self, principal: Principal, operation: Operation) -> bool:
        match principal:
            case AdminPrincipal():
                return True
            case CustomerPrincipal():
                return operation in self._CUSTOMER_OPERATIONS
            case GuestPrincipal():
                return operation in self._GUEST_OPERATIONS
            case _:
                return operation in self._ANONYMOUS_OPERATIONS


class Authorizer(ABC):
    """Port: fine-grained, per-resource access decisions.

    Invoked by handlers that already hold the loaded resource, after the
    permission gate has allowed the operation kind.
    """

    @abstractmethod
    async def check_session_access(self, principal: Principal, session: Session) -> bool: ...


class OwnershipAuthorizer(Authorizer):
    """OSS default authorizer: a session is accessible to the guest instance or
    customer that owns it, and to admins."""

    @override
    async def check_session_access(self, principal: Principal, session: Session) -> bool:
        match principal:
            case AdminPrincipal():
                return True
            case CustomerPrincipal():
                return (
                    session.customer_id == principal.customer_id
                    and session.customer_id != CustomerStore.GUEST_ID
                )
            case GuestPrincipal():
                return (
                    session.customer_id == CustomerStore.GUEST_ID
                    and session.metadata.get(GUEST_SESSION_METADATA_KEY)
                    == principal.guest_instance_id
                )
            case _:
                return False


class CompositeAuthorizationPolicy(AuthorizationPolicy):
    """The OSS production policy, composed of swappable ports.

    Pipeline per request: the ``Authenticator`` resolves the credentials into a
    ``Principal`` (cached on the request); the ``PermissionGate`` applies the
    per-kind operation allowlist; the ``RateLimiter`` throttles. Fine-grained
    session ownership is exposed via ``authorize_session_access`` for the
    handlers that hold the loaded session.

    Enterprise deployments customize by swapping any port (e.g. an SSO
    authenticator, an RBAC gate, a tenant-aware authorizer) while reusing the
    rest.
    """

    def __init__(
        self,
        *,
        authenticator: Authenticator,
        permission_gate: PermissionGate | None = None,
        authorizer: Authorizer | None = None,
        rate_limiter: RateLimiter | None = None,
        customer_store: CustomerStore | None = None,
        auto_provision_customers: bool = False,
    ) -> None:
        self.authenticator = authenticator
        self.permission_gate = permission_gate or DefaultPermissionGate()
        self.authorizer = authorizer or OwnershipAuthorizer()
        self.rate_limiter = rate_limiter or BasicRateLimiter(
            rate_limit_item_per_operation={
                Operation.READ_AGENT: RateLimitItemPerMinute(30),
                Operation.CREATE_GUEST_SESSION: RateLimitItemPerMinute(10),
                Operation.READ_SESSION: RateLimitItemPerMinute(30),
                Operation.LIST_EVENTS: RateLimitItemPerMinute(240),
                Operation.CREATE_CUSTOMER_EVENT: RateLimitItemPerMinute(30),
                Operation.CREATE_STATUS_EVENT: RateLimitItemPerMinute(60),
            }
        )

        self._customer_store = customer_store
        self._auto_provision_customers = auto_provision_customers

    async def configure_app(self, app: FastAPI) -> FastAPI:
        # By default, allow all origins. It's recommended to override this
        # method in a subclass to restrict CORS to the origins (site URLs) from
        # which your application is actually served (e.g. https://your-site.com).
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            expose_headers=[GUEST_TOKEN_HEADER],
        )

        return app

    @property
    @override
    def name(self) -> str:
        return "composite"

    @override
    async def resolve_principal(self, request: Request) -> Principal:
        cached: Principal | None = getattr(request.state, "parlant_principal", None)

        if cached is not None:
            return cached

        principal = await self.authenticator.resolve(request)

        if isinstance(principal, CustomerPrincipal):
            await self._ensure_customer_exists(principal)

        request.state.parlant_principal = principal

        return principal

    @override
    async def check_permission(self, request: Request, operation: Operation) -> bool:
        principal = await self.resolve_principal(request)

        return await self.permission_gate.check(principal, operation)

    @override
    async def check_rate_limit(self, request: Request, operation: Operation) -> bool:
        if isinstance(await self.resolve_principal(request), AdminPrincipal):
            return True

        return await self.rate_limiter.check(request, operation)

    @override
    async def check_websocket_permission(self, websocket: WebSocket, operation: Operation) -> bool:
        principal = await self.authenticator.resolve_websocket(websocket)

        return isinstance(principal, AdminPrincipal)

    @override
    async def check_session_access(self, request: Request, session: Session) -> bool:
        principal = await self.resolve_principal(request)

        return await self.authorizer.check_session_access(principal, session)

    @override
    def issue_guest_token(self, guest_instance_id: str) -> str | None:
        return self.authenticator.issue_guest_token(guest_instance_id)

    async def _ensure_customer_exists(self, principal: CustomerPrincipal) -> None:
        if self._customer_store is None:
            return

        try:
            await self._customer_store.read_customer(principal.customer_id)
        except ItemNotFoundError:
            if not self._auto_provision_customers:
                raise AuthenticationError(
                    f"Unknown customer '{principal.customer_id}' (auto-provisioning is disabled)"
                )

            name = principal.attributes.get("name")

            await self._customer_store.create_customer(
                name=name if isinstance(name, str) and name else str(principal.customer_id),
                id=principal.customer_id,
            )


@dataclass(frozen=True)
class AuthConfig:
    """Declarative configuration for the OSS composite authorization policy.

    ``customer_jwt_secret`` and ``customer_jwks_url`` are mutually exclusive;
    leave both unset to disable customer authentication entirely (guests and
    admin only). If ``guest_token_secret`` is unset, an ephemeral secret is
    generated at startup — guest tokens will then not survive a restart.
    """

    admin_api_key: str
    customer_jwt_secret: str | None = None
    customer_jwks_url: str | None = None
    guest_token_secret: str | None = None
    auto_provision_customers: bool = False


def create_composite_authorization_policy(
    config: AuthConfig,
    customer_store: CustomerStore | None = None,
) -> CompositeAuthorizationPolicy:
    """Builds the OSS composite policy from a declarative config."""
    if config.customer_jwt_secret and config.customer_jwks_url:
        raise ValueError(
            "customer_jwt_secret and customer_jwks_url are mutually exclusive; "
            "configure exactly one customer token verification method"
        )

    return CompositeAuthorizationPolicy(
        authenticator=TokenAuthenticator(
            guest_token_secret=config.guest_token_secret or _secrets.token_hex(32),
            admin_api_key=config.admin_api_key,
            customer_jwt_secret=config.customer_jwt_secret,
            customer_jwks_url=config.customer_jwks_url,
        ),
        customer_store=customer_store,
        auto_provision_customers=config.auto_provision_customers,
    )


class BasicRateLimiter(RateLimiter):
    def __init__(
        self,
        rate_limit_item_per_operation: dict[Operation, RateLimitItem],
        storage: MemoryStorage | None = None,
        limiter_type: type[
            MovingWindowRateLimiter | FixedWindowRateLimiter | SlidingWindowCounterRateLimiter
        ] = MovingWindowRateLimiter,
    ) -> None:
        self.rate_limit_item_per_operation = rate_limit_item_per_operation
        self._limiter = limiter_type(storage or MemoryStorage())
        self._default_rate_limit_item = RateLimitItemPerMinute(100)

    async def check(
        self,
        request: Request,
        operation: Operation,
    ) -> bool:
        if item := self.rate_limit_item_per_operation.get(operation):
            return self._limiter.hit(item, self._build_key(request, operation))

        return self._limiter.hit(self._default_rate_limit_item, self._build_key(request, None))

    def _build_key(
        self,
        request: Request,
        operation: Operation | None,
    ) -> str:
        ip = self._get_client_ip(request)

        if not ip:
            raise AuthorizationException(
                request=request,
                operation=operation,
                message_prefix="Authorization failed: No client IP found",
            )

        return f"IP={ip}--OP={operation.value if operation else 'GENERIC'}"

    @staticmethod
    def _get_client_ip(request: Request) -> str | None:
        headers = request.headers

        if xff := headers.get("x-forwarded-for"):
            return xff.split(",")[0].strip()

        if xri := headers.get("x-real-ip"):
            return xri.strip()

        if cf := headers.get("cf-connecting-ip"):
            return cf.strip()

        return request.client.host if request.client else None
