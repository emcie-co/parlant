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

import hmac
import re
import secrets as _secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from typing_extensions import override
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers

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
from parlant.core.customers import CustomerKind, CustomerStore
from parlant.core.sessions import Session
from parlant.core import metadata


# Reserved session-metadata key tagging a guest-created session with the guest
# instance that owns it. Written at session creation; read by the ownership
# authorizer to scope a guest to its own sessions.
GUEST_SESSION_METADATA_KEY = "guest_instance_id"
GUEST_SESSION_METADATA_PATH = ("auth", GUEST_SESSION_METADATA_KEY)

# Response header carrying a freshly minted guest token when an anonymous
# request creates its first guest session.
GUEST_TOKEN_HEADER = "X-Parlant-Guest-Token"

_HTTP_FIELD_NAME = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+", re.ASCII)
_MAX_HTTP_FIELD_NAME_LENGTH = 128


class Operation(Enum):
    ACCESS_INTEGRATED_UI = "access_integrated_ui"
    ACCESS_API_DOCS = "access_api_docs"

    READ_AGENT = "read_agent"
    LIST_AGENTS = "list_agents"

    CREATE_CUSTOMER = "create_customer"
    READ_CUSTOMER = "read_customer"
    LIST_CUSTOMERS = "list_customers"
    UPDATE_CUSTOMER = "update_customer"
    DELETE_CUSTOMER = "delete_customer"

    CREATE_GUEST_SESSION = "create_guest_session"
    CREATE_CUSTOMER_SESSION = "create_customer_session"
    CREATE_PERSONA_SESSION = "create_persona_session"
    READ_SESSION = "read_session"
    LIST_SESSIONS = "list_sessions"
    UPDATE_SESSION = "update_session"
    DELETE_SESSION = "delete_session"
    DELETE_SESSIONS = "delete_sessions"
    CREATE_CUSTOMER_EVENT = "create_customer_event"
    CREATE_CUSTOMER_EVENT_WITH_ATTACHMENTS = "create_customer_event_with_attachments"
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
    READ_EVENT_REASONING = "read_event_reasoning"
    READ_TOOL_RESULT_DATA = "read_tool_result_data"
    READ_INTERNAL_EVENT_METADATA = "read_internal_event_metadata"
    DELETE_EVENTS = "delete_events"
    UPDATE_EVENT = "update_event"

    LIST_GROUPS = "list_groups"

    STREAM_LOGS = "stream_logs"
    READ_DIAGNOSTIC_REPORT = "read_diagnostic_report"
    READ_HEALTH_REPORT = "read_health_report"


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


class HeaderApiKeyAuthorizationPolicy(AuthorizationPolicy):
    """Grants full admin access if and only if a request carries exactly one
    exact-match API key in the configured header.

    Built for trusted machine-to-machine callers (e.g. a managing platform
    backend); it never authenticates browsers, guests, or customers.
    """

    def __init__(self, *, api_key: str, header_name: str) -> None:
        if not api_key.strip():
            raise ValueError("API key must not be blank.")

        if (
            not header_name
            or len(header_name) > _MAX_HTTP_FIELD_NAME_LENGTH
            or _HTTP_FIELD_NAME.fullmatch(header_name) is None
        ):
            raise ValueError("Invalid API credential header name.")

        self._api_key = api_key.encode("utf-8")
        self._header_name = header_name.lower()

    def _authenticate(self, headers: Headers) -> AdminPrincipal:
        credentials = headers.getlist(self._header_name)
        if len(credentials) != 1 or not hmac.compare_digest(
            credentials[0].encode("utf-8"), self._api_key
        ):
            raise AuthenticationError("Invalid or missing API credential.")

        return AdminPrincipal()

    @override
    async def check_permission(self, request: Request, operation: Operation) -> bool:
        self._authenticate(request.headers)
        return True

    @override
    async def check_rate_limit(self, request: Request, operation: Operation) -> bool:
        return True

    @override
    async def check_websocket_permission(self, websocket: WebSocket, operation: Operation) -> bool:
        self._authenticate(websocket.headers)
        return True

    @override
    async def resolve_principal(self, request: Request) -> Principal:
        return self._authenticate(request.headers)

    @override
    async def check_session_access(self, request: Request, session: Session) -> bool:
        self._authenticate(request.headers)
        return True

    @property
    @override
    def name(self) -> str:
        return "header-api-key"


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
        }
    )

    _GUEST_OPERATIONS = frozenset(
        {
            Operation.CREATE_GUEST_SESSION,
            Operation.READ_AGENT,
            Operation.READ_SESSION,
            Operation.LIST_EVENTS,
            Operation.READ_EVENT,
            Operation.CREATE_CUSTOMER_EVENT,
            Operation.CREATE_CUSTOMER_EVENT_WITH_ATTACHMENTS,
            Operation.CREATE_STATUS_EVENT,
        }
    )

    _CUSTOMER_OPERATIONS = frozenset(
        {
            Operation.CREATE_CUSTOMER_SESSION,
            Operation.READ_AGENT,
            Operation.READ_SESSION,
            Operation.LIST_SESSIONS,
            Operation.LIST_EVENTS,
            Operation.READ_EVENT,
            Operation.CREATE_CUSTOMER_EVENT,
            Operation.CREATE_CUSTOMER_EVENT_WITH_ATTACHMENTS,
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
                    and metadata.internal_value(session.metadata, GUEST_SESSION_METADATA_PATH)
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
        trusted_proxies: frozenset[str] = frozenset(),
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
            },
            trusted_proxies=trusted_proxies,
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
            await self._resolve_customer(principal)

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

    async def _resolve_customer(self, principal: CustomerPrincipal) -> None:
        """Checks that the token's subject names a customer anyone may be served as.

        A persona is not such a customer. The subject is read as a customer id
        and whatever is found under it is accepted, so a persona sharing an id
        with a real person's token would quietly become that person's
        identity: their session would attach to the persona, and nothing would
        report it. Refusing the token here is what confines a persona to
        admin-authenticated callers, and it covers every endpoint at once
        because they all resolve their principal through this method.
        """
        if self._customer_store is None:
            return

        try:
            customer = await self._customer_store.read_customer(principal.customer_id)
        except ItemNotFoundError:
            if not self._auto_provision_customers:
                raise AuthenticationError(
                    f"Unknown customer '{principal.customer_id}' (auto-provisioning is disabled)"
                )

            name = principal.attributes.get("name")

            # Provisioning leaves the kind at its default. A customer that
            # exists because somebody authenticated as them is a real one.
            await self._customer_store.create_customer(
                name=name if isinstance(name, str) and name else str(principal.customer_id),
                id=principal.customer_id,
            )

            return

        if customer.kind is CustomerKind.PERSONA:
            raise AuthenticationError(
                f"Customer '{principal.customer_id}' is a persona and cannot be authenticated"
            )


@dataclass(frozen=True)
class AuthConfig:
    """Declarative configuration for the OSS composite authorization policy.

    ``customer_jwt_secret`` and ``customer_jwks_url`` are mutually exclusive;
    leave both unset to disable customer authentication entirely (guests and
    admin only). If ``guest_token_secret`` is unset, an ephemeral secret is
    generated at startup — guest tokens will then not survive a restart.

    ``customer_jwt_audience`` and ``customer_jwt_issuer``, when set, are enforced
    on customer tokens (``aud``/``iss``) so a token minted for a different
    audience or issuer of the same signing authority is rejected.
    ``require_customer_jwt_expiry`` additionally rejects customer tokens without
    an ``exp`` claim; it defaults to ``False`` to preserve compatibility with
    integrators minting non-expiring tokens.
    """

    admin_api_key: str
    customer_jwt_secret: str | None = None
    customer_jwks_url: str | None = None
    customer_jwt_audience: str | None = None
    customer_jwt_issuer: str | None = None
    require_customer_jwt_expiry: bool = False
    guest_token_secret: str | None = None
    auto_provision_customers: bool = False
    trusted_proxies: frozenset[str] = frozenset()


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
            customer_jwt_audience=config.customer_jwt_audience,
            customer_jwt_issuer=config.customer_jwt_issuer,
            require_customer_jwt_expiry=config.require_customer_jwt_expiry,
        ),
        customer_store=customer_store,
        auto_provision_customers=config.auto_provision_customers,
        trusted_proxies=config.trusted_proxies,
    )


class BasicRateLimiter(RateLimiter):
    def __init__(
        self,
        rate_limit_item_per_operation: dict[Operation, RateLimitItem],
        storage: MemoryStorage | None = None,
        limiter_type: type[
            MovingWindowRateLimiter | FixedWindowRateLimiter | SlidingWindowCounterRateLimiter
        ] = MovingWindowRateLimiter,
        trusted_proxies: frozenset[str] = frozenset(),
    ) -> None:
        self.rate_limit_item_per_operation = rate_limit_item_per_operation
        self._limiter = limiter_type(storage or MemoryStorage())
        self._default_rate_limit_item = RateLimitItemPerMinute(100)
        # Forwarding headers (X-Forwarded-For, X-Real-IP, CF-Connecting-IP) are
        # only honored when the immediate peer is one of these addresses.
        # Empty (the default) means never trust them: a direct client cannot
        # spoof its identity to reset its own rate-limit bucket.
        self._trusted_proxies = trusted_proxies

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

    def _get_client_ip(self, request: Request) -> str | None:
        peer = request.client.host if request.client else None

        # Only a request arriving from a configured trusted proxy may name its
        # client through a forwarding header; otherwise the header is
        # attacker-controlled and must be ignored.
        if peer is not None and peer in self._trusted_proxies:
            headers = request.headers

            if xff := headers.get("x-forwarded-for"):
                return xff.split(",")[0].strip()

            if xri := headers.get("x-real-ip"):
                return xri.strip()

            if cf := headers.get("cf-connecting-ip"):
                return cf.strip()

        return peer
