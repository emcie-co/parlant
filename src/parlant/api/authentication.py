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

"""Authentication: resolving a request's credentials into a Principal.

This module is the "who is calling" half of the API's auth layer (the "what may
they do" half lives in ``authorization.py``). A ``Principal`` is resolved once
per request by an ``Authenticator`` and then consulted by the authorization
policy for both coarse (per-operation) and fine-grained (per-resource) checks.

The OSS default, ``TokenAuthenticator``, is fully stateless:

- **Guests** present a Parlant-minted HS256 token carrying a random
  ``guest_instance_id``. The token is minted when an anonymous browser creates
  its first session and is used to tie that browser to the sessions it created.
- **Customers** present a JWT minted by the integrator's own backend (BYO
  tokens): either HS256 with a shared secret, or an asymmetric algorithm
  verified against the integrator's JWKS URL. The ``sub`` claim is the Parlant
  ``CustomerId``. Parlant never issues, refreshes, or revokes customer tokens.
- **Admins** present a static API key.

All credentials arrive as ``Authorization: Bearer <credential>``. WebSockets,
which cannot set headers from browsers, may pass the credential via a ``token``
query parameter instead.
"""

import hmac
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import jwt
from fastapi import Request, WebSocket

from parlant.core.customers import CustomerId, CustomerStore


GUEST_TOKEN_TYPE = "parlant:guest"

DEFAULT_GUEST_TOKEN_TTL_SECONDS = 60 * 60 * 24 * 7  # one week

_CUSTOMER_JWT_ALGORITHMS = ["HS256", "HS384", "HS512"]
_CUSTOMER_JWKS_ALGORITHMS = ["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"]


class AuthenticationError(Exception):
    """Raised when a request presents credentials that fail verification.

    Distinct from ``AuthorizationException`` (valid identity, forbidden action):
    this maps to HTTP 401, while authorization failures map to 403. A request
    with *no* credentials does not raise — it resolves to ``AnonymousPrincipal``
    and is then subject to the (very limited) anonymous allowlist.
    """


@dataclass(frozen=True, kw_only=True)
class Principal:
    """An authenticated caller identity.

    ``attributes`` is an open extension bag (e.g. the remaining JWT claims)
    that custom permission gates and authorizers may consult. The OSS defaults
    ignore unknown attributes.
    """

    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class AnonymousPrincipal(Principal):
    """A caller that presented no credentials."""


@dataclass(frozen=True, kw_only=True)
class GuestPrincipal(Principal):
    """An unidentified browser holding a Parlant-minted guest token.

    All guests share the singleton guest *customer*; the ``guest_instance_id``
    distinguishes one browser's sessions from another's.
    """

    guest_instance_id: str


@dataclass(frozen=True, kw_only=True)
class CustomerPrincipal(Principal):
    """A specific customer, vouched for by the integrator's token."""

    customer_id: CustomerId


@dataclass(frozen=True, kw_only=True)
class AdminPrincipal(Principal):
    """A fully privileged caller (backoffice / server-to-server)."""


class Authenticator(ABC):
    """Port: resolves a request's credentials into a Principal.

    Implementations must raise ``AuthenticationError`` for credentials that
    fail verification, and return ``AnonymousPrincipal`` when no credentials
    are presented.
    """

    @abstractmethod
    async def resolve(self, request: Request) -> Principal: ...

    @abstractmethod
    async def resolve_websocket(self, websocket: WebSocket) -> Principal: ...

    def issue_guest_token(
        self, guest_instance_id: str, ttl_seconds: int | None = None
    ) -> str | None:
        """Mints a guest credential for the given instance id, if this
        authenticator supports guest identities. Returns None otherwise."""
        return None


class TokenAuthenticator(Authenticator):
    """The OSS default authenticator. Stateless; see the module docstring."""

    def __init__(
        self,
        *,
        guest_token_secret: str,
        admin_api_key: str | None = None,
        customer_jwt_secret: str | None = None,
        customer_jwks_url: str | None = None,
        customer_id_claim: str = "sub",
        guest_token_ttl_seconds: int = DEFAULT_GUEST_TOKEN_TTL_SECONDS,
        jwks_client: Any | None = None,
    ) -> None:
        self._guest_token_secret = guest_token_secret
        self._admin_api_key = admin_api_key
        self._customer_jwt_secret = customer_jwt_secret
        self._customer_id_claim = customer_id_claim
        self._guest_token_ttl_seconds = guest_token_ttl_seconds

        if jwks_client is not None:
            self._jwks_client = jwks_client
        elif customer_jwks_url is not None:
            self._jwks_client = jwt.PyJWKClient(customer_jwks_url)
        else:
            self._jwks_client = None

    async def resolve(self, request: Request) -> Principal:
        return self._resolve_authorization_header(request.headers.get("authorization"))

    async def resolve_websocket(self, websocket: WebSocket) -> Principal:
        if authorization := websocket.headers.get("authorization"):
            return self._resolve_authorization_header(authorization)

        if token := websocket.query_params.get("token"):
            return self._resolve_bearer_token(token)

        return AnonymousPrincipal()

    def issue_guest_token(self, guest_instance_id: str, ttl_seconds: int | None = None) -> str:
        now = int(time.time())
        ttl = self._guest_token_ttl_seconds if ttl_seconds is None else ttl_seconds

        return jwt.encode(
            {
                "sub": guest_instance_id,
                "typ": GUEST_TOKEN_TYPE,
                "iat": now,
                "exp": now + ttl,
            },
            self._guest_token_secret,
            algorithm="HS256",
        )

    def _resolve_authorization_header(self, authorization: str | None) -> Principal:
        if not authorization:
            return AnonymousPrincipal()

        scheme, _, token = authorization.partition(" ")

        if scheme.lower() != "bearer" or not token.strip():
            raise AuthenticationError(
                f"Unsupported authorization scheme: expected 'Bearer', got '{scheme}'"
            )

        return self._resolve_bearer_token(token.strip())

    def _resolve_bearer_token(self, token: str) -> Principal:
        if self._admin_api_key is not None and hmac.compare_digest(token, self._admin_api_key):
            return AdminPrincipal()

        if self._looks_like_guest_token(token):
            return self._verify_guest_token(token)

        return self._verify_customer_token(token)

    def _looks_like_guest_token(self, token: str) -> bool:
        # Routing only — the claim is read unverified here and the token is then
        # strictly verified against the guest secret. A forged "typ" merely
        # routes a token into a verification it cannot pass.
        try:
            claims = jwt.decode(token, options={"verify_signature": False})
        except jwt.InvalidTokenError:
            return False

        return bool(claims.get("typ") == GUEST_TOKEN_TYPE)

    def _verify_guest_token(self, token: str) -> GuestPrincipal:
        try:
            claims = jwt.decode(token, self._guest_token_secret, algorithms=["HS256"])
        except jwt.InvalidTokenError as exc:
            raise AuthenticationError(f"Invalid guest token: {exc}") from exc

        guest_instance_id = claims.get("sub")

        if not guest_instance_id:
            raise AuthenticationError("Guest token is missing its subject claim")

        return GuestPrincipal(guest_instance_id=guest_instance_id, attributes=claims)

    def _verify_customer_token(self, token: str) -> CustomerPrincipal:
        claims = self._verified_customer_claims(token)

        customer_id = claims.get(self._customer_id_claim)

        if not customer_id or not isinstance(customer_id, str):
            raise AuthenticationError(
                f"Customer token is missing its '{self._customer_id_claim}' claim"
            )

        if customer_id == CustomerStore.GUEST_ID:
            raise AuthenticationError(
                f"'{CustomerStore.GUEST_ID}' is a reserved customer identity and cannot be "
                "claimed by a customer token"
            )

        return CustomerPrincipal(customer_id=CustomerId(customer_id), attributes=claims)

    def _verified_customer_claims(self, token: str) -> Mapping[str, Any]:
        if self._jwks_client is not None:
            try:
                signing_key = self._jwks_client.get_signing_key_from_jwt(token)
                claims: Mapping[str, Any] = jwt.decode(
                    token,
                    signing_key.key,
                    algorithms=_CUSTOMER_JWKS_ALGORITHMS,
                )
                return claims
            except (jwt.InvalidTokenError, jwt.PyJWKClientError) as exc:
                raise AuthenticationError(f"Invalid customer token: {exc}") from exc

        if self._customer_jwt_secret is not None:
            try:
                claims = jwt.decode(
                    token,
                    self._customer_jwt_secret,
                    algorithms=_CUSTOMER_JWT_ALGORITHMS,
                )
                return claims
            except jwt.InvalidTokenError as exc:
                raise AuthenticationError(f"Invalid customer token: {exc}") from exc

        raise AuthenticationError(
            "The presented token could not be verified: no customer token verification "
            "is configured (set a customer JWT secret or a JWKS URL)"
        )
