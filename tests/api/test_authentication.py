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

import time
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Request, WebSocket

from parlant.api.authentication import (
    AdminPrincipal,
    AnonymousPrincipal,
    AuthenticationError,
    CustomerPrincipal,
    GuestPrincipal,
    TokenAuthenticator,
)


GUEST_SECRET = "test-guest-secret-0123456789abcdef"
ADMIN_API_KEY = "test-admin-api-key"
CUSTOMER_JWT_SECRET = "integrator-shared-secret-0123456789abcdef"


def make_request(authorization: str | None = None) -> Request:
    headers = []

    if authorization is not None:
        headers.append((b"authorization", authorization.encode("latin-1")))

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "query_string": b"",
        "http_version": "1.1",
        "scheme": "http",
        "server": ("testserver", 80),
    }

    return Request(scope)


def make_websocket(query_string: str = "") -> WebSocket:
    scope = {
        "type": "websocket",
        "path": "/",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "query_string": query_string.encode("latin-1"),
        "scheme": "ws",
        "server": ("testserver", 80),
    }

    async def receive() -> dict[str, Any]:
        return {"type": "websocket.connect"}

    async def send(message: dict[str, Any]) -> None:
        pass

    return WebSocket(scope, receive=receive, send=send)


def make_authenticator(
    *,
    admin_api_key: str | None = ADMIN_API_KEY,
    customer_jwt_secret: str | None = CUSTOMER_JWT_SECRET,
    customer_jwks_url: str | None = None,
    jwks_client: Any = None,
) -> TokenAuthenticator:
    return TokenAuthenticator(
        guest_token_secret=GUEST_SECRET,
        admin_api_key=admin_api_key,
        customer_jwt_secret=customer_jwt_secret,
        customer_jwks_url=customer_jwks_url,
        jwks_client=jwks_client,
    )


def mint_customer_jwt(
    customer_id: str | None = "cust_42",
    *,
    secret: str = CUSTOMER_JWT_SECRET,
    expires_in_seconds: int = 300,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    claims: dict[str, Any] = {"exp": int(time.time()) + expires_in_seconds}

    if customer_id is not None:
        claims["sub"] = customer_id

    claims.update(extra_claims or {})

    return jwt.encode(claims, secret, algorithm="HS256")


async def test_that_a_request_without_credentials_resolves_to_an_anonymous_principal() -> None:
    authenticator = make_authenticator()

    principal = await authenticator.resolve(make_request())

    assert isinstance(principal, AnonymousPrincipal)


async def test_that_a_guest_token_round_trips_its_guest_instance_id() -> None:
    authenticator = make_authenticator()

    token = authenticator.issue_guest_token("guest-instance-123")

    principal = await authenticator.resolve(make_request(authorization=f"Bearer {token}"))

    assert isinstance(principal, GuestPrincipal)
    assert principal.guest_instance_id == "guest-instance-123"


async def test_that_a_guest_token_is_rejected_by_an_authenticator_with_a_different_secret() -> None:
    minting_authenticator = TokenAuthenticator(
        guest_token_secret="secret-one-0123456789abcdef-0123456789"
    )
    verifying_authenticator = TokenAuthenticator(
        guest_token_secret="secret-two-0123456789abcdef-0123456789"
    )

    token = minting_authenticator.issue_guest_token("guest-instance-123")

    with pytest.raises(AuthenticationError):
        await verifying_authenticator.resolve(make_request(authorization=f"Bearer {token}"))


async def test_that_a_tampered_guest_token_is_rejected() -> None:
    authenticator = make_authenticator()

    token = authenticator.issue_guest_token("guest-instance-123")

    header, payload, signature = token.split(".")
    tampered_token = ".".join((header, payload + "x", signature))

    with pytest.raises(AuthenticationError):
        await authenticator.resolve(make_request(authorization=f"Bearer {tampered_token}"))


async def test_that_an_expired_guest_token_is_rejected() -> None:
    authenticator = make_authenticator()

    token = authenticator.issue_guest_token("guest-instance-123", ttl_seconds=-10)

    with pytest.raises(AuthenticationError):
        await authenticator.resolve(make_request(authorization=f"Bearer {token}"))


async def test_that_an_admin_api_key_resolves_to_an_admin_principal() -> None:
    authenticator = make_authenticator()

    principal = await authenticator.resolve(make_request(authorization=f"Bearer {ADMIN_API_KEY}"))

    assert isinstance(principal, AdminPrincipal)


async def test_that_a_wrong_admin_api_key_is_rejected() -> None:
    authenticator = make_authenticator(customer_jwt_secret=None)

    with pytest.raises(AuthenticationError):
        await authenticator.resolve(make_request(authorization="Bearer not-the-admin-key"))


async def test_that_a_customer_jwt_signed_with_the_shared_secret_resolves_to_a_customer_principal() -> (
    None
):
    authenticator = make_authenticator()

    token = mint_customer_jwt("cust_42")

    principal = await authenticator.resolve(make_request(authorization=f"Bearer {token}"))

    assert isinstance(principal, CustomerPrincipal)
    assert principal.customer_id == "cust_42"


async def test_that_a_customer_jwt_with_an_invalid_signature_is_rejected() -> None:
    authenticator = make_authenticator()

    token = mint_customer_jwt("cust_42", secret="a-completely-different-secret-0123456789")

    with pytest.raises(AuthenticationError):
        await authenticator.resolve(make_request(authorization=f"Bearer {token}"))


async def test_that_an_expired_customer_jwt_is_rejected() -> None:
    authenticator = make_authenticator()

    token = mint_customer_jwt("cust_42", expires_in_seconds=-10)

    with pytest.raises(AuthenticationError):
        await authenticator.resolve(make_request(authorization=f"Bearer {token}"))


async def test_that_a_customer_jwt_missing_the_subject_claim_is_rejected() -> None:
    authenticator = make_authenticator()

    token = mint_customer_jwt(customer_id=None)

    with pytest.raises(AuthenticationError):
        await authenticator.resolve(make_request(authorization=f"Bearer {token}"))


async def test_that_customer_jwt_claims_are_exposed_as_principal_attributes() -> None:
    authenticator = make_authenticator()

    token = mint_customer_jwt("cust_42", extra_claims={"plan": "gold", "org": "acme"})

    principal = await authenticator.resolve(make_request(authorization=f"Bearer {token}"))

    assert isinstance(principal, CustomerPrincipal)
    assert principal.attributes["plan"] == "gold"
    assert principal.attributes["org"] == "acme"


async def test_that_a_customer_jwt_is_rejected_when_no_customer_verification_is_configured() -> (
    None
):
    authenticator = make_authenticator(customer_jwt_secret=None)

    token = mint_customer_jwt("cust_42")

    with pytest.raises(AuthenticationError):
        await authenticator.resolve(make_request(authorization=f"Bearer {token}"))


async def test_that_a_non_bearer_authorization_scheme_is_rejected() -> None:
    authenticator = make_authenticator()

    with pytest.raises(AuthenticationError):
        await authenticator.resolve(make_request(authorization="Basic dXNlcjpwYXNz"))


class _StubJWKSClient:
    def __init__(self, key: Any) -> None:
        self._key = key

    def get_signing_key_from_jwt(self, token: str) -> Any:
        class _SigningKey:
            def __init__(self, key: Any) -> None:
                self.key = key

        return _SigningKey(self._key)


async def test_that_a_jwks_signed_customer_jwt_resolves_to_a_customer_principal() -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    authenticator = make_authenticator(
        customer_jwt_secret=None,
        customer_jwks_url="https://idp.example.com/.well-known/jwks.json",
        jwks_client=_StubJWKSClient(private_key.public_key()),
    )

    token = jwt.encode(
        {"sub": "cust_77", "exp": int(time.time()) + 300},
        private_key,
        algorithm="RS256",
    )

    principal = await authenticator.resolve(make_request(authorization=f"Bearer {token}"))

    assert isinstance(principal, CustomerPrincipal)
    assert principal.customer_id == "cust_77"


async def test_that_a_jwks_customer_jwt_signed_with_a_different_key_is_rejected() -> None:
    trusted_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    rogue_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    authenticator = make_authenticator(
        customer_jwt_secret=None,
        customer_jwks_url="https://idp.example.com/.well-known/jwks.json",
        jwks_client=_StubJWKSClient(trusted_key.public_key()),
    )

    token = jwt.encode(
        {"sub": "cust_77", "exp": int(time.time()) + 300},
        rogue_key,
        algorithm="RS256",
    )

    with pytest.raises(AuthenticationError):
        await authenticator.resolve(make_request(authorization=f"Bearer {token}"))


async def test_that_a_websocket_with_an_admin_token_query_param_resolves_to_an_admin_principal() -> (
    None
):
    authenticator = make_authenticator()

    principal = await authenticator.resolve_websocket(make_websocket(f"token={ADMIN_API_KEY}"))

    assert isinstance(principal, AdminPrincipal)


async def test_that_a_websocket_without_credentials_resolves_to_an_anonymous_principal() -> None:
    authenticator = make_authenticator()

    principal = await authenticator.resolve_websocket(make_websocket())

    assert isinstance(principal, AnonymousPrincipal)
