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
from typing import Any, MutableMapping

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

    async def send(message: MutableMapping[str, Any]) -> None:
        pass

    return WebSocket(scope, receive=receive, send=send)


def make_authenticator(
    *,
    admin_api_key: str | None = ADMIN_API_KEY,
    customer_jwt_secret: str | None = CUSTOMER_JWT_SECRET,
    customer_jwks_url: str | None = None,
    customer_jwt_audience: str | None = None,
    customer_jwt_issuer: str | None = None,
    require_customer_jwt_expiry: bool = False,
    jwks_client: Any = None,
) -> TokenAuthenticator:
    return TokenAuthenticator(
        guest_token_secret=GUEST_SECRET,
        admin_api_key=admin_api_key,
        customer_jwt_secret=customer_jwt_secret,
        customer_jwks_url=customer_jwks_url,
        customer_jwt_audience=customer_jwt_audience,
        customer_jwt_issuer=customer_jwt_issuer,
        require_customer_jwt_expiry=require_customer_jwt_expiry,
        jwks_client=jwks_client,
    )


def mint_customer_jwt(
    customer_id: str | None = "cust_42",
    *,
    secret: str = CUSTOMER_JWT_SECRET,
    expires_in_seconds: int | None = 300,
    extra_claims: dict[str, Any] | None = None,
) -> str:
    claims: dict[str, Any] = {}

    if expires_in_seconds is not None:
        claims["exp"] = int(time.time()) + expires_in_seconds

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


async def test_that_a_customer_jwt_for_a_different_audience_is_rejected() -> None:
    authenticator = make_authenticator(customer_jwt_audience="parlant-api")

    token = mint_customer_jwt("cust_42", extra_claims={"aud": "some-other-service"})

    with pytest.raises(AuthenticationError):
        await authenticator.resolve(make_request(authorization=f"Bearer {token}"))


async def test_that_a_customer_jwt_from_a_different_issuer_is_rejected() -> None:
    authenticator = make_authenticator(customer_jwt_issuer="https://idp.example.com/")

    token = mint_customer_jwt("cust_42", extra_claims={"iss": "https://evil.example.com/"})

    with pytest.raises(AuthenticationError):
        await authenticator.resolve(make_request(authorization=f"Bearer {token}"))


async def test_that_a_customer_jwt_with_the_configured_audience_and_issuer_is_accepted() -> None:
    authenticator = make_authenticator(
        customer_jwt_audience="parlant-api",
        customer_jwt_issuer="https://idp.example.com/",
    )

    token = mint_customer_jwt(
        "cust_42",
        extra_claims={"aud": "parlant-api", "iss": "https://idp.example.com/"},
    )

    principal = await authenticator.resolve(make_request(authorization=f"Bearer {token}"))

    assert isinstance(principal, CustomerPrincipal)
    assert principal.customer_id == "cust_42"


async def test_that_a_customer_jwt_without_an_expiry_is_rejected_when_expiry_is_required() -> None:
    authenticator = make_authenticator(require_customer_jwt_expiry=True)

    token = mint_customer_jwt("cust_42", expires_in_seconds=None)

    with pytest.raises(AuthenticationError):
        await authenticator.resolve(make_request(authorization=f"Bearer {token}"))


async def test_that_a_customer_jwt_without_an_expiry_is_accepted_by_default() -> None:
    authenticator = make_authenticator()

    token = mint_customer_jwt("cust_42", expires_in_seconds=None)

    principal = await authenticator.resolve(make_request(authorization=f"Bearer {token}"))

    assert isinstance(principal, CustomerPrincipal)
    assert principal.customer_id == "cust_42"


# Guest token renewal
###############################################################################

DAY = 24 * 60 * 60


def mint_guest_token(
    *,
    issued_days_ago: float,
    expires_in_days: float,
    identity_started_days_ago: float | None = None,
    guest_instance_id: str = "guest-instance-123",
) -> str:
    now = int(time.time())
    claims: dict[str, Any] = {
        "sub": guest_instance_id,
        "typ": "parlant:guest",
        "iat": now - int(issued_days_ago * DAY),
        "exp": now + int(expires_in_days * DAY),
    }

    if identity_started_days_ago is not None:
        claims["identity_iat"] = now - int(identity_started_days_ago * DAY)

    return jwt.encode(claims, GUEST_SECRET, algorithm="HS256")


async def resolve_guest(authenticator: TokenAuthenticator, token: str) -> GuestPrincipal:
    principal = await authenticator.resolve(make_request(authorization=f"Bearer {token}"))
    assert isinstance(principal, GuestPrincipal)
    return principal


async def test_that_a_new_guest_token_records_when_the_guests_identity_started() -> None:
    authenticator = TokenAuthenticator(guest_token_secret=GUEST_SECRET)

    principal = await resolve_guest(authenticator, authenticator.issue_guest_token("guest-1"))

    assert principal.attributes["identity_iat"] == principal.attributes["iat"]


async def test_that_a_guest_token_past_half_its_lifetime_is_renewed_for_the_same_guest() -> None:
    authenticator = TokenAuthenticator(guest_token_secret=GUEST_SECRET)
    aging_token = mint_guest_token(
        issued_days_ago=4, expires_in_days=3, identity_started_days_ago=4
    )

    renewed_token = authenticator.renew_guest_token(await resolve_guest(authenticator, aging_token))

    assert renewed_token is not None
    renewed = await resolve_guest(authenticator, renewed_token)
    assert renewed.guest_instance_id == "guest-instance-123"
    assert renewed.attributes["exp"] >= int(time.time()) + 7 * DAY - 60


async def test_that_a_guest_token_in_the_first_half_of_its_lifetime_is_not_renewed() -> None:
    authenticator = TokenAuthenticator(guest_token_secret=GUEST_SECRET)
    young_token = mint_guest_token(
        issued_days_ago=1, expires_in_days=6, identity_started_days_ago=1
    )

    assert authenticator.renew_guest_token(await resolve_guest(authenticator, young_token)) is None


async def test_that_a_renewed_guest_token_keeps_the_identitys_original_start() -> None:
    authenticator = TokenAuthenticator(guest_token_secret=GUEST_SECRET)
    aging_token = mint_guest_token(
        issued_days_ago=4, expires_in_days=3, identity_started_days_ago=10
    )
    aging = await resolve_guest(authenticator, aging_token)

    renewed_token = authenticator.renew_guest_token(aging)

    assert renewed_token is not None
    renewed = await resolve_guest(authenticator, renewed_token)
    assert renewed.attributes["identity_iat"] == aging.attributes["identity_iat"]


async def test_that_a_renewed_guest_token_never_expires_after_the_maximum_identity_age() -> None:
    authenticator = TokenAuthenticator(
        guest_token_secret=GUEST_SECRET, guest_identity_max_age_seconds=28 * DAY
    )
    aging_token = mint_guest_token(
        issued_days_ago=4, expires_in_days=3, identity_started_days_ago=23
    )
    aging = await resolve_guest(authenticator, aging_token)

    renewed_token = authenticator.renew_guest_token(aging)

    assert renewed_token is not None
    renewed = await resolve_guest(authenticator, renewed_token)
    assert renewed.attributes["exp"] == aging.attributes["identity_iat"] + 28 * DAY


async def test_that_a_guest_token_already_expiring_at_the_maximum_identity_age_is_not_renewed() -> (
    None
):
    authenticator = TokenAuthenticator(
        guest_token_secret=GUEST_SECRET, guest_identity_max_age_seconds=28 * DAY
    )
    capped_token = mint_guest_token(
        issued_days_ago=5, expires_in_days=2, identity_started_days_ago=26
    )

    assert authenticator.renew_guest_token(await resolve_guest(authenticator, capped_token)) is None


async def test_that_the_maximum_guest_identity_age_defaults_to_28_days() -> None:
    authenticator = TokenAuthenticator(guest_token_secret=GUEST_SECRET)
    old_identity_token = mint_guest_token(
        issued_days_ago=5, expires_in_days=2, identity_started_days_ago=26
    )

    assert (
        authenticator.renew_guest_token(await resolve_guest(authenticator, old_identity_token))
        is None
    )


async def test_that_a_guest_token_without_an_identity_start_counts_its_age_from_its_issue_time() -> (
    None
):
    authenticator = TokenAuthenticator(guest_token_secret=GUEST_SECRET)
    legacy_token = mint_guest_token(issued_days_ago=4, expires_in_days=3)
    legacy = await resolve_guest(authenticator, legacy_token)

    renewed_token = authenticator.renew_guest_token(legacy)

    assert renewed_token is not None
    renewed = await resolve_guest(authenticator, renewed_token)
    assert renewed.attributes["identity_iat"] == legacy.attributes["iat"]
