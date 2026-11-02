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

from fastapi import Request

from parlant.api.client_address import ClientAddressResolver

# A Cloudflare edge range, as a deployment behind Cloudflare would list it.
CLOUDFLARE_EDGE = "173.245.48.0/20"


def make_request(
    *,
    client_host: str | None,
    headers: dict[str, str] | None = None,
) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/sessions",
            "headers": [
                (name.lower().encode("latin-1"), value.encode("latin-1"))
                for name, value in (headers or {}).items()
            ],
            "client": (client_host, 50000) if client_host is not None else None,
            "query_string": b"",
            "http_version": "1.1",
            "scheme": "http",
            "server": ("testserver", 80),
        }
    )


def test_that_the_peer_address_is_the_client_when_no_proxy_is_trusted() -> None:
    resolver = ClientAddressResolver(trusted_proxies=frozenset())

    request = make_request(client_host="203.0.113.7")

    assert resolver.resolve(request) == "203.0.113.7"


def test_that_forwarded_for_is_ignored_from_an_untrusted_peer() -> None:
    resolver = ClientAddressResolver(trusted_proxies=frozenset({"10.0.0.5"}))

    request = make_request(
        client_host="198.51.100.20",
        headers={"X-Forwarded-For": "203.0.113.7"},
    )

    assert resolver.resolve(request) == "198.51.100.20"


def test_that_the_rightmost_untrusted_forwarded_address_is_the_client() -> None:
    resolver = ClientAddressResolver(trusted_proxies=frozenset({"10.0.0.5", CLOUDFLARE_EDGE}))

    # The browser prepended a forged entry; Cloudflare appended the real
    # client; the local proxy appended the Cloudflare edge it received from.
    request = make_request(
        client_host="10.0.0.5",
        headers={"X-Forwarded-For": "198.51.100.66, 203.0.113.7, 173.245.48.10"},
    )

    assert resolver.resolve(request) == "203.0.113.7"


def test_that_trusted_proxies_accept_cidr_ranges() -> None:
    resolver = ClientAddressResolver(trusted_proxies=frozenset({"10.0.0.0/8"}))

    request = make_request(
        client_host="10.20.30.40",
        headers={"X-Forwarded-For": "203.0.113.7"},
    )

    assert resolver.resolve(request) == "203.0.113.7"


def test_that_an_ipv6_client_is_resolved_through_an_ipv4_proxy_chain() -> None:
    resolver = ClientAddressResolver(trusted_proxies=frozenset({"10.0.0.5", CLOUDFLARE_EDGE}))

    request = make_request(
        client_host="10.0.0.5",
        headers={"X-Forwarded-For": "2001:db8::7, 173.245.48.10"},
    )

    assert resolver.resolve(request) == "2001:db8::7"


def test_that_an_invalid_forwarded_address_is_never_returned_as_the_client() -> None:
    resolver = ClientAddressResolver(trusted_proxies=frozenset({"10.0.0.5"}))

    request = make_request(
        client_host="10.0.0.5",
        headers={"X-Forwarded-For": "203.0.113.7, not-an-address"},
    )

    # The chain cannot be followed past a hop that is not an address, so the
    # last address that is known to be real (the trusted peer) is used.
    assert resolver.resolve(request) == "10.0.0.5"


def test_that_the_leftmost_address_is_the_client_when_every_forwarded_hop_is_trusted() -> None:
    resolver = ClientAddressResolver(trusted_proxies=frozenset({"10.0.0.0/8"}))

    # Every hop is a trusted proxy, so the furthest one the chain names is the
    # closest thing to a client it can identify.
    request = make_request(
        client_host="10.0.0.5",
        headers={"X-Forwarded-For": "10.1.1.1, 10.2.2.2"},
    )

    assert resolver.resolve(request) == "10.1.1.1"


def test_that_x_real_ip_is_used_from_a_trusted_peer_without_forwarded_for() -> None:
    resolver = ClientAddressResolver(trusted_proxies=frozenset({"10.0.0.5"}))

    request = make_request(
        client_host="10.0.0.5",
        headers={"X-Real-IP": "203.0.113.7"},
    )

    assert resolver.resolve(request) == "203.0.113.7"


def test_that_no_client_is_resolved_without_a_peer_address() -> None:
    resolver = ClientAddressResolver(trusted_proxies=frozenset({"10.0.0.5"}))

    request = make_request(client_host=None, headers={"X-Forwarded-For": "203.0.113.7"})

    assert resolver.resolve(request) is None
