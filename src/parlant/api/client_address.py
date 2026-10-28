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

"""Resolve the address of the client behind a chain of trusted proxies."""

from collections.abc import Iterable
import ipaddress
from typing import TypeAlias

from fastapi import Request

IPAddress: TypeAlias = ipaddress.IPv4Address | ipaddress.IPv6Address
IPNetwork: TypeAlias = ipaddress.IPv4Network | ipaddress.IPv6Network


def parse_trusted_proxies(entries: Iterable[str]) -> tuple[IPNetwork, ...]:
    """Parse trusted proxy entries: single addresses or CIDR ranges.

    Raises:
        ValueError: An entry is neither an IP address nor a CIDR range. The
            message names the entry so the configuration can be corrected.
    """
    networks: list[IPNetwork] = []

    for entry in entries:
        value = entry.strip()

        if not value:
            continue

        try:
            networks.append(ipaddress.ip_network(value, strict=False))
        except ValueError:
            raise ValueError(
                f"Trusted proxy {value!r} is neither an IP address nor a CIDR range. "
                "List proxies as addresses or ranges, for example "
                "'10.0.0.5' or '173.245.48.0/20', separated by commas."
            ) from None

    return tuple(networks)


def _parse_address(value: str) -> IPAddress | None:
    try:
        address = ipaddress.ip_address(value.strip())
    except ValueError:
        return None

    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped

    return address


class ClientAddressResolver:
    """Find the client's IP address for a request.

    Forwarding headers are honored only when the immediate peer is a trusted
    proxy. ``X-Forwarded-For`` is then read from the right, skipping each
    trusted proxy, and the first address that is not trusted is the client.
    Entries further left were written by the client itself and are never
    used. Trusted proxies may be single addresses or CIDR ranges, so a
    deployment can trust its load balancers and a CDN's published networks.
    """

    def __init__(self, trusted_proxies: Iterable[str] = ()) -> None:
        self._trusted_networks = parse_trusted_proxies(trusted_proxies)

    def is_trusted(self, address: IPAddress) -> bool:
        return any(
            address.version == network.version and address in network
            for network in self._trusted_networks
        )

    def resolve(self, request: Request) -> str | None:
        """Return the client address, or None if the request has no peer."""
        peer = request.client.host if request.client else None

        if peer is None:
            return None

        peer_address = _parse_address(peer)

        if peer_address is None or not self.is_trusted(peer_address):
            return peer

        forwarded_for = request.headers.getlist("x-forwarded-for")

        if forwarded_for:
            hops = [hop.strip() for value in forwarded_for for hop in value.split(",")]
            client = str(peer_address)

            for hop in reversed(hops):
                address = _parse_address(hop)

                # The chain cannot be followed past an entry that is not an
                # address; the last real address reached stands.
                if address is None:
                    return client

                client = str(address)

                if not self.is_trusted(address):
                    return client

            return client

        for header in ("x-real-ip", "cf-connecting-ip"):
            if value := request.headers.get(header):
                if address := _parse_address(value):
                    return str(address)

        return peer
