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

"""What a session records about the browser that opened it."""

import ipaddress
import re
from typing import Mapping
from urllib.parse import urlsplit

from parlant.core.common import JSONSerializable
from parlant.core.geolocation import GeoLocator, NullGeoLocator

SESSION_CLIENT_METADATA_KEY = "client"
"""The key under the session's reserved ``__parlant__`` metadata."""


def web_origin(url: str | None) -> str | None:
    """The http(s) origin of ``url`` (scheme, host and port), or None.

    Paths, queries, fragments and credentials are dropped, so the page
    address can never carry personal data from a URL into the session.
    Opaque origins (``null``) and non-web schemes yield None.
    """
    if not url:
        return None

    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError:
        return None

    scheme = parts.scheme.lower()

    if scheme not in ("http", "https") or not parts.hostname:
        return None

    host = _web_host(parts.hostname)

    if host is None:
        return None

    return f"{scheme}://{host}" + (f":{port}" if port is not None else "")


_MAX_HOST_LENGTH = 253
_HOST_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


def _web_host(hostname: str) -> str | None:
    """An IP literal or a DNS hostname (IDNA-encoded), or None if malformed."""
    if len(hostname) > _MAX_HOST_LENGTH:
        return None

    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        return f"[{address}]" if address.version == 6 else str(address)

    try:
        ascii_host = hostname.encode("idna").decode("ascii").lower()
    except (UnicodeError, ValueError):
        return None

    labels = ascii_host.removesuffix(".").split(".")

    if len(ascii_host) > _MAX_HOST_LENGTH or not all(_HOST_LABEL.match(label) for label in labels):
        return None

    return ascii_host


def normalized_ip(value: str | None) -> str | None:
    if not value:
        return None

    try:
        address = ipaddress.ip_address(value.strip())
    except ValueError:
        return None

    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return str(address.ipv4_mapped)

    return str(address)


class SessionClientRecorder:
    """Builds the trusted ``client`` record for a new session.

    The record holds the browser's IP address, the page origin, and an
    approximate location looked up locally. Each part is optional; when
    nothing valid remains, or recording is switched off, there is no record.
    """

    def __init__(
        self,
        geo_locator: GeoLocator | None = None,
        enabled: bool = True,
    ) -> None:
        self._geo_locator = geo_locator or NullGeoLocator()
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def record(
        self,
        ip: str | None,
        origin: str | None,
    ) -> Mapping[str, JSONSerializable] | None:
        if not self._enabled:
            return None

        client: dict[str, JSONSerializable] = {}

        if address := normalized_ip(ip):
            client["ip"] = address

        if page_origin := web_origin(origin):
            client["origin"] = page_origin

        if address:
            try:
                location = await self._geo_locator.locate(address)
            except Exception:
                location = None

            if location:
                client["location"] = {key: str(value) for key, value in location.items()}

        return client or None
