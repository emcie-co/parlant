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

"""Approximate location for an IP address, looked up locally."""

from abc import ABC, abstractmethod
from typing import TypedDict


class ApproximateLocation(TypedDict, total=False):
    """Where an IP address is registered, as a local database records it.

    Approximate by nature: it locates the network, not the device. Every
    field may be missing; ``source`` names the database that answered.
    """

    country_code: str
    country: str
    region: str
    city: str
    source: str


class GeoLocator(ABC):
    """Looks up an approximate location without any network request."""

    @abstractmethod
    async def locate(self, ip: str) -> ApproximateLocation | None:
        """Return the location for ``ip``, or None when it is not known.

        Never raises: a missing database, an unknown address or a corrupt
        file all mean "no location", so callers can always proceed.
        """
        ...


class NullGeoLocator(GeoLocator):
    """The locator used when no GeoIP database is configured."""

    async def locate(self, ip: str) -> ApproximateLocation | None:
        return None
