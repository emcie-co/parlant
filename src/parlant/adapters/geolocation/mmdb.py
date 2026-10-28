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

"""GeoIP lookups against a local MaxMind DB (.mmdb) file.

Works with any City-shaped database in the MaxMind DB format, such as DB-IP
City Lite or MaxMind GeoLite2 City. The file is memory-mapped and read
locally: a lookup never makes a network request. Provisioning and refreshing
the file is the deployment's job.
"""

from pathlib import Path
import time
from typing import Any, Mapping

import maxminddb

from parlant.core.geolocation import ApproximateLocation, GeoLocator
from parlant.core.loggers import Logger

DEFAULT_RETRY_INTERVAL_SECONDS = 300.0
"""How long to wait before trying again to open a database that failed to open."""


def _english_name(record: Any) -> str | None:
    if not isinstance(record, Mapping):
        return None
    names = record.get("names")
    if not isinstance(names, Mapping):
        return None
    name = names.get("en")
    return name if isinstance(name, str) and name else None


class MmdbGeoLocator(GeoLocator):
    """Locate addresses with a City database in MaxMind DB format."""

    def __init__(
        self,
        database_path: Path,
        logger: Logger | None = None,
        retry_interval_seconds: float = DEFAULT_RETRY_INTERVAL_SECONDS,
    ) -> None:
        self._database_path = database_path
        self._logger = logger
        self._retry_interval_seconds = retry_interval_seconds
        self._reader: Any = None
        self._next_attempt = 0.0
        self._warned = False

    def _open(self) -> Any:
        # A host may provision the database after the server starts, so a
        # failed open is retried, but at most once per interval: lookups
        # between attempts cost only this clock check.
        if self._reader is None and time.monotonic() >= self._next_attempt:
            try:
                self._reader = maxminddb.open_database(str(self._database_path))
            except (OSError, ValueError, maxminddb.InvalidDatabaseError) as exc:
                self._next_attempt = time.monotonic() + self._retry_interval_seconds
                if self._logger and not self._warned:
                    self._warned = True
                    self._logger.warning(
                        f"GeoIP database {self._database_path} could not be opened "
                        f"({type(exc).__name__}); sessions will record no location until it "
                        f"can be opened (retried every {self._retry_interval_seconds:g}s). "
                        "Check that the file exists and is a MaxMind DB City database."
                    )
        return self._reader

    async def locate(self, ip: str) -> ApproximateLocation | None:
        reader = self._open()
        if reader is None:
            return None

        try:
            record = reader.get(ip)
        except (ValueError, maxminddb.InvalidDatabaseError):
            return None

        if not isinstance(record, Mapping):
            return None

        location = ApproximateLocation()

        country = record.get("country")
        if isinstance(country, Mapping) and isinstance(country.get("iso_code"), str):
            location["country_code"] = country["iso_code"]
        if country_name := _english_name(country):
            location["country"] = country_name

        subdivisions = record.get("subdivisions")
        if isinstance(subdivisions, list) and subdivisions:
            if region := _english_name(subdivisions[0]):
                location["region"] = region

        if city := _english_name(record.get("city")):
            location["city"] = city

        if not location:
            return None

        location["source"] = str(reader.metadata().database_type)
        return location
