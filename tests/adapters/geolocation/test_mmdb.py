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

from pathlib import Path

from parlant.adapters.geolocation.mmdb import MmdbGeoLocator

from tests.mmdb_utilities import city_record, write_city_database


def write_fixture(directory: Path) -> Path:
    return write_city_database(
        directory / "city.mmdb",
        {
            "203.0.113.0/24": city_record("GB", "United Kingdom", "England", "London"),
            "2001:db8::/32": city_record("DE", "Germany", "Berlin", "Berlin"),
            "192.0.2.0/24": city_record("FR", "France"),
        },
    )


async def test_that_a_city_record_yields_country_region_and_city(tmp_path: Path) -> None:
    locator = MmdbGeoLocator(write_fixture(tmp_path))

    location = await locator.locate("203.0.113.7")

    assert location == {
        "country_code": "GB",
        "country": "United Kingdom",
        "region": "England",
        "city": "London",
        "source": "DBIP-City-Lite",
    }


async def test_that_an_ipv6_address_is_located(tmp_path: Path) -> None:
    locator = MmdbGeoLocator(write_fixture(tmp_path))

    location = await locator.locate("2001:db8::7")

    assert location is not None
    assert location["country_code"] == "DE"


async def test_that_a_country_only_record_omits_region_and_city(tmp_path: Path) -> None:
    locator = MmdbGeoLocator(write_fixture(tmp_path))

    location = await locator.locate("192.0.2.1")

    assert location == {
        "country_code": "FR",
        "country": "France",
        "source": "DBIP-City-Lite",
    }


async def test_that_an_address_missing_from_the_database_yields_no_location(
    tmp_path: Path,
) -> None:
    locator = MmdbGeoLocator(write_fixture(tmp_path))

    assert await locator.locate("198.51.100.20") is None


async def test_that_a_missing_database_yields_no_location_instead_of_failing(
    tmp_path: Path,
) -> None:
    locator = MmdbGeoLocator(tmp_path / "does-not-exist.mmdb")

    assert await locator.locate("203.0.113.7") is None


async def test_that_a_corrupt_database_yields_no_location_instead_of_failing(
    tmp_path: Path,
) -> None:
    corrupt = tmp_path / "corrupt.mmdb"
    corrupt.write_bytes(b"this is not a MaxMind database")

    locator = MmdbGeoLocator(corrupt)

    assert await locator.locate("203.0.113.7") is None


async def test_that_a_database_provisioned_after_startup_is_picked_up(tmp_path: Path) -> None:
    database = tmp_path / "city.mmdb"
    locator = MmdbGeoLocator(database, retry_interval_seconds=0)

    assert await locator.locate("203.0.113.7") is None

    write_city_database(
        database,
        {"203.0.113.0/24": city_record("GB", "United Kingdom", "England", "London")},
    )

    location = await locator.locate("203.0.113.7")
    assert location is not None
    assert location["city"] == "London"


async def test_that_a_missing_database_is_not_reopened_on_every_lookup(tmp_path: Path) -> None:
    database = tmp_path / "city.mmdb"
    locator = MmdbGeoLocator(database, retry_interval_seconds=3600)

    assert await locator.locate("203.0.113.7") is None

    write_city_database(
        database,
        {"203.0.113.0/24": city_record("GB", "United Kingdom", "England", "London")},
    )

    # Within the retry interval the lookup does not touch the filesystem again.
    assert await locator.locate("203.0.113.7") is None
