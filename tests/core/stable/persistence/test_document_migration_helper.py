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

from typing import Any, Awaitable, Callable, Mapping, Optional, cast

import pytest

from parlant.core.common import Version
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import BaseDocument
from parlant.core.persistence.document_database_helper import DocumentMigrationHelper

Converters = Mapping[str, Callable[[BaseDocument], Awaitable[Optional[BaseDocument]]]]


class _VersionedStore:
    """The only thing the helper needs of a store is its target version."""

    def __init__(self, version: str) -> None:
        self.VERSION = Version.from_string(version)


def _document(version: str) -> BaseDocument:
    return BaseDocument(
        id=ObjectId("doc-1"),
        version=Version.String(version),
    )


def _helper(
    store_version: str,
    converters: Optional[Converters] = None,
) -> DocumentMigrationHelper[BaseDocument]:
    return DocumentMigrationHelper[BaseDocument](
        _VersionedStore(store_version),
        converters or {},
    )


async def test_that_a_document_at_the_target_version_is_loaded_as_is() -> None:
    doc = _document("0.6.0")

    assert await _helper("0.6.0").migrate(doc) == doc


@pytest.mark.parametrize("document_version", ["0.5.0", "0.7.0"])
async def test_that_a_document_from_an_unregistered_version_is_not_loadable(
    document_version: str,
) -> None:
    # This release is a clean break: no store registers a converter, so data
    # from any other version — older or newer — is unreadable by design, in
    # either direction. The raise is the mechanism: every document store turns a
    # loader error into an archived document in its failed-migrations
    # collection, which is the intended outcome, not a bug. This test exists so
    # that relaxing the policy (tolerating drift, reading newer data as-is) has
    # to be a deliberate change rather than an accident.
    with pytest.raises(KeyError):
        await _helper("0.6.0").migrate(_document(document_version))


async def test_that_a_document_is_migrated_through_every_registered_converter() -> None:
    async def to_0_6_0(doc: BaseDocument) -> Optional[BaseDocument]:
        return cast(BaseDocument, {**doc, "version": "0.6.0", "upgraded": True})

    async def to_0_7_0(doc: BaseDocument) -> Optional[BaseDocument]:
        return cast(BaseDocument, {**doc, "version": "0.7.0"})

    migrated = await _helper(
        "0.7.0",
        {"0.5.0": to_0_6_0, "0.6.0": to_0_7_0},
    ).migrate(_document("0.5.0"))

    assert migrated is not None
    assert migrated["version"] == "0.7.0"
    assert cast(dict[str, Any], migrated)["upgraded"] is True


async def test_that_a_converter_which_declines_to_migrate_drops_the_document() -> None:
    async def decline(doc: BaseDocument) -> Optional[BaseDocument]:
        return None

    assert await _helper("0.6.0", {"0.5.0": decline}).migrate(_document("0.5.0")) is None
