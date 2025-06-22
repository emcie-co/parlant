# Copyright 2025 Emcie Co Ltd.
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

from typing import Awaitable, Callable, Generic, Mapping, Optional, cast
from typing_extensions import Self
from parlant.core.common import Version
from parlant.core.persistence.common import MigrationRequired, ServerOutdated, VersionedStore
from parlant.core.persistence.vector_database import BaseDocument, TDocument, VectorDatabase


class VectorDocumentStoreMigrationHelper:
    def __init__(
        self,
        store: VersionedStore,
        database: VectorDatabase,
        allow_migration: bool,
    ):
        self._store_name = store.__class__.__name__
        self._runtime_store_version = store.VERSION.to_string()
        self._database = database
        self._allow_migration = allow_migration

    @staticmethod
    def get_store_version_key(store_name: str) -> str:
        return f"{store_name}_version"

    async def __aenter__(self) -> Self:
        migration_required = await self._is_migration_required(
            self._database,
            self._runtime_store_version,
        )

        if migration_required and not self._allow_migration:
            raise MigrationRequired(f"Migration required for {self._store_name}.")

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> bool:
        if exc_type is None:
            await self._update_metadata_version(
                self._database,
                self._runtime_store_version,
            )

        return False

    async def _is_migration_required(
        self,
        database: VectorDatabase,
        runtime_store_version: Version.String,
    ) -> bool:
        metadata = await database.read_metadata()
        key = self.get_store_version_key(self._store_name)
        if key in metadata:
            if Version.from_string(cast(str, metadata[key])) > Version.from_string(
                runtime_store_version
            ):
                raise ServerOutdated

            return metadata[key] != runtime_store_version
        else:
            await database.upsert_metadata(key, runtime_store_version)
            return False  # No migration is required for a new store

    async def _update_metadata_version(
        self,
        database: VectorDatabase,
        runtime_store_version: Version.String,
    ) -> None:
        await database.upsert_metadata("version", runtime_store_version)


class VectorDocumentMigrationHelper(Generic[TDocument]):
    def __init__(
        self,
        versioned_store: VersionedStore,
        converters: Mapping[str, Callable[[BaseDocument], Awaitable[Optional[BaseDocument]]]],
    ) -> None:
        self.target_version = versioned_store.VERSION.to_string()
        self.converters = converters

    async def migrate(self, doc: BaseDocument) -> Optional[TDocument]:
        while doc["version"] != self.target_version:
            if converted_doc := await self.converters[doc["version"]](doc):
                doc = converted_doc
            else:
                return None

        return cast(TDocument, doc)
