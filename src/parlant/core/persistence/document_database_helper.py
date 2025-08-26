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
from typing_extensions import TypedDict, Self
from parlant.core.common import Version, generate_id
from parlant.core.persistence.common import (
    MigrationRequired,
    ObjectId,
    ServerOutdated,
    VersionedStore,
)
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentDatabase,
    TDocument,
    identity_loader,
)


class MetadataDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String


class DocumentStoreMigrationHelper:
    def __init__(
        self,
        store: VersionedStore,
        database: DocumentDatabase,
        allow_migration: bool,
    ):
        self._store_name = store.__class__.__name__
        self._runtime_store_version = store.VERSION.to_string()
        self._database = database
        self._allow_migration = allow_migration

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
            await self._update_metadata_version(self._database, self._runtime_store_version)

        return False

    async def _is_migration_required(
        self,
        database: DocumentDatabase,
        runtime_store_version: Version.String,
    ) -> bool:
        metadata_collection = await database.get_or_create_collection(
            "metadata",
            MetadataDocument,
            identity_loader,
        )

        if metadata := await metadata_collection.find_one({}):
            if Version.from_string(cast(str, metadata["version"])) > Version.from_string(
                runtime_store_version
            ):
                raise ServerOutdated

            return metadata["version"] != runtime_store_version
        else:
            await metadata_collection.insert_one(
                {
                    "id": ObjectId(generate_id()),
                    "version": runtime_store_version,
                }
            )
            return False  # No migration is required for a new store

    async def _update_metadata_version(
        self,
        database: DocumentDatabase,
        runtime_store_version: Version.String,
    ) -> None:
        metadata_collection = await database.get_or_create_collection(
            "metadata",
            MetadataDocument,
            identity_loader,
        )

        for doc in await metadata_collection.find({}):
            await metadata_collection.update_one(
                filters={"id": {"$eq": doc["id"]}},
                params={"version": runtime_store_version},
            )


class DocumentMigrationHelper(Generic[TDocument]):
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
