from typing import Awaitable, Callable, Generic, Mapping, Optional, cast
from typing_extensions import Self
from parlant.core.common import Version
from parlant.core.persistence.common import MigrationRequired, VersionedStore
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
        if "version" in metadata:
            return metadata["version"] != runtime_store_version
        else:
            await database.upsert_metadata("version", runtime_store_version)
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
