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

from __future__ import annotations

from datetime import datetime, timezone
from typing import Mapping, Optional, cast

from typing_extensions import Self, TypedDict, override

from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import IdGenerator, JSONSerializable, Version
from parlant.core.nlp.model_routing import (
    ModelArgument,
    ModelConfig,
    ModelRouteBinding,
    ModelRouteStore,
    ModelRoutingError,
    ModelScope,
    ModelScopeKind,
    SecretRef,
)
from parlant.core.nlp.rate_limits import RateLimit
from parlant.core.persistence.common import ObjectId, Where
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentCollection,
    DocumentDatabase,
)
from parlant.core.persistence.document_database_helper import DocumentStoreMigrationHelper


class _ModelRouteDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    scope_kind: str
    scope_id: str
    route: str
    target: Mapping[str, JSONSerializable]
    revision: str


class ModelRouteDocumentStore(ModelRouteStore):
    VERSION = Version.from_string("0.1.0")

    def __init__(
        self,
        id_generator: IdGenerator,
        database: DocumentDatabase,
        allow_migration: bool = False,
        collections_prefix: str | None = None,
    ) -> None:
        self._id_generator = id_generator
        self._database = database
        self._allow_migration = allow_migration
        self._collections_prefix = collections_prefix
        self._collection: DocumentCollection[_ModelRouteDocument]
        self._lock = ReaderWriterLock()

    async def _document_loader(
        self,
        doc: BaseDocument,
    ) -> Optional[_ModelRouteDocument]:
        if doc["version"] == self.VERSION.to_string():
            return cast(_ModelRouteDocument, doc)
        return None

    async def __aenter__(self) -> Self:
        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
            collections_prefix=self._collections_prefix,
        ):
            self._collection = await self._database.get_or_create_collection(
                name=(
                    f"{self._collections_prefix}_model_routes"
                    if self._collections_prefix
                    else "model_routes"
                ),
                schema=_ModelRouteDocument,
                document_loader=self._document_loader,
            )
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        pass

    @override
    async def read_all_routes(self) -> Mapping[ModelScope, Mapping[str, ModelRouteBinding]]:
        async with self._lock.reader_lock:
            documents = await self._collection.find(filters={})
        result: dict[ModelScope, dict[str, ModelRouteBinding]] = {}
        for document in documents:
            kind = ModelScopeKind(document["scope_kind"])
            if kind not in (ModelScopeKind.SERVER, ModelScopeKind.AGENT):
                continue
            scope = ModelScope(kind, document["scope_id"] or None)
            result.setdefault(scope, {})[document["route"]] = ModelRouteBinding(
                target=self._deserialize_target(document["target"]), revision=document["revision"]
            )
        return result

    @override
    async def read_routes(
        self,
        scope: ModelScope,
    ) -> Mapping[str, ModelRouteBinding]:
        async with self._lock.reader_lock:
            documents = await self._collection.find(
                filters={
                    "scope_kind": {"$eq": scope.kind.value},
                    "scope_id": {"$eq": scope.id or ""},
                }
            )

        return {
            document["route"]: ModelRouteBinding(
                target=self._deserialize_target(document["target"]),
                revision=document["revision"],
            )
            for document in documents
        }

    @override
    async def update_routes(
        self,
        scope: ModelScope,
        updates: Mapping[str, ModelRouteBinding | None],
    ) -> None:
        async with self._lock.writer_lock:
            for route, binding in updates.items():
                filters = cast(
                    Where,
                    {
                        "scope_kind": {"$eq": scope.kind.value},
                        "scope_id": {"$eq": scope.id or ""},
                        "route": {"$eq": route},
                    },
                )
                existing = await self._collection.find_one(filters=filters)

                if binding is None:
                    if existing:
                        await self._collection.delete_one(filters=filters)
                    continue

                target = self._serialize_target(binding.target)
                if existing:
                    await self._collection.update_one(
                        filters=filters,
                        params={
                            "target": target,
                            "revision": binding.revision,
                        },
                    )
                    continue

                await self._collection.insert_one(
                    document=_ModelRouteDocument(
                        id=ObjectId(
                            self._id_generator.generate(
                                f"{scope.kind.value}:{scope.id or ''}:{route}"
                            )
                        ),
                        version=self.VERSION.to_string(),
                        creation_utc=datetime.now(timezone.utc).isoformat(),
                        scope_kind=scope.kind.value,
                        scope_id=scope.id or "",
                        route=route,
                        target=target,
                        revision=binding.revision,
                    )
                )

    def _serialize_target(
        self,
        target: object,
    ) -> Mapping[str, JSONSerializable]:
        if isinstance(target, str):
            return {"kind": "provider", "provider": target}
        if not isinstance(target, ModelConfig):
            raise ModelRoutingError("Unsupported model route target cannot be persisted")
        if callable(target.model):
            raise ModelRoutingError("ModelConfig provider callables cannot be persisted")

        return {
            "kind": "config",
            "model": target.model,
            "args": {name: self._serialize_argument(value) for name, value in target.args.items()},
            "rate_limit": self._serialize_rate_limit(target.rate_limit),
        }

    def _deserialize_target(
        self,
        target: Mapping[str, JSONSerializable],
    ) -> ModelConfig | str:
        if target["kind"] == "provider":
            return cast(str, target["provider"])

        args = cast(Mapping[str, JSONSerializable], target.get("args", {}))
        return ModelConfig(
            model=cast(str, target["model"]),
            args={name: self._deserialize_argument(value) for name, value in args.items()},
            rate_limit=self._deserialize_rate_limit(
                cast(Mapping[str, JSONSerializable] | None, target.get("rate_limit"))
            ),
        )

    def _serialize_rate_limit(
        self,
        rate_limit: RateLimit | None,
    ) -> Mapping[str, JSONSerializable] | None:
        if rate_limit is None:
            return None
        return {
            "max_concurrency": rate_limit.max_concurrency,
            "requests_per_minute": rate_limit.requests_per_minute,
            "bucket": rate_limit.bucket,
        }

    def _deserialize_rate_limit(
        self,
        rate_limit: Mapping[str, JSONSerializable] | None,
    ) -> RateLimit | None:
        if rate_limit is None:
            return None
        return RateLimit(
            max_concurrency=cast(int | None, rate_limit.get("max_concurrency")),
            requests_per_minute=cast(float | None, rate_limit.get("requests_per_minute")),
            bucket=cast(str | None, rate_limit.get("bucket")),
        )

    def _serialize_argument(self, value: ModelArgument) -> JSONSerializable:
        if isinstance(value, SecretRef):
            return {"__secret_ref__": value.key}
        if isinstance(value, Mapping):
            return {str(key): self._serialize_argument(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._serialize_argument(cast(ModelArgument, item)) for item in value]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        raise ModelRoutingError(
            f"Model argument of type {type(value).__name__} cannot be persisted"
        )

    def _deserialize_argument(self, value: JSONSerializable) -> ModelArgument:
        if isinstance(value, Mapping):
            if set(value) == {"__secret_ref__"}:
                return SecretRef(cast(str, value["__secret_ref__"]))
            return cast(
                ModelArgument,
                {str(key): self._deserialize_argument(item) for key, item in value.items()},
            )
        if isinstance(value, list):
            return cast(
                ModelArgument,
                [self._deserialize_argument(item) for item in value],
            )
        return cast(ModelArgument, value)
