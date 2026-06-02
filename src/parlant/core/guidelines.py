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

from itertools import chain
from typing import Awaitable, Callable, Mapping, NewType, Optional, Sequence, Set, cast
from typing_extensions import override, TypedDict, Self, Required
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone

from parlant.core import async_utils
from parlant.core.agents import CompositionMode
from parlant.core.async_utils import ReaderWriterLock, safe_gather
from parlant.core.common import (
    Criticality,
    ItemNotFoundError,
    try_or_none,
    JSONSerializable,
    UniqueId,
    Version,
    IdGenerator,
    xxh3_checksum,
)
from parlant.core.nlp.embedding import Embedder, EmbedderFactory
from parlant.core.persistence.common import ObjectId, Where
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentDatabase,
    DocumentCollection,
)
from parlant.core.persistence.document_database_helper import (
    DocumentStoreMigrationHelper,
    DocumentMigrationHelper,
)
from parlant.core.persistence.vector_database import (
    SimilarDocumentResult,
    VectorCollection,
    VectorCollectionIndex,
    VectorDatabase,
    BaseDocument as VectorDocument,
)
from parlant.core.persistence.vector_database_helper import (
    VectorDocumentStoreMigrationHelper,
    calculate_min_vectors_for_max_item_count,
    query_chunks,
)
from parlant.core.tags import TagId

GuidelineId = NewType("GuidelineId", str)


@dataclass(frozen=True)
class GuidelineContent:
    condition: str
    action: Optional[str]
    description: Optional[str] = field(default=None)


@dataclass(frozen=True)
class Guideline:
    id: GuidelineId
    creation_utc: datetime
    last_modified_utc: datetime
    content: GuidelineContent
    enabled: bool
    tags: Sequence[TagId]
    metadata: Mapping[str, JSONSerializable]
    criticality: Criticality
    title: Optional[str] = None
    labels: Set[str] = field(default_factory=set)
    composition_mode: Optional[CompositionMode] = None
    track: bool = True
    priority: int = 0
    signals: Sequence[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.content.condition and self.content.action:
            return f"When {self.content.condition}, then {self.content.action}"
        elif self.content.condition:
            return f"Observation: {self.content.condition}"
        elif self.content.action:
            return self.content.action
        else:
            raise Exception("Invalid guideline content")

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(self.id)


class GuidelineUpdateParams(TypedDict, total=False):
    condition: str
    action: Optional[str]
    description: Optional[str]
    title: Optional[str]
    criticality: Criticality
    enabled: bool
    metadata: Mapping[str, JSONSerializable]
    composition_mode: Optional[CompositionMode]
    track: bool
    priority: int
    signals: Sequence[str]


@dataclass(frozen=True)
class GuidelineRelevanceResult:
    guideline: Guideline
    score: float


class GuidelineStore(ABC):
    @abstractmethod
    async def create_guideline(
        self,
        condition: str,
        action: Optional[str] = None,
        description: Optional[str] = None,
        title: Optional[str] = None,
        criticality: Optional[Criticality] = None,
        metadata: Mapping[str, JSONSerializable] = {},
        creation_utc: Optional[datetime] = None,
        enabled: bool = True,
        tags: Optional[Sequence[TagId]] = None,
        id: Optional[GuidelineId] = None,
        composition_mode: Optional[CompositionMode] = None,
        track: bool = True,
        labels: Optional[Set[str]] = None,
        priority: int = 0,
        signals: Sequence[str] = [],
    ) -> Guideline: ...

    @abstractmethod
    async def list_guidelines(
        self,
        tags: Optional[Sequence[TagId]] = None,
        labels: Optional[Set[str]] = None,
    ) -> Sequence[Guideline]: ...

    @abstractmethod
    async def find_relevant_guidelines(
        self,
        query: str,
        available_guidelines: Sequence[Guideline],
        max_count: int,
    ) -> Sequence[GuidelineRelevanceResult]: ...

    @abstractmethod
    async def read_guideline(
        self,
        guideline_id: GuidelineId,
    ) -> Guideline: ...

    @abstractmethod
    async def delete_guideline(
        self,
        guideline_id: GuidelineId,
    ) -> None: ...

    @abstractmethod
    async def update_guideline(
        self,
        guideline_id: GuidelineId,
        params: GuidelineUpdateParams,
    ) -> Guideline: ...

    @abstractmethod
    async def find_guideline(
        self,
        guideline_content: GuidelineContent,
    ) -> Guideline: ...

    @abstractmethod
    async def upsert_tag(
        self,
        guideline_id: GuidelineId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> bool: ...

    @abstractmethod
    async def remove_tag(
        self,
        guideline_id: GuidelineId,
        tag_id: TagId,
    ) -> None: ...

    @abstractmethod
    async def set_metadata(
        self,
        guideline_id: GuidelineId,
        key: str,
        value: JSONSerializable,
    ) -> Guideline: ...

    @abstractmethod
    async def unset_metadata(
        self,
        guideline_id: GuidelineId,
        key: str,
    ) -> Guideline: ...

    @abstractmethod
    async def upsert_labels(
        self,
        guideline_id: GuidelineId,
        labels: Set[str],
    ) -> Guideline: ...

    @abstractmethod
    async def remove_labels(
        self,
        guideline_id: GuidelineId,
        labels: Set[str],
    ) -> Guideline: ...


class GuidelineDocument_v0_1_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    guideline_set: str
    condition: str
    action: str


class GuidelineDocument_v0_2_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    guideline_set: str
    condition: str
    action: str
    enabled: bool


class GuidelineDocument_v0_3_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    condition: str
    action: str
    enabled: bool


class GuidelineDocument_v0_4_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    condition: str
    action: Optional[str]
    description: Optional[str]
    enabled: bool
    metadata: Mapping[str, JSONSerializable]


class GuidelineDocument_v0_5_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    condition: str
    action: Optional[str]
    description: Optional[str]
    enabled: bool
    metadata: Mapping[str, JSONSerializable]


class GuidelineDocument_v0_6_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    condition: str
    action: Optional[str]
    description: Optional[str]
    criticality: str
    enabled: bool
    metadata: Mapping[str, JSONSerializable]


class GuidelineDocument_v0_7_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    condition: str
    action: Optional[str]
    description: Optional[str]
    criticality: str
    enabled: bool
    metadata: Mapping[str, JSONSerializable]
    composition_mode: Optional[str]


class GuidelineDocument_v0_8_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    condition: str
    action: Optional[str]
    description: Optional[str]
    criticality: str
    enabled: bool
    metadata: Mapping[str, JSONSerializable]
    composition_mode: Optional[str]
    track: bool


class GuidelineDocument_v0_9_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    condition: str
    action: Optional[str]
    description: Optional[str]
    criticality: str
    enabled: bool
    metadata: Mapping[str, JSONSerializable]
    composition_mode: Optional[str]
    track: bool
    labels: Sequence[str]


class GuidelineDocument_v0_10_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    condition: str
    action: Optional[str]
    description: Optional[str]
    criticality: str
    enabled: bool
    metadata: Mapping[str, JSONSerializable]
    composition_mode: Optional[str]
    track: bool
    labels: Sequence[str]
    priority: int


class GuidelineDocument_v0_11_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    last_modified: str
    condition: str
    action: Optional[str]
    description: Optional[str]
    title: Optional[str]
    criticality: str
    enabled: bool
    metadata: Mapping[str, JSONSerializable]
    composition_mode: Optional[str]
    track: bool
    labels: Sequence[str]
    priority: int


class GuidelineDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    last_modified: str
    condition: str
    action: Optional[str]
    description: Optional[str]
    title: Optional[str]
    criticality: str
    enabled: bool
    metadata: Mapping[str, JSONSerializable]
    composition_mode: Optional[str]
    track: bool
    labels: Sequence[str]
    priority: int
    signals: Sequence[str]


class GuidelineVectorDocument(TypedDict, total=False):
    id: ObjectId
    guideline_id: ObjectId
    version: Version.String
    content: str
    checksum: Required[str]


class GuidelineTagAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    guideline_id: GuidelineId
    tag_id: TagId


async def guideline_document_converter_0_1_0_to_0_2_0(doc: BaseDocument) -> Optional[BaseDocument]:
    d = cast(GuidelineDocument_v0_1_0, doc)
    return GuidelineDocument_v0_2_0(
        id=d["id"],
        version=Version.String("0.2.0"),
        creation_utc=d["creation_utc"],
        guideline_set=d["guideline_set"],
        condition=d["condition"],
        action=d["action"],
        enabled=True,
    )


class GuidelineVectorStore(GuidelineStore):
    VERSION = Version.from_string("0.12.0")

    def __init__(
        self,
        id_generator: IdGenerator,
        vector_db: VectorDatabase,
        document_db: DocumentDatabase,
        embedder_type_provider: Callable[[], Awaitable[type[Embedder]]],
        embedder_factory: EmbedderFactory,
        allow_migration: bool = True,
        collections_prefix: str | None = None,
    ) -> None:
        self._id_generator = id_generator

        self._vector_db = vector_db
        self._database = document_db
        self._vector_collection: VectorCollection[GuidelineVectorDocument]
        self._collection: DocumentCollection[GuidelineDocument]
        self._tag_association_collection: DocumentCollection[GuidelineTagAssociationDocument]

        self._allow_migration = allow_migration
        self._collections_prefix = collections_prefix
        self._lock = ReaderWriterLock()
        self._embedder_factory = embedder_factory
        self._embedder_type_provider = embedder_type_provider
        self._embedder: Embedder

    async def _vector_document_loader(
        self, doc: VectorDocument
    ) -> Optional[GuidelineVectorDocument]:
        return cast(GuidelineVectorDocument, doc)

    async def _document_loader(self, doc: BaseDocument) -> Optional[GuidelineDocument]:
        async def v0_11_0_to_v0_12_0(doc: BaseDocument) -> Optional[BaseDocument]:
            d = cast(GuidelineDocument_v0_11_0, doc)
            return GuidelineDocument(
                id=d["id"],
                version=Version.String("0.12.0"),
                creation_utc=d["creation_utc"],
                last_modified=d.get("last_modified", d["creation_utc"]),
                condition=d["condition"],
                action=d["action"],
                description=d.get("description", None),
                title=d.get("title", None),
                criticality=d["criticality"],
                enabled=d["enabled"],
                metadata=d["metadata"],
                composition_mode=d.get("composition_mode"),
                track=d.get("track", True),
                labels=d.get("labels", []),
                priority=d.get("priority", 0),
                signals=[],
            )

        async def v0_10_0_to_v0_11_0(doc: BaseDocument) -> Optional[BaseDocument]:
            d = cast(GuidelineDocument_v0_10_0, doc)
            return GuidelineDocument_v0_11_0(
                id=d["id"],
                version=Version.String("0.11.0"),
                creation_utc=d["creation_utc"],
                last_modified=d["creation_utc"],
                condition=d["condition"],
                action=d["action"],
                description=d.get("description", None),
                title=None,  # Default to None for existing guidelines
                criticality=d["criticality"],
                enabled=d["enabled"],
                metadata=d["metadata"],
                composition_mode=d.get("composition_mode"),
                track=d.get("track", True),
                labels=d.get("labels", []),
                priority=d.get("priority", 0),
            )

        async def v0_9_0_to_v0_10_0(doc: BaseDocument) -> Optional[BaseDocument]:
            d = cast(GuidelineDocument_v0_9_0, doc)
            return GuidelineDocument_v0_10_0(
                id=d["id"],
                version=Version.String("0.10.0"),
                creation_utc=d["creation_utc"],
                condition=d["condition"],
                action=d["action"],
                description=d.get("description", None),
                criticality=d["criticality"],
                enabled=d["enabled"],
                metadata=d["metadata"],
                composition_mode=d.get("composition_mode"),
                track=d.get("track", True),
                labels=d.get("labels", []),
                priority=0,  # Default to 0 for existing guidelines
            )

        async def v0_8_0_to_v0_9_0(doc: BaseDocument) -> Optional[BaseDocument]:
            d = cast(GuidelineDocument_v0_8_0, doc)
            return GuidelineDocument_v0_9_0(
                id=d["id"],
                version=Version.String("0.9.0"),
                creation_utc=d["creation_utc"],
                condition=d["condition"],
                action=d["action"],
                description=d.get("description", None),
                criticality=d["criticality"],
                enabled=d["enabled"],
                metadata=d["metadata"],
                composition_mode=d.get("composition_mode"),
                track=d.get("track", True),
                labels=[],  # Default to empty labels for existing guidelines
            )

        async def v0_7_0_to_v0_8_0(doc: BaseDocument) -> Optional[BaseDocument]:
            d = cast(GuidelineDocument_v0_7_0, doc)
            return GuidelineDocument_v0_8_0(
                id=d["id"],
                version=Version.String("0.8.0"),
                creation_utc=d["creation_utc"],
                condition=d["condition"],
                action=d["action"],
                description=d.get("description", None),
                criticality=d["criticality"],
                enabled=d["enabled"],
                metadata=d["metadata"],
                composition_mode=d.get("composition_mode"),
                track=True,  # Default to True for existing guidelines
            )

        async def v0_6_0_to_v0_7_0(doc: BaseDocument) -> Optional[BaseDocument]:
            d = cast(GuidelineDocument_v0_6_0, doc)
            return GuidelineDocument_v0_7_0(
                id=d["id"],
                version=Version.String("0.7.0"),
                creation_utc=d["creation_utc"],
                condition=d["condition"],
                action=d["action"],
                description=d.get("description", None),
                criticality=d["criticality"],
                enabled=d["enabled"],
                metadata=d["metadata"],
                composition_mode=None,  # Default to None for existing guidelines
            )

        async def v0_5_0_to_v0_6_0(doc: BaseDocument) -> Optional[BaseDocument]:
            d = cast(GuidelineDocument_v0_5_0, doc)
            return GuidelineDocument_v0_6_0(
                id=d["id"],
                version=Version.String("0.6.0"),
                creation_utc=d["creation_utc"],
                condition=d["condition"],
                action=d["action"],
                description=d.get("description", None),
                criticality="medium",  # Default to MEDIUM for existing guidelines
                enabled=d["enabled"],
                metadata=d["metadata"],
            )

        async def v0_4_0_to_v0_5_0(doc: BaseDocument) -> Optional[BaseDocument]:
            d = cast(GuidelineDocument_v0_4_0, doc)
            return GuidelineDocument_v0_5_0(
                id=d["id"],
                version=Version.String("0.5.0"),
                creation_utc=d["creation_utc"],
                condition=d["condition"],
                action=d["action"],
                description=d.get("description", None),
                enabled=d["enabled"],
                metadata=d["metadata"],
            )

        async def v0_3_0_to_v0_4_0(doc: BaseDocument) -> Optional[BaseDocument]:
            d = cast(GuidelineDocument_v0_3_0, doc)
            return GuidelineDocument_v0_4_0(
                id=d["id"],
                version=Version.String("0.4.0"),
                creation_utc=d["creation_utc"],
                condition=d["condition"],
                action=d["action"],
                enabled=d["enabled"],
                metadata={},
            )

        async def v0_2_0_to_v0_3_0(doc: BaseDocument) -> Optional[BaseDocument]:
            raise Exception(
                "This code should not be reached! Please run the 'parlant-prepare-migration' script."
            )

        return await DocumentMigrationHelper[GuidelineDocument](
            self,
            {
                "0.1.0": guideline_document_converter_0_1_0_to_0_2_0,
                "0.2.0": v0_2_0_to_v0_3_0,
                "0.3.0": v0_3_0_to_v0_4_0,
                "0.4.0": v0_4_0_to_v0_5_0,
                "0.5.0": v0_5_0_to_v0_6_0,
                "0.6.0": v0_6_0_to_v0_7_0,
                "0.7.0": v0_7_0_to_v0_8_0,
                "0.8.0": v0_8_0_to_v0_9_0,
                "0.9.0": v0_9_0_to_v0_10_0,
                "0.10.0": v0_10_0_to_v0_11_0,
                "0.11.0": v0_11_0_to_v0_12_0,
            },
        ).migrate(doc)

    async def _association_document_loader(
        self, doc: BaseDocument
    ) -> Optional[GuidelineTagAssociationDocument]:
        if doc["version"] == "0.3.0":
            d = cast(GuidelineTagAssociationDocument, doc)
            return GuidelineTagAssociationDocument(
                id=d["id"],
                version=Version.String("0.5.0"),
                creation_utc=d["creation_utc"],
                guideline_id=d["guideline_id"],
                tag_id=d["tag_id"],
            )

        if doc["version"] == "0.4.0":
            d = cast(GuidelineTagAssociationDocument, doc)
            return GuidelineTagAssociationDocument(
                id=d["id"],
                version=Version.String("0.5.0"),
                creation_utc=d["creation_utc"],
                guideline_id=d["guideline_id"],
                tag_id=d["tag_id"],
            )

        if doc["version"] == "0.5.0":
            return cast(GuidelineTagAssociationDocument, doc)

        return None

    async def __aenter__(self) -> Self:
        embedder_type = await self._embedder_type_provider()

        self._embedder = self._embedder_factory.create_embedder(embedder_type)

        async with VectorDocumentStoreMigrationHelper(
            store=self,
            database=self._vector_db,
            allow_migration=self._allow_migration,
        ):
            self._vector_collection = await self._vector_db.get_or_create_collection(
                name=f"{self._collections_prefix}_guidelines"
                if self._collections_prefix
                else "guidelines",
                schema=GuidelineVectorDocument,
                embedder_type=embedder_type,
                document_loader=self._vector_document_loader,
            )
            await self._vector_collection.ensure_indexes(
                [VectorCollectionIndex(field="guideline_id")]
            )

        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
            collections_prefix=self._collections_prefix,
        ):
            self._collection = await self._database.get_or_create_collection(
                name=f"{self._collections_prefix}_guidelines"
                if self._collections_prefix
                else "guidelines",
                schema=GuidelineDocument,
                document_loader=self._document_loader,
            )

            self._tag_association_collection = await self._database.get_or_create_collection(
                name=f"{self._collections_prefix}_guideline_tag_associations"
                if self._collections_prefix
                else "guideline_tag_associations",
                schema=GuidelineTagAssociationDocument,
                document_loader=self._association_document_loader,
            )

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        pass

    def _serialize(
        self,
        guideline: Guideline,
    ) -> GuidelineDocument:
        return GuidelineDocument(
            id=ObjectId(guideline.id),
            version=self.VERSION.to_string(),
            creation_utc=guideline.creation_utc.isoformat(),
            last_modified=guideline.last_modified_utc.isoformat(),
            condition=guideline.content.condition,
            action=guideline.content.action,
            description=guideline.content.description,
            title=guideline.title,
            criticality=guideline.criticality.value,
            enabled=guideline.enabled,
            metadata=guideline.metadata,
            composition_mode=(
                guideline.composition_mode.value if guideline.composition_mode else None
            ),
            track=guideline.track,
            labels=list(guideline.labels),
            priority=guideline.priority,
            signals=list(guideline.signals),
        )

    async def _deserialize(
        self,
        guideline_document: GuidelineDocument,
    ) -> Guideline:
        tag_ids = [
            d["tag_id"]
            for d in await self._tag_association_collection.find(
                {"guideline_id": {"$eq": guideline_document["id"]}}
            )
        ]

        composition_mode_str = guideline_document.get("composition_mode")
        composition_mode = CompositionMode(composition_mode_str) if composition_mode_str else None

        return Guideline(
            id=GuidelineId(guideline_document["id"]),
            creation_utc=datetime.fromisoformat(guideline_document["creation_utc"]),
            last_modified_utc=datetime.fromisoformat(guideline_document["last_modified"]),
            content=GuidelineContent(
                condition=guideline_document["condition"],
                action=guideline_document["action"],
                description=guideline_document.get("description", None),
            ),
            title=guideline_document.get("title", None),
            criticality=Criticality(guideline_document["criticality"]),
            enabled=guideline_document["enabled"],
            tags=[TagId(tag_id) for tag_id in tag_ids],
            metadata=guideline_document["metadata"],
            labels=set(guideline_document.get("labels", [])),
            composition_mode=composition_mode,
            track=guideline_document.get("track", True),
            priority=guideline_document.get("priority", 0),
            signals=list(guideline_document.get("signals", [])),
        )

    def _guideline_embedding_content(self, content: GuidelineContent) -> str:
        """Render a guideline's content as the string to embed.

        Treats ``None`` / empty / whitespace-only condition, action, and
        description as absent. The condition+action pair reads as
        ``When {condition}, then {action}``; a lone condition or action gets a
        labeled form; a non-empty description is appended as its own block.
        """
        condition = (content.condition or "").strip()
        action = (content.action or "").strip()
        description = (content.description or "").strip()

        if condition and action:
            head = f"When {condition}, then {action}"
        elif condition:
            head = f"Condition: {condition}"
        elif action:
            head = f"Action: {action}"
        else:
            raise ValueError("Guideline must have at least a condition or an action")

        if description:
            return f"{head}\n\nDescription: {description}"

        return head

    def _list_guideline_contents(self, guideline: Guideline) -> list[str]:
        """The independent strings to embed for a guideline: its rendered
        content followed by each signal as its own vector."""
        return [self._guideline_embedding_content(guideline.content), *guideline.signals]

    async def _insert_vector_documents(self, guideline: Guideline) -> None:
        insertion_tasks = []

        for content in self._list_guideline_contents(guideline):
            doc_id = self._id_generator.generate(xxh3_checksum(f"{guideline.id}{content}"))

            vec_doc = GuidelineVectorDocument(
                id=ObjectId(doc_id),
                guideline_id=ObjectId(guideline.id),
                version=self.VERSION.to_string(),
                content=content,
                checksum=xxh3_checksum(content),
            )

            insertion_tasks.append(self._vector_collection.insert_one(document=vec_doc))

        await async_utils.safe_gather(*insertion_tasks)

    async def _delete_vector_documents(self, guideline_id: GuidelineId) -> None:
        for v_doc in await self._vector_collection.find(
            filters={"guideline_id": {"$eq": guideline_id}}
        ):
            await self._vector_collection.delete_one(filters={"id": {"$eq": v_doc["id"]}})

    @override
    async def create_guideline(
        self,
        condition: str,
        action: Optional[str] = None,
        description: Optional[str] = None,
        title: Optional[str] = None,
        criticality: Optional[Criticality] = None,
        metadata: Mapping[str, JSONSerializable] = {},
        creation_utc: Optional[datetime] = None,
        enabled: bool = True,
        tags: Optional[Sequence[TagId]] = None,
        id: Optional[GuidelineId] = None,
        composition_mode: Optional[CompositionMode] = None,
        track: bool = True,
        labels: Optional[Set[str]] = None,
        priority: int = 0,
        signals: Sequence[str] = [],
    ) -> Guideline:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)
            criticality = criticality or Criticality.MEDIUM

            # Use provided ID or generate one
            if id is not None:
                guideline_id = id

                # Check if guideline with this ID already exists
                existing = await self._collection.find_one(filters={"id": {"$eq": guideline_id}})
                if existing:
                    raise ValueError(f"Guideline with id '{guideline_id}' already exists")
            else:
                guideline_checksum = xxh3_checksum(f"{condition}{action or ''}{enabled}{metadata}")
                guideline_id = GuidelineId(self._id_generator.generate(guideline_checksum))

            guideline = Guideline(
                id=guideline_id,
                creation_utc=creation_utc,
                last_modified_utc=creation_utc,
                content=GuidelineContent(
                    condition=condition,
                    action=action,
                    description=description,
                ),
                title=title,
                criticality=criticality,
                enabled=enabled,
                tags=tags or [],
                metadata=metadata,
                labels=labels or set(),
                composition_mode=composition_mode,
                track=track,
                priority=priority,
                signals=list(signals),
            )

            await self._insert_vector_documents(guideline)

            await self._collection.insert_one(
                document=self._serialize(
                    guideline=guideline,
                )
            )

            for tag_id in tags or []:
                tag_checksum = xxh3_checksum(f"{guideline.id}{tag_id}")

                await self._tag_association_collection.insert_one(
                    document={
                        "id": ObjectId(self._id_generator.generate(tag_checksum)),
                        "version": self.VERSION.to_string(),
                        "creation_utc": creation_utc.isoformat(),
                        "guideline_id": guideline.id,
                        "tag_id": tag_id,
                    }
                )

        return guideline

    @override
    async def list_guidelines(
        self,
        tags: Optional[Sequence[TagId]] = None,
        labels: Optional[Set[str]] = None,
    ) -> Sequence[Guideline]:
        filters: Where = {}

        async with self._lock.reader_lock:
            if tags is not None:
                if len(tags) == 0:
                    guideline_ids = {
                        doc["guideline_id"]
                        for doc in await self._tag_association_collection.find(filters={})
                    }

                    filters = (
                        {"$and": [{"id": {"$ne": id}} for id in guideline_ids]}
                        if guideline_ids
                        else {}
                    )
                else:
                    tag_filters: Where = {"$or": [{"tag_id": {"$eq": tag}} for tag in tags]}
                    tag_associations = await self._tag_association_collection.find(
                        filters=tag_filters
                    )
                    guideline_ids = {assoc["guideline_id"] for assoc in tag_associations}

                    if not guideline_ids:
                        return []

                    filters = {"$or": [{"id": {"$eq": id}} for id in guideline_ids]}

            guidelines = [
                await self._deserialize(d) for d in await self._collection.find(filters=filters)
            ]

            # Filter by labels if specified
            if labels is not None:
                guidelines = [g for g in guidelines if labels.issubset(g.labels)]

            return guidelines

    @override
    async def read_guideline(
        self,
        guideline_id: GuidelineId,
    ) -> Guideline:
        async with self._lock.reader_lock:
            guideline_document = await self._collection.find_one(
                filters={
                    "id": {"$eq": guideline_id},
                }
            )

        if not guideline_document:
            raise ItemNotFoundError(item_id=UniqueId(guideline_id))

        return await self._deserialize(guideline_document=guideline_document)

    @override
    async def delete_guideline(
        self,
        guideline_id: GuidelineId,
    ) -> None:
        async with self._lock.writer_lock:
            await self._delete_vector_documents(guideline_id)

            result = await self._collection.delete_one(
                filters={
                    "id": {"$eq": guideline_id},
                }
            )

            for doc in await self._tag_association_collection.find(
                filters={
                    "guideline_id": {"$eq": guideline_id},
                }
            ):
                await self._tag_association_collection.delete_one(
                    filters={"id": {"$eq": doc["id"]}}
                )

        if not result.deleted_document:
            raise ItemNotFoundError(item_id=UniqueId(guideline_id))

    @override
    async def update_guideline(
        self,
        guideline_id: GuidelineId,
        params: GuidelineUpdateParams,
    ) -> Guideline:
        async with self._lock.writer_lock:
            guideline_document = GuidelineDocument(
                {
                    **({"condition": params["condition"]} if "condition" in params else {}),
                    **({"action": params["action"]} if "action" in params else {}),
                    **({"description": params["description"]} if "description" in params else {}),
                    **({"title": params["title"]} if "title" in params else {}),
                    **(
                        {"criticality": params["criticality"].value}
                        if "criticality" in params
                        else {}
                    ),
                    **({"enabled": params["enabled"]} if "enabled" in params else {}),
                    **(
                        {
                            "composition_mode": (
                                # Note that updating to None is also valid
                                params["composition_mode"].value
                                if params["composition_mode"] is not None
                                else None
                            )
                        }
                        if "composition_mode" in params
                        else {}
                    ),
                    **({"priority": params["priority"]} if "priority" in params else {}),
                    **({"signals": params["signals"]} if "signals" in params else {}),
                    "last_modified": datetime.now(timezone.utc).isoformat(),
                }
            )

            result = await self._collection.update_one(
                filters={"id": {"$eq": guideline_id}},
                params=guideline_document,
            )

            assert result.updated_document

            updated = await self._deserialize(guideline_document=result.updated_document)

            # Re-sync the embedded vectors if any embedded field changed
            # (condition / action / description / signals).
            if any(key in params for key in ("condition", "action", "description", "signals")):
                await self._delete_vector_documents(guideline_id)
                await self._insert_vector_documents(updated)

        return updated

    @override
    async def find_guideline(
        self,
        guideline_content: GuidelineContent,
    ) -> Guideline:
        async with self._lock.reader_lock:
            filters = {
                "condition": {"$eq": guideline_content.condition},
                **(
                    {"action": {"$eq": guideline_content.action}}
                    if guideline_content.action
                    else {}
                ),
            }

            guideline_document = await self._collection.find_one(filters=cast(Where, filters))

        if not guideline_document:
            raise ItemNotFoundError(
                item_id=UniqueId(f"{guideline_content.condition}{guideline_content.action}")
            )

        return await self._deserialize(guideline_document=guideline_document)

    @override
    async def find_relevant_guidelines(
        self,
        query: str,
        available_guidelines: Sequence[Guideline],
        max_count: int,
    ) -> Sequence[GuidelineRelevanceResult]:
        if not available_guidelines:
            return []

        guidelines_by_id = {g.id: g for g in available_guidelines}

        async with self._lock.reader_lock:
            queries = await query_chunks(query, self._embedder)
            filters: Where = {"guideline_id": {"$in": [str(g.id) for g in available_guidelines]}}

            tasks = [
                self._vector_collection.find_similar_documents(
                    filters=filters,
                    query=q,
                    k=calculate_min_vectors_for_max_item_count(
                        items=available_guidelines,
                        count_item_vectors=lambda g: len(self._list_guideline_contents(g)),
                        max_items_to_return=max_count,
                    ),
                    hints={"tag": "guidelines"},
                )
                for q in queries
            ]

        all_sdocs = chain.from_iterable(await async_utils.safe_gather(*tasks))

        # Dedupe by guideline, keeping the closest matching vector per guideline.
        unique_sdocs: dict[str, SimilarDocumentResult[GuidelineVectorDocument]] = {}

        for similar_doc in all_sdocs:
            guideline_id = similar_doc.document["guideline_id"]
            if (
                guideline_id not in unique_sdocs
                or unique_sdocs[guideline_id].distance > similar_doc.distance
            ):
                unique_sdocs[guideline_id] = similar_doc

        top_results = sorted(unique_sdocs.values(), key=lambda r: r.distance)[:max_count]

        return [
            GuidelineRelevanceResult(
                guideline=guidelines_by_id[GuidelineId(r.document["guideline_id"])],
                score=1.0 - r.distance,
            )
            for r in top_results
            if GuidelineId(r.document["guideline_id"]) in guidelines_by_id
        ]

    @override
    async def upsert_tag(
        self,
        guideline_id: GuidelineId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> bool:
        async with self._lock.writer_lock:
            guideline = await self.read_guideline(guideline_id)

            if tag_id in guideline.tags:
                return False

            creation_utc = creation_utc or datetime.now(timezone.utc)

            association_checksum = xxh3_checksum(f"{guideline.id}{tag_id}")

            association_document: GuidelineTagAssociationDocument = {
                "id": ObjectId(self._id_generator.generate(association_checksum)),
                "version": self.VERSION.to_string(),
                "creation_utc": creation_utc.isoformat(),
                "guideline_id": GuidelineId(guideline_id),
                "tag_id": tag_id,
            }

            _ = await self._tag_association_collection.insert_one(document=association_document)

            guideline_document = await self._collection.find_one({"id": {"$eq": guideline_id}})

        if not guideline_document:
            raise ItemNotFoundError(item_id=UniqueId(guideline_id))

        return True

    @override
    async def remove_tag(
        self,
        guideline_id: GuidelineId,
        tag_id: TagId,
    ) -> None:
        async with self._lock.writer_lock:
            delete_result = await self._tag_association_collection.delete_one(
                {
                    "guideline_id": {"$eq": guideline_id},
                    "tag_id": {"$eq": tag_id},
                }
            )

            if delete_result.deleted_count == 0:
                raise ItemNotFoundError(item_id=UniqueId(tag_id))

            guideline_document = await self._collection.find_one({"id": {"$eq": guideline_id}})

        if not guideline_document:
            raise ItemNotFoundError(item_id=UniqueId(guideline_id))

    @override
    async def set_metadata(
        self,
        guideline_id: GuidelineId,
        key: str,
        value: JSONSerializable,
    ) -> Guideline:
        async with self._lock.writer_lock:
            guideline_document = await self._collection.find_one({"id": {"$eq": guideline_id}})

            if not guideline_document:
                raise ItemNotFoundError(item_id=UniqueId(guideline_id))

            updated_metadata = {**guideline_document["metadata"], key: value}

            result = await self._collection.update_one(
                filters={"id": {"$eq": guideline_id}},
                params={
                    "metadata": updated_metadata,
                    "last_modified": datetime.now(timezone.utc).isoformat(),
                },
            )

        assert result.updated_document

        return await self._deserialize(guideline_document=result.updated_document)

    @override
    async def unset_metadata(
        self,
        guideline_id: GuidelineId,
        key: str,
    ) -> Guideline:
        async with self._lock.writer_lock:
            guideline_document = await self._collection.find_one({"id": {"$eq": guideline_id}})

            if not guideline_document:
                raise ItemNotFoundError(item_id=UniqueId(guideline_id))

            updated_metadata = {k: v for k, v in guideline_document["metadata"].items() if k != key}

            result = await self._collection.update_one(
                filters={"id": {"$eq": guideline_id}},
                params={
                    "metadata": updated_metadata,
                    "last_modified": datetime.now(timezone.utc).isoformat(),
                },
            )

        assert result.updated_document

        return await self._deserialize(guideline_document=result.updated_document)

    @override
    async def upsert_labels(
        self,
        guideline_id: GuidelineId,
        labels: Set[str],
    ) -> Guideline:
        async with self._lock.writer_lock:
            guideline_document = await self._collection.find_one({"id": {"$eq": guideline_id}})

            if not guideline_document:
                raise ItemNotFoundError(item_id=UniqueId(guideline_id))

            current_labels = set(guideline_document.get("labels", []))
            updated_labels = list(current_labels | labels)

            result = await self._collection.update_one(
                filters={"id": {"$eq": guideline_id}},
                params={
                    "labels": updated_labels,
                },
            )

        assert result.updated_document

        return await self._deserialize(guideline_document=result.updated_document)

    @override
    async def remove_labels(
        self,
        guideline_id: GuidelineId,
        labels: Set[str],
    ) -> Guideline:
        async with self._lock.writer_lock:
            guideline_document = await self._collection.find_one({"id": {"$eq": guideline_id}})

            if not guideline_document:
                raise ItemNotFoundError(item_id=UniqueId(guideline_id))

            current_labels = set(guideline_document.get("labels", []))
            updated_labels = list(current_labels - labels)

            result = await self._collection.update_one(
                filters={"id": {"$eq": guideline_id}},
                params={
                    "labels": updated_labels,
                },
            )

        assert result.updated_document

        return await self._deserialize(guideline_document=result.updated_document)


class CompositeGuidelineStore(GuidelineStore):
    def __init__(
        self,
        writable_store: GuidelineStore,
        readable_stores: Sequence[GuidelineStore],
    ) -> None:
        self._writable_store = writable_store
        self._readable_stores = readable_stores
        self._all_stores: Sequence[GuidelineStore] = [writable_store, *readable_stores]

    @override
    async def create_guideline(
        self,
        condition: str,
        action: Optional[str] = None,
        description: Optional[str] = None,
        title: Optional[str] = None,
        criticality: Optional[Criticality] = None,
        metadata: Mapping[str, JSONSerializable] = {},
        creation_utc: Optional[datetime] = None,
        enabled: bool = True,
        tags: Optional[Sequence[TagId]] = None,
        id: Optional[GuidelineId] = None,
        composition_mode: Optional[CompositionMode] = None,
        track: bool = True,
        labels: Optional[Set[str]] = None,
        priority: int = 0,
        signals: Sequence[str] = [],
    ) -> Guideline:
        return await self._writable_store.create_guideline(
            condition=condition,
            action=action,
            description=description,
            title=title,
            criticality=criticality,
            metadata=metadata,
            creation_utc=creation_utc,
            enabled=enabled,
            tags=tags,
            id=id,
            composition_mode=composition_mode,
            track=track,
            labels=labels,
            priority=priority,
            signals=signals,
        )

    @override
    async def list_guidelines(
        self,
        tags: Optional[Sequence[TagId]] = None,
        labels: Optional[Set[str]] = None,
    ) -> Sequence[Guideline]:
        results = await safe_gather(
            *[store.list_guidelines(tags=tags, labels=labels) for store in self._all_stores]
        )
        return list(chain.from_iterable(results))

    @override
    async def find_relevant_guidelines(
        self,
        query: str,
        available_guidelines: Sequence[Guideline],
        max_count: int,
    ) -> Sequence[GuidelineRelevanceResult]:
        results = await safe_gather(
            *[
                store.find_relevant_guidelines(query, available_guidelines, max_count)
                for store in self._all_stores
            ]
        )
        merged = list(chain.from_iterable(results))
        return sorted(merged, key=lambda r: r.score, reverse=True)[:max_count]

    @override
    async def read_guideline(
        self,
        guideline_id: GuidelineId,
    ) -> Guideline:
        results = await safe_gather(
            *[try_or_none(store.read_guideline(guideline_id)) for store in self._all_stores]
        )
        result = next((r for r in results if r is not None), None)
        if result is None:
            raise ItemNotFoundError(item_id=UniqueId(guideline_id))
        return result

    @override
    async def delete_guideline(
        self,
        guideline_id: GuidelineId,
    ) -> None:
        return await self._writable_store.delete_guideline(guideline_id)

    @override
    async def update_guideline(
        self,
        guideline_id: GuidelineId,
        params: GuidelineUpdateParams,
    ) -> Guideline:
        return await self._writable_store.update_guideline(guideline_id, params)

    @override
    async def find_guideline(
        self,
        guideline_content: GuidelineContent,
    ) -> Guideline:
        results = await safe_gather(
            *[try_or_none(store.find_guideline(guideline_content)) for store in self._all_stores]
        )
        result = next((r for r in results if r is not None), None)
        if result is None:
            raise ItemNotFoundError(
                item_id=UniqueId(f"{guideline_content.condition}{guideline_content.action}")
            )
        return result

    @override
    async def upsert_tag(
        self,
        guideline_id: GuidelineId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> bool:
        return await self._writable_store.upsert_tag(guideline_id, tag_id, creation_utc)

    @override
    async def remove_tag(
        self,
        guideline_id: GuidelineId,
        tag_id: TagId,
    ) -> None:
        return await self._writable_store.remove_tag(guideline_id, tag_id)

    @override
    async def set_metadata(
        self,
        guideline_id: GuidelineId,
        key: str,
        value: JSONSerializable,
    ) -> Guideline:
        return await self._writable_store.set_metadata(guideline_id, key, value)

    @override
    async def unset_metadata(
        self,
        guideline_id: GuidelineId,
        key: str,
    ) -> Guideline:
        return await self._writable_store.unset_metadata(guideline_id, key)

    @override
    async def upsert_labels(
        self,
        guideline_id: GuidelineId,
        labels: Set[str],
    ) -> Guideline:
        return await self._writable_store.upsert_labels(guideline_id, labels)

    @override
    async def remove_labels(
        self,
        guideline_id: GuidelineId,
        labels: Set[str],
    ) -> Guideline:
        return await self._writable_store.remove_labels(guideline_id, labels)
