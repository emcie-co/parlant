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

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import chain
from typing import Awaitable, Callable, Mapping, NewType, Optional, Sequence, cast
from typing_extensions import override, TypedDict, Self, Required

from parlant.core.async_utils import ReaderWriterLock, safe_gather
from parlant.core.common import JSONSerializable, md5_checksum
from parlant.core.common import ItemNotFoundError, UniqueId, Version, IdGenerator, to_json_dict
from parlant.core.guidelines import GuidelineId
from parlant.core.nlp.embedding import Embedder, EmbedderFactory
from parlant.core.persistence.common import (
    ObjectId,
    Where,
)
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentDatabase,
    DocumentCollection,
)
from parlant.core.persistence.document_database_helper import (
    DocumentMigrationHelper,
    DocumentStoreMigrationHelper,
)
from parlant.core.persistence.vector_database import (
    VectorCollection,
    VectorDatabase,
    BaseDocument as VectorDocument,
)
from parlant.core.persistence.vector_database_helper import (
    VectorDocumentMigrationHelper,
    VectorDocumentStoreMigrationHelper,
    query_chunks,
)
from parlant.core.tags import TagId
from parlant.core.tools import ToolId

JourneyId = NewType("JourneyId", str)
JourneyNodeId = NewType("JourneyNodeId", str)
JourneyEdgeId = NewType("JourneyEdgeId", str)


@dataclass(frozen=True)
class JourneyNode:
    id: JourneyNodeId
    creation_utc: datetime
    action: Optional[str]
    tools: Sequence[ToolId]
    metadata: Mapping[str, JSONSerializable]

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass(frozen=True)
class JourneyEdge:
    id: JourneyEdgeId
    creation_utc: datetime
    source: JourneyNodeId
    target: JourneyNodeId
    condition: Optional[str]
    metadata: Mapping[str, JSONSerializable]

    def __hash__(self) -> int:
        return hash(self.id)


@dataclass(frozen=True)
class Journey:
    id: JourneyId
    creation_utc: datetime
    description: str
    conditions: Sequence[GuidelineId]
    title: str
    root_id: JourneyNodeId
    tags: Sequence[TagId]

    def __hash__(self) -> int:
        return hash(self.id)


class JourneyUpdateParams(TypedDict, total=False):
    title: str
    description: str


class JourneyNodeUpdateParams(TypedDict, total=False):
    action: Optional[str]
    tools: Optional[Sequence[ToolId]]


class JourneyEdgeUpdateParams(TypedDict, total=False):
    condition: Optional[str]


class JourneyStore(ABC):
    END_NODE_ID = JourneyNodeId("end")

    DEFAULT_ROOT_ACTION = (
        "<<JOURNEY ROOT: start the journey at the appropriate step based on the context>>"
    )

    @abstractmethod
    async def create_journey(
        self,
        title: str,
        description: str,
        conditions: Sequence[GuidelineId],
        creation_utc: Optional[datetime] = None,
        tags: Optional[Sequence[TagId]] = None,
    ) -> Journey: ...

    @abstractmethod
    async def list_journeys(
        self,
        tags: Optional[Sequence[TagId]] = None,
        condition: Optional[GuidelineId] = None,
    ) -> Sequence[Journey]: ...

    @abstractmethod
    async def read_journey(
        self,
        journey_id: JourneyId,
    ) -> Journey: ...

    @abstractmethod
    async def update_journey(
        self,
        journey_id: JourneyId,
        params: JourneyUpdateParams,
    ) -> Journey: ...

    @abstractmethod
    async def delete_journey(
        self,
        journey_id: JourneyId,
    ) -> None: ...

    @abstractmethod
    async def add_condition(
        self,
        journey_id: JourneyId,
        condition: GuidelineId,
    ) -> bool: ...

    @abstractmethod
    async def remove_condition(
        self,
        journey_id: JourneyId,
        condition: GuidelineId,
    ) -> bool: ...

    @abstractmethod
    async def upsert_tag(
        self,
        journey_id: JourneyId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> bool: ...

    @abstractmethod
    async def remove_tag(
        self,
        journey_id: JourneyId,
        tag_id: TagId,
    ) -> None: ...

    @abstractmethod
    async def find_relevant_journeys(
        self,
        query: str,
        available_journeys: Sequence[Journey],
        max_journeys: int = 5,
    ) -> Sequence[Journey]: ...

    @abstractmethod
    async def create_node(
        self,
        journey_id: JourneyId,
        action: Optional[str],
        tools: Sequence[ToolId],
    ) -> JourneyNode: ...

    @abstractmethod
    async def read_node(
        self,
        node_id: JourneyNodeId,
    ) -> JourneyNode: ...

    @abstractmethod
    async def update_node(
        self,
        node_id: JourneyNodeId,
        params: JourneyNodeUpdateParams,
    ) -> JourneyNode: ...

    @abstractmethod
    async def delete_node(
        self,
        node_id: JourneyNodeId,
    ) -> None: ...

    @abstractmethod
    async def list_nodes(
        self,
        journey_id: JourneyId,
    ) -> Sequence[JourneyNode]: ...

    @abstractmethod
    async def set_node_metadata(
        self,
        node_id: JourneyNodeId,
        key: str,
        value: JSONSerializable,
    ) -> JourneyNode: ...

    @abstractmethod
    async def unset_node_metadata(
        self,
        node_id: JourneyNodeId,
        key: str,
    ) -> JourneyNode: ...

    @abstractmethod
    async def create_edge(
        self,
        journey_id: JourneyId,
        source: JourneyNodeId,
        target: JourneyNodeId,
        condition: Optional[str],
    ) -> JourneyEdge: ...

    @abstractmethod
    async def read_edge(
        self,
        edge_id: JourneyNodeId,
    ) -> JourneyEdge: ...

    @abstractmethod
    async def update_edge(
        self,
        edge_id: JourneyNodeId,
        params: JourneyEdgeUpdateParams,
    ) -> JourneyEdge: ...

    @abstractmethod
    async def list_edges(
        self,
        journey_id: JourneyId,
        node_id: Optional[JourneyNodeId] = None,
    ) -> Sequence[JourneyEdge]: ...

    @abstractmethod
    async def delete_edge(
        self,
        edge_id: JourneyEdgeId,
    ) -> None: ...

    @abstractmethod
    async def set_edge_metadata(
        self,
        edge_id: JourneyEdgeId,
        key: str,
        value: JSONSerializable,
    ) -> JourneyEdge: ...

    @abstractmethod
    async def unset_edge_metadata(
        self,
        edge_id: JourneyEdgeId,
        key: str,
    ) -> JourneyEdge: ...


class JourneyDocument_v0_1_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    title: str
    description: str


class JourneyDocument_v0_2_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    content: str
    checksum: Required[str]
    title: str
    description: str


class JourneyVectorDocument(TypedDict, total=False):
    id: ObjectId
    journey_id: JourneyId
    version: Version.String
    content: str
    checksum: Required[str]


class JourneyDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    title: str
    description: str
    root_id: JourneyNodeId


class JourneyConditionAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    journey_id: JourneyId
    condition: GuidelineId


class JourneyNodeAssociationDocument(TypedDict, total=False):
    id: ObjectId
    node_id: JourneyNodeId
    version: Version.String
    creation_utc: str
    journey_id: JourneyId
    action: Optional[str]
    tools: Sequence[ToolId]
    metadata: Mapping[str, JSONSerializable]


class JourneyEdgeAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    journey_id: JourneyId
    condition: Optional[str]
    source: JourneyNodeId
    target: JourneyNodeId
    metadata: Mapping[str, JSONSerializable]


class JourneyTagAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    journey_id: JourneyId
    tag_id: TagId


class JourneyVectorStore(JourneyStore):
    VERSION = Version.from_string("0.3.0")

    def __init__(
        self,
        id_generator: IdGenerator,
        vector_db: VectorDatabase,
        document_db: DocumentDatabase,
        embedder_type_provider: Callable[[], Awaitable[type[Embedder]]],
        embedder_factory: EmbedderFactory,
        allow_migration: bool = True,
    ):
        self._id_generator = id_generator

        self._vector_db = vector_db
        self._document_db = document_db
        self._vector_collection: VectorCollection[JourneyVectorDocument]
        self._collection: DocumentCollection[JourneyDocument]
        self._node_association_collection: DocumentCollection[JourneyNodeAssociationDocument]
        self._edge_association_collection: DocumentCollection[JourneyEdgeAssociationDocument]

        self._tag_association_collection: DocumentCollection[JourneyTagAssociationDocument]
        self._condition_association_collection: DocumentCollection[
            JourneyConditionAssociationDocument
        ]

        self._allow_migration = allow_migration

        self._embedder_factory = embedder_factory
        self._embedder_type_provider = embedder_type_provider
        self._embedder: Embedder

        self._lock = ReaderWriterLock()

    async def _vector_document_loader(self, doc: VectorDocument) -> Optional[JourneyVectorDocument]:
        async def v0_1_0_to_v0_3_0(doc: VectorDocument) -> Optional[VectorDocument]:
            raise Exception(
                "This code should not be reached! Please run the 'parlant-prepare-migration' script."
            )

        return await VectorDocumentMigrationHelper[JourneyVectorDocument](
            self,
            {
                "0.1.0": v0_1_0_to_v0_3_0,
                "0.2.0": v0_1_0_to_v0_3_0,
            },
        ).migrate(doc)

    async def _document_loader(self, doc: BaseDocument) -> Optional[JourneyDocument]:
        async def v0_1_0_to_v0_3_0(doc: BaseDocument) -> Optional[BaseDocument]:
            raise Exception(
                "This code should not be reached! Please run the 'parlant-prepare-migration' script."
            )

        return await DocumentMigrationHelper[JourneyDocument](
            self,
            {
                "0.1.0": v0_1_0_to_v0_3_0,
                "0.2.0": v0_1_0_to_v0_3_0,
            },
        ).migrate(doc)

    async def _tag_association_loader(
        self, doc: BaseDocument
    ) -> Optional[JourneyTagAssociationDocument]:
        async def v0_1_0_to_v0_3_0(doc: BaseDocument) -> Optional[BaseDocument]:
            raise Exception(
                "This code should not be reached! Please run the 'parlant-prepare-migration' script."
            )

        return await DocumentMigrationHelper[JourneyTagAssociationDocument](
            self,
            {
                "0.1.0": v0_1_0_to_v0_3_0,
            },
        ).migrate(doc)

    async def _condition_association_loader(
        self, doc: BaseDocument
    ) -> Optional[JourneyConditionAssociationDocument]:
        async def v0_1_0_to_v0_3_0(doc: BaseDocument) -> Optional[BaseDocument]:
            raise Exception(
                "This code should not be reached! Please run the 'parlant-prepare-migration' script."
            )

        return await DocumentMigrationHelper[JourneyConditionAssociationDocument](
            self,
            {
                "0.1.0": v0_1_0_to_v0_3_0,
            },
        ).migrate(doc)

    async def _node_association_loader(
        self, doc: BaseDocument
    ) -> Optional[JourneyNodeAssociationDocument]:
        async def v0_1_0_to_v0_3_0(doc: BaseDocument) -> Optional[BaseDocument]:
            raise Exception(
                "This code should not be reached! Please run the 'parlant-prepare-migration' script."
            )

        return await DocumentMigrationHelper[JourneyNodeAssociationDocument](
            self,
            {
                "0.1.0": v0_1_0_to_v0_3_0,
            },
        ).migrate(doc)

    async def _edge_association_loader(
        self, doc: BaseDocument
    ) -> Optional[JourneyEdgeAssociationDocument]:
        async def v0_1_0_to_v0_3_0(doc: BaseDocument) -> Optional[BaseDocument]:
            raise Exception(
                "This code should not be reached! Please run the 'parlant-prepare-migration' script."
            )

        return await DocumentMigrationHelper[JourneyEdgeAssociationDocument](
            self,
            {
                "0.1.0": v0_1_0_to_v0_3_0,
            },
        ).migrate(doc)

    async def __aenter__(self) -> Self:
        embedder_type = await self._embedder_type_provider()
        self._embedder = self._embedder_factory.create_embedder(embedder_type)

        async with VectorDocumentStoreMigrationHelper(
            store=self,
            database=self._vector_db,
            allow_migration=self._allow_migration,
        ):
            self._vector_collection = await self._vector_db.get_or_create_collection(
                name="journeys",
                schema=JourneyVectorDocument,
                embedder_type=embedder_type,
                document_loader=self._vector_document_loader,
            )

        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._document_db,
            allow_migration=self._allow_migration,
        ):
            self._collection = await self._document_db.get_or_create_collection(
                name="journeys",
                schema=JourneyDocument,
                document_loader=self._document_loader,
            )

            self._node_association_collection = await self._document_db.get_or_create_collection(
                name="journey_nodes",
                schema=JourneyNodeAssociationDocument,
                document_loader=self._node_association_loader,
            )

            self._edge_association_collection = await self._document_db.get_or_create_collection(
                name="journey_edges",
                schema=JourneyEdgeAssociationDocument,
                document_loader=self._edge_association_loader,
            )

            self._tag_association_collection = await self._document_db.get_or_create_collection(
                name="journey_tags",
                schema=JourneyTagAssociationDocument,
                document_loader=self._tag_association_loader,
            )

            self._condition_association_collection = (
                await self._document_db.get_or_create_collection(
                    name="journey_conditions",
                    schema=JourneyConditionAssociationDocument,
                    document_loader=self._condition_association_loader,
                )
            )

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> bool:
        return False

    def _serialize(
        self,
        journey: Journey,
    ) -> JourneyDocument:
        return JourneyDocument(
            id=ObjectId(journey.id),
            version=self.VERSION.to_string(),
            creation_utc=journey.creation_utc.isoformat(),
            title=journey.title,
            description=journey.description,
            root_id=journey.root_id,
        )

    async def _deserialize(self, doc: JourneyDocument) -> Journey:
        tags = [
            d["tag_id"]
            for d in await self._tag_association_collection.find({"journey_id": {"$eq": doc["id"]}})
        ]

        conditions = [
            d["condition"]
            for d in await self._condition_association_collection.find(
                {"journey_id": {"$eq": doc["id"]}}
            )
        ]

        return Journey(
            id=JourneyId(doc["id"]),
            creation_utc=datetime.fromisoformat(doc["creation_utc"]),
            conditions=conditions,
            title=doc["title"],
            description=doc["description"],
            root_id=JourneyNodeId(doc["root_id"]),
            tags=tags,
        )

    def _serialize_node(
        self,
        node: JourneyNode,
        journey_id: JourneyId,
    ) -> JourneyNodeAssociationDocument:
        id_checksum = md5_checksum(f"{journey_id}{node.id}")

        return JourneyNodeAssociationDocument(
            id=ObjectId(self._id_generator.generate(id_checksum)),
            node_id=node.id,
            version=self.VERSION.to_string(),
            creation_utc=datetime.now(timezone.utc).isoformat(),
            journey_id=journey_id,
            action=node.action,
            tools=node.tools,
            metadata=node.metadata,
        )

    def _deserialize_node(self, doc: JourneyNodeAssociationDocument) -> JourneyNode:
        return JourneyNode(
            id=JourneyNodeId(doc["node_id"]),
            creation_utc=datetime.fromisoformat(doc["creation_utc"]),
            action=doc["action"],
            tools=doc["tools"],
            metadata=doc["metadata"],
        )

    def _serialize_edge(
        self,
        edge: JourneyEdge,
        journey_id: JourneyId,
    ) -> JourneyEdgeAssociationDocument:
        return JourneyEdgeAssociationDocument(
            id=ObjectId(edge.id),
            version=self.VERSION.to_string(),
            creation_utc=datetime.now(timezone.utc).isoformat(),
            journey_id=journey_id,
            condition=edge.condition,
            source=edge.source,
            target=edge.target,
            metadata=edge.metadata,
        )

    def _deserialize_edge(self, doc: JourneyEdgeAssociationDocument) -> JourneyEdge:
        return JourneyEdge(
            id=JourneyEdgeId(doc["id"]),
            creation_utc=datetime.fromisoformat(doc["creation_utc"]),
            source=JourneyNodeId(doc["source"]),
            target=JourneyNodeId(doc["target"]),
            condition=doc["condition"],
            metadata=doc["metadata"],
        )

    @staticmethod
    def assemble_content(
        title: str,
        description: str,
        nodes: Sequence[JourneyNode],
        edges: Sequence[JourneyEdge],
    ) -> str:
        # TODO: Research is needed to determine the best way to assemble journey content,
        # including how many vectors to generate and what content each vector should contain.
        return f"{title}\n{description}\nNodes: {', '.join(n.action for n in nodes if n.action)}\nEdges: {', '.join(e.condition for e in edges if e.condition)}"

    @override
    async def create_journey(
        self,
        title: str,
        description: str,
        conditions: Sequence[GuidelineId],
        creation_utc: Optional[datetime] = None,
        tags: Optional[Sequence[TagId]] = None,
    ) -> Journey:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)

            journey_checksum = md5_checksum(f"{title}{description}{conditions}")

            journey_id = JourneyId(self._id_generator.generate(journey_checksum))
            journey_root_id = JourneyNodeId(self._id_generator.generate(f"{journey_id}root"))

            root = JourneyNode(
                id=journey_root_id,
                creation_utc=creation_utc,
                action=None,
                tools=[],
                metadata={},
            )

            await self._node_association_collection.insert_one(
                document=self._serialize_node(root, journey_id)
            )

            journey = Journey(
                id=journey_id,
                creation_utc=creation_utc,
                conditions=conditions,
                title=title,
                description=description,
                root_id=journey_root_id,
                tags=tags or [],
            )

            content = self.assemble_content(
                title=title,
                description=description,
                nodes=[],
                edges=[],
            )

            await self._collection.insert_one(document=self._serialize(journey))
            await self._vector_collection.insert_one(
                document={
                    "id": ObjectId(self._id_generator.generate(md5_checksum(content))),
                    "version": self.VERSION.to_string(),
                    "journey_id": journey.id,
                    "content": content,
                    "checksum": md5_checksum(content),
                }
            )

            for tag_id in tags or []:
                tag_checksum = md5_checksum(f"{journey.id}{tag_id}")

                await self._tag_association_collection.insert_one(
                    document={
                        "id": ObjectId(self._id_generator.generate(tag_checksum)),
                        "version": self.VERSION.to_string(),
                        "creation_utc": creation_utc.isoformat(),
                        "journey_id": journey.id,
                        "tag_id": tag_id,
                    }
                )

            for condition in conditions:
                condition_checksum = md5_checksum(f"{journey.id}{condition}")

                await self._condition_association_collection.insert_one(
                    document={
                        "id": ObjectId(self._id_generator.generate(condition_checksum)),
                        "version": self.VERSION.to_string(),
                        "creation_utc": creation_utc.isoformat(),
                        "journey_id": journey.id,
                        "condition": condition,
                    }
                )

        return journey

    @override
    async def read_journey(self, journey_id: JourneyId) -> Journey:
        async with self._lock.reader_lock:
            doc = await self._collection.find_one({"id": {"$eq": journey_id}})

        if not doc:
            raise ItemNotFoundError(item_id=UniqueId(journey_id))

        return await self._deserialize(doc)

    @override
    async def update_journey(
        self,
        journey_id: JourneyId,
        params: JourneyUpdateParams,
    ) -> Journey:
        async with self._lock.writer_lock:
            doc = await self._collection.find_one({"id": {"$eq": journey_id}})

            if not doc:
                raise ItemNotFoundError(item_id=UniqueId(journey_id))

            nodes = await self.list_nodes(journey_id=journey_id)
            edges = await self.list_edges(journey_id=journey_id)

            updated = {**doc, **params}

            content = self.assemble_content(
                title=cast(str, updated["title"]),
                description=cast(str, updated["description"]),
                nodes=nodes,
                edges=edges,
            )

            result = await self._collection.update_one(
                filters={"id": {"$eq": journey_id}},
                params=cast(JourneyDocument, to_json_dict(updated)),
            )

            await self._vector_collection.update_one(
                filters={"journey_id": {"$eq": journey_id}},
                params={
                    "content": content,
                    "checksum": md5_checksum(content),
                },
            )

        assert result.updated_document

        return await self._deserialize(result.updated_document)

    @override
    async def list_journeys(
        self,
        tags: Optional[Sequence[TagId]] = None,
        condition: Optional[GuidelineId] = None,
    ) -> Sequence[Journey]:
        filters: Where = {}
        journey_ids: set[JourneyId] = set()
        condition_journey_ids: set[JourneyId] = set()

        async with self._lock.reader_lock:
            if tags is not None:
                if len(tags) == 0:
                    journey_ids = {
                        doc["journey_id"]
                        for doc in await self._tag_association_collection.find(filters={})
                    }

                    if not journey_ids:
                        filters = {}

                    elif len(journey_ids) == 1:
                        filters = {"id": {"$ne": journey_ids.pop()}}

                    else:
                        filters = {"$and": [{"id": {"$ne": id}} for id in journey_ids]}

                else:
                    tag_filters: Where = {"$or": [{"tag_id": {"$eq": tag}} for tag in tags]}
                    tag_associations = await self._tag_association_collection.find(
                        filters=tag_filters
                    )
                    journey_ids = {assoc["journey_id"] for assoc in tag_associations}

                    if not journey_ids:
                        return []

                    if len(journey_ids) == 1:
                        filters = {"id": {"$eq": journey_ids.pop()}}

                    else:
                        filters = {"$or": [{"id": {"$eq": id}} for id in journey_ids]}

            if condition is not None:
                condition_journey_ids = {
                    c_doc["journey_id"]
                    for c_doc in await self._condition_association_collection.find(
                        filters={"condition": {"$eq": condition}}
                    )
                }

                if not journey_ids:
                    journey_ids = condition_journey_ids
                else:
                    journey_ids.intersection_update(condition_journey_ids)

            if journey_ids:
                filters = {"$or": [{"id": {"$eq": id}} for id in journey_ids]}

            return [
                await self._deserialize(d) for d in await self._collection.find(filters=filters)
            ]

    @override
    async def delete_journey(
        self,
        journey_id: JourneyId,
    ) -> None:
        async with self._lock.writer_lock:
            for n_doc in await self._node_association_collection.find(
                filters={
                    "journey_id": {"$eq": journey_id},
                }
            ):
                await self._node_association_collection.delete_one(
                    filters={"id": {"$eq": n_doc["id"]}}
                )

            for e_doc in await self._edge_association_collection.find(
                filters={
                    "journey_id": {"$eq": journey_id},
                }
            ):
                await self._edge_association_collection.delete_one(
                    filters={"id": {"$eq": e_doc["id"]}}
                )

            for c_doc in await self._condition_association_collection.find(
                filters={
                    "journey_id": {"$eq": journey_id},
                }
            ):
                await self._condition_association_collection.delete_one(
                    filters={"id": {"$eq": c_doc["id"]}}
                )

            for t_doc in await self._tag_association_collection.find(
                filters={
                    "journey_id": {"$eq": journey_id},
                }
            ):
                await self._tag_association_collection.delete_one(
                    filters={"id": {"$eq": t_doc["id"]}}
                )

            result = await self._collection.delete_one({"id": {"$eq": journey_id}})

        if result.deleted_count == 0:
            raise ItemNotFoundError(item_id=UniqueId(journey_id))

    @override
    async def add_condition(
        self,
        journey_id: JourneyId,
        condition: GuidelineId,
    ) -> bool:
        async with self._lock.writer_lock:
            journey = await self.read_journey(journey_id)

            if condition in journey.conditions:
                return False

            condition_checksum = md5_checksum(f"{journey_id}{condition}")

            await self._condition_association_collection.insert_one(
                document={
                    "id": ObjectId(self._id_generator.generate(condition_checksum)),
                    "version": self.VERSION.to_string(),
                    "creation_utc": datetime.now(timezone.utc).isoformat(),
                    "journey_id": journey_id,
                    "condition": condition,
                }
            )

            return True

    @override
    async def remove_condition(
        self,
        journey_id: JourneyId,
        condition: GuidelineId,
    ) -> bool:
        async with self._lock.writer_lock:
            await self._condition_association_collection.delete_one(
                filters={
                    "journey_id": {"$eq": journey_id},
                    "condition": {"$eq": condition},
                }
            )

            return True

    @override
    async def upsert_tag(
        self,
        journey_id: JourneyId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> bool:
        creation_utc = creation_utc or datetime.now(timezone.utc)

        async with self._lock.writer_lock:
            journey = await self.read_journey(journey_id)

            if tag_id in journey.tags:
                return False

            association_checksum = md5_checksum(f"{journey_id}{tag_id}")

            association_document: JourneyTagAssociationDocument = {
                "id": ObjectId(self._id_generator.generate(association_checksum)),
                "version": self.VERSION.to_string(),
                "creation_utc": creation_utc.isoformat(),
                "journey_id": journey_id,
                "tag_id": tag_id,
            }

            _ = await self._tag_association_collection.insert_one(document=association_document)

        return True

    @override
    async def remove_tag(
        self,
        journey_id: JourneyId,
        tag_id: TagId,
    ) -> None:
        async with self._lock.writer_lock:
            delete_result = await self._tag_association_collection.delete_one(
                {
                    "journey_id": {"$eq": journey_id},
                    "tag_id": {"$eq": tag_id},
                }
            )

            if delete_result.deleted_count == 0:
                raise ItemNotFoundError(item_id=UniqueId(tag_id))

    @override
    async def find_relevant_journeys(
        self,
        query: str,
        available_journeys: Sequence[Journey],
        max_journeys: int = 5,
    ) -> Sequence[Journey]:
        if not available_journeys:
            return []

        async with self._lock.reader_lock:
            queries = await query_chunks(query, self._embedder)
            filters: Where = {"journey_id": {"$in": [str(j.id) for j in available_journeys]}}

            tasks = [
                self._vector_collection.find_similar_documents(
                    filters=filters,
                    query=q,
                    k=max_journeys,
                )
                for q in queries
            ]

        all_results = chain.from_iterable(await safe_gather(*tasks))
        unique_results = list(set(all_results))
        top_vectors = sorted(unique_results, key=lambda r: r.distance)[:max_journeys]

        return [
            await self._deserialize(doc)
            for doc in await self._collection.find(
                filters={"id": {"$in": [r.document["journey_id"] for r in top_vectors]}}
            )
        ]

    @override
    async def create_node(
        self,
        journey_id: JourneyId,
        action: Optional[str],
        tools: Sequence[ToolId],
        creation_utc: Optional[datetime] = None,
    ) -> JourneyNode:
        creation_utc = creation_utc or datetime.now(timezone.utc)

        node_checksum = md5_checksum(f"{journey_id}{action}{tools}")

        async with self._lock.writer_lock:
            node = JourneyNode(
                id=JourneyNodeId(self._id_generator.generate(node_checksum)),
                creation_utc=creation_utc,
                action=action,
                tools=tools,
                metadata={},
            )

            await self._node_association_collection.insert_one(
                document=self._serialize_node(node, journey_id)
            )

        return node

    @override
    async def read_node(
        self,
        node_id: JourneyNodeId,
    ) -> JourneyNode:
        async with self._lock.reader_lock:
            doc = await self._node_association_collection.find_one({"node_id": {"$eq": node_id}})

        if not doc:
            raise ItemNotFoundError(item_id=UniqueId(node_id))

        return self._deserialize_node(doc)

    @override
    async def update_node(
        self,
        node_id: JourneyNodeId,
        params: JourneyNodeUpdateParams,
    ) -> JourneyNode:
        async with self._lock.writer_lock:
            doc = await self._node_association_collection.find_one({"node_id": {"$eq": node_id}})

            if not doc:
                raise ItemNotFoundError(item_id=UniqueId(node_id))

            updated = {**doc, **params}

            result = await self._node_association_collection.update_one(
                filters={"node_id": {"$eq": node_id}},
                params=cast(JourneyNodeAssociationDocument, to_json_dict(updated)),
            )

        assert result.updated_document

        return self._deserialize_node(result.updated_document)

    @override
    async def delete_node(
        self,
        node_id: JourneyNodeId,
    ) -> None:
        async with self._lock.writer_lock:
            node_doc = await self._node_association_collection.find_one(
                {"node_id": {"$eq": node_id}}
            )

            if not node_doc:
                raise ItemNotFoundError(item_id=UniqueId(node_id))

            edges = await self.list_edges(journey_id=node_doc["journey_id"], node_id=node_id)

            for edge in edges:
                await self.delete_edge(edge.id)

            result = await self._node_association_collection.delete_one(
                filters={"node_id": {"$eq": node_id}}
            )

        if result.deleted_count == 0:
            raise ItemNotFoundError(item_id=UniqueId(node_id))

    @override
    async def list_nodes(
        self,
        journey_id: JourneyId,
    ) -> Sequence[JourneyNode]:
        async with self._lock.reader_lock:
            journey = await self.read_journey(journey_id)

            if not journey:
                raise ItemNotFoundError(item_id=UniqueId(journey_id))

            docs = await self._node_association_collection.find(
                filters={"journey_id": {"$eq": journey_id}}
            )

        return [self._deserialize_node(doc) for doc in docs] + [
            JourneyNode(
                id=self.END_NODE_ID,
                creation_utc=datetime.now(timezone.utc),
                action=None,
                tools=[],
                metadata={},
            )
        ]

    @override
    async def set_node_metadata(
        self,
        node_id: JourneyNodeId,
        key: str,
        value: JSONSerializable,
    ) -> JourneyNode:
        async with self._lock.writer_lock:
            doc = await self._node_association_collection.find_one({"node_id": {"$eq": node_id}})

            if not doc:
                raise ItemNotFoundError(item_id=UniqueId(node_id))

            updated_metadata = {**doc["metadata"], key: value}

            result = await self._node_association_collection.update_one(
                filters={"node_id": {"$eq": node_id}},
                params={
                    "metadata": updated_metadata,
                },
            )

        assert result.updated_document

        return self._deserialize_node(result.updated_document)

    @override
    async def unset_node_metadata(
        self,
        node_id: JourneyNodeId,
        key: str,
    ) -> JourneyNode:
        async with self._lock.writer_lock:
            doc = await self._node_association_collection.find_one({"node_id": {"$eq": node_id}})

            if not doc:
                raise ItemNotFoundError(item_id=UniqueId(node_id))

            updated_metadata = {k: v for k, v in doc["metadata"].items() if k != key}

            result = await self._node_association_collection.update_one(
                filters={"node_id": {"$eq": node_id}},
                params={
                    "metadata": updated_metadata,
                },
            )

        assert result.updated_document

        return self._deserialize_node(result.updated_document)

    @override
    async def create_edge(
        self,
        journey_id: JourneyId,
        source: JourneyNodeId,
        target: JourneyNodeId,
        condition: Optional[str] = None,
    ) -> JourneyEdge:
        async with self._lock.writer_lock:
            edge_checksum = md5_checksum(f"{journey_id}{source}{target}{condition}")

            edge = JourneyEdge(
                id=JourneyEdgeId(self._id_generator.generate(edge_checksum)),
                creation_utc=datetime.now(timezone.utc),
                source=source,
                target=target,
                condition=condition,
                metadata={},
            )

            await self._edge_association_collection.insert_one(
                document=self._serialize_edge(edge, journey_id)
            )

        return edge

    @override
    async def read_edge(
        self,
        edge_id: JourneyNodeId,
    ) -> JourneyEdge:
        async with self._lock.reader_lock:
            doc = await self._edge_association_collection.find_one({"id": {"$eq": edge_id}})

            if not doc:
                raise ItemNotFoundError(item_id=UniqueId(edge_id))

        return self._deserialize_edge(doc)

    @override
    async def update_edge(
        self,
        edge_id: JourneyNodeId,
        params: JourneyEdgeUpdateParams,
    ) -> JourneyEdge:
        async with self._lock.writer_lock:
            doc = await self._edge_association_collection.find_one({"id": {"$eq": edge_id}})

            if not doc:
                raise ItemNotFoundError(item_id=UniqueId(edge_id))

            updated = {**doc, **params}

            result = await self._edge_association_collection.update_one(
                filters={"id": {"$eq": edge_id}},
                params=cast(JourneyEdgeAssociationDocument, to_json_dict(updated)),
            )

        assert result.updated_document

        return self._deserialize_edge(result.updated_document)

    @override
    async def list_edges(
        self,
        journey_id: JourneyId,
        node_id: Optional[JourneyNodeId] = None,
    ) -> Sequence[JourneyEdge]:
        async with self._lock.reader_lock:
            if journey_id is not None:
                journey = await self.read_journey(journey_id)

                if not journey:
                    raise ItemNotFoundError(item_id=UniqueId(journey_id))

                filters: Where = {"journey_id": {"$eq": journey_id}}

            if node_id is not None:
                filters = {
                    "$or": [
                        {"source": {"$eq": node_id}},
                        {"target": {"$eq": node_id}},
                    ]
                }

            docs = await self._edge_association_collection.find(filters=filters)

        return [self._deserialize_edge(doc) for doc in docs]

    @override
    async def delete_edge(
        self,
        edge_id: JourneyEdgeId,
    ) -> None:
        async with self._lock.writer_lock:
            result = await self._edge_association_collection.delete_one(
                filters={"id": {"$eq": edge_id}}
            )

        if result.deleted_count == 0:
            raise ItemNotFoundError(item_id=UniqueId(edge_id))

    @override
    async def set_edge_metadata(
        self,
        edge_id: JourneyEdgeId,
        key: str,
        value: JSONSerializable,
    ) -> JourneyEdge:
        async with self._lock.writer_lock:
            doc = await self._edge_association_collection.find_one({"id": {"$eq": edge_id}})

            if not doc:
                raise ItemNotFoundError(item_id=UniqueId(edge_id))

            updated_metadata = {**doc["metadata"], key: value}

            result = await self._edge_association_collection.update_one(
                filters={"id": {"$eq": edge_id}},
                params={
                    "metadata": updated_metadata,
                },
            )

        assert result.updated_document

        return self._deserialize_edge(result.updated_document)

    @override
    async def unset_edge_metadata(
        self,
        edge_id: JourneyEdgeId,
        key: str,
    ) -> JourneyEdge:
        async with self._lock.writer_lock:
            doc = await self._edge_association_collection.find_one({"id": {"$eq": edge_id}})

            if not doc:
                raise ItemNotFoundError(item_id=UniqueId(edge_id))

            updated_metadata = {k: v for k, v in doc["metadata"].items() if k != key}

            result = await self._edge_association_collection.update_one(
                filters={"id": {"$eq": edge_id}},
                params={
                    "metadata": updated_metadata,
                },
            )

        assert result.updated_document

        return self._deserialize_edge(result.updated_document)
