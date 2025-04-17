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
from enum import Enum
from typing import NewType, Optional, Sequence, Union
from typing_extensions import override, TypedDict, Self

import networkx  # type: ignore

from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import ItemNotFoundError, UniqueId, Version, generate_id
from parlant.core.guidelines import GuidelineId
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentDatabase,
    DocumentCollection,
)
from parlant.core.persistence.document_database_helper import (
    DocumentMigrationHelper,
    DocumentStoreMigrationHelper,
)
from parlant.core.tags import TagId
from parlant.core.tools import ToolId

RelationshipId = NewType("RelationshipId", str)


class GuidelineRelationshipKind(Enum):
    ENTAILMENT = "entailment"
    PRECEDENCE = "precedence"
    REQUIREMENT = "requirement"
    PRIORITY = "priority"
    PERSISTENCE = "persistence"
    DEPENDENCY = "dependency"


class ToolRelationshipKind(Enum):
    OVERLAP = "overlap"


RelationshipKind = Union[GuidelineRelationshipKind, ToolRelationshipKind]


EntityIdType = Union[GuidelineId, TagId, ToolId]


class EntityType(Enum):
    GUIDELINE = "guideline"
    TAG = "tag"
    TOOL = "tool"


@dataclass(frozen=True)
class RelationshipEntity:
    id: EntityIdType
    type: EntityType

    def id_to_string(self) -> str:
        return str(self.id) if not isinstance(self.id, ToolId) else self.id.to_string()


@dataclass(frozen=True)
class Relationship:
    id: RelationshipId
    creation_utc: datetime
    source: RelationshipEntity
    target: RelationshipEntity
    kind: RelationshipKind


class RelationshipStore(ABC):
    @abstractmethod
    async def create_relationship(
        self,
        source: RelationshipEntity,
        target: RelationshipEntity,
        kind: RelationshipKind,
    ) -> Relationship: ...

    @abstractmethod
    async def read_relationship(
        self,
        id: RelationshipId,
    ) -> Relationship: ...

    @abstractmethod
    async def delete_relationship(
        self,
        id: RelationshipId,
    ) -> None: ...

    @abstractmethod
    async def list_relationships(
        self,
        kind: RelationshipKind,
        indirect: bool,
        source_id: Optional[EntityIdType] = None,
        target_id: Optional[EntityIdType] = None,
    ) -> Sequence[Relationship]: ...


class GuidelineRelationshipDocument_v0_1_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    source: GuidelineId
    target: GuidelineId


class GuidelineRelationshipDocument_v0_2_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    source: GuidelineId
    target: GuidelineId
    kind: GuidelineRelationshipKind


class RelationshipDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    source: str
    source_type: str
    target: str
    target_type: str
    kind: str


class RelationshipDocumentStore(RelationshipStore):
    VERSION = Version.from_string("0.3.0")

    def __init__(self, database: DocumentDatabase, allow_migration: bool = False) -> None:
        self._database = database
        self._collection: DocumentCollection[RelationshipDocument]
        self._graphs: dict[GuidelineRelationshipKind | ToolRelationshipKind, networkx.DiGraph] = {}
        self._allow_migration = allow_migration
        self._lock = ReaderWriterLock()

    async def _document_loader(self, doc: BaseDocument) -> Optional[RelationshipDocument]:
        async def v0_2_0_to_v_0_3_0(doc: BaseDocument) -> Optional[BaseDocument]:
            raise ValueError("Cannot load v0.2.0 relationships")

        async def v0_1_0_to_v_0_2_0(doc: BaseDocument) -> Optional[BaseDocument]:
            raise ValueError("Cannot load v0.1.0 relationships")

        return await DocumentMigrationHelper[RelationshipDocument](
            self,
            {
                "0.1.0": v0_1_0_to_v_0_2_0,
                "0.2.0": v0_2_0_to_v_0_3_0,
            },
        ).migrate(doc)

    async def __aenter__(self) -> Self:
        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
        ):
            self._collection = await self._database.get_or_create_collection(
                name="relationships",
                schema=RelationshipDocument,
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

    def _serialize(
        self,
        relationship: Relationship,
    ) -> RelationshipDocument:
        return RelationshipDocument(
            id=ObjectId(relationship.id),
            version=self.VERSION.to_string(),
            creation_utc=relationship.creation_utc.isoformat(),
            source=relationship.source.id_to_string(),
            source_type=relationship.source.type.value,
            target=relationship.target.id_to_string(),
            target_type=relationship.target.type.value,
            kind=relationship.kind.value,
        )

    def _deserialize(
        self,
        relationship_document: RelationshipDocument,
    ) -> Relationship:
        def _deserialize_entity(
            id: str,
            entity_type_str: str,
        ) -> RelationshipEntity:
            entity_type = EntityType(entity_type_str)

            if entity_type == EntityType.GUIDELINE:
                return RelationshipEntity(id=GuidelineId(id), type=EntityType.GUIDELINE)
            elif entity_type == EntityType.TAG:
                return RelationshipEntity(id=TagId(id), type=EntityType.TAG)
            elif entity_type == EntityType.TOOL:
                return RelationshipEntity(id=ToolId.from_string(id), type=EntityType.TOOL)

            raise ValueError(f"Unknown entity type: {entity_type_str}")

        source = _deserialize_entity(
            relationship_document["source"],
            relationship_document["source_type"],
        )
        target = _deserialize_entity(
            relationship_document["target"],
            relationship_document["target_type"],
        )

        kind = (
            GuidelineRelationshipKind(relationship_document["kind"])
            if source.type in {EntityType.GUIDELINE, EntityType.TAG}
            else ToolRelationshipKind(relationship_document["kind"])
        )

        return Relationship(
            id=RelationshipId(relationship_document["id"]),
            creation_utc=datetime.fromisoformat(relationship_document["creation_utc"]),
            source=source,
            target=target,
            kind=kind,
        )

    async def _get_relationships_graph(self, kind: RelationshipKind) -> networkx.DiGraph:
        if kind not in self._graphs:
            g = networkx.DiGraph()
            g.graph["strict"] = True  # Ensure no loops are allowed

            relationships = [
                self._deserialize(d)
                for d in await self._collection.find(filters={"kind": {"$eq": kind.value}})
            ]

            nodes = set()
            edges = list()

            for r in relationships:
                nodes.add(r.source.id)
                nodes.add(r.target.id)
                edges.append(
                    (
                        r.source.id,
                        r.target.id,
                        {
                            "id": r.id,
                        },
                    )
                )

            g.update(edges=edges, nodes=nodes)

            self._graphs[kind] = g

        return self._graphs[kind]

    @override
    async def create_relationship(
        self,
        source: RelationshipEntity,
        target: RelationshipEntity,
        kind: RelationshipKind,
        creation_utc: Optional[datetime] = None,
    ) -> Relationship:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)

            relationship = Relationship(
                id=RelationshipId(generate_id()),
                creation_utc=creation_utc,
                source=source,
                target=target,
                kind=kind,
            )

            result = await self._collection.update_one(
                filters={
                    "source": {"$eq": source.id_to_string()},
                    "target": {"$eq": target.id_to_string()},
                    "kind": {"$eq": kind.value},
                },
                params=self._serialize(relationship),
                upsert=True,
            )

            assert result.updated_document

            graph = await self._get_relationships_graph(kind)

            graph.add_node(source.id)
            graph.add_node(target.id)

            graph.add_edge(
                source.id,
                target.id,
                id=relationship.id,
            )

        return relationship

    @override
    async def read_relationship(
        self,
        id: RelationshipId,
    ) -> Relationship:
        async with self._lock.reader_lock:
            relationship_document = await self._collection.find_one(filters={"id": {"$eq": id}})

            if not relationship_document:
                raise ItemNotFoundError(item_id=UniqueId(id))

        return self._deserialize(relationship_document)

    @override
    async def delete_relationship(
        self,
        id: RelationshipId,
    ) -> None:
        async with self._lock.writer_lock:
            relationship_document = await self._collection.find_one(filters={"id": {"$eq": id}})

            if not relationship_document:
                raise ItemNotFoundError(item_id=UniqueId(id))

            relationship = self._deserialize(relationship_document)

            graph = await self._get_relationships_graph(relationship.kind)

            graph.remove_edge(relationship.source.id, relationship.target.id)

            await self._collection.delete_one(filters={"id": {"$eq": id}})

    @override
    async def list_relationships(
        self,
        kind: RelationshipKind,
        indirect: bool,
        source_id: Optional[EntityIdType] = None,
        target_id: Optional[EntityIdType] = None,
    ) -> Sequence[Relationship]:
        assert (source_id or target_id) and not (source_id and target_id)

        async def get_node_relationships(
            source_id: EntityIdType,
            reversed_graph: bool = False,
        ) -> Sequence[Relationship]:
            if not graph.has_node(source_id):
                return []

            _graph = graph.reverse() if reversed_graph else graph

            if indirect:
                descendant_edges = networkx.bfs_edges(_graph, source_id)
                relationships = []

                for edge_source, edge_target in descendant_edges:
                    edge_data = _graph.get_edge_data(edge_source, edge_target)

                    relationship_document = await self._collection.find_one(
                        filters={"id": {"$eq": edge_data["id"]}},
                    )

                    if not relationship_document:
                        raise ItemNotFoundError(item_id=UniqueId(edge_data["id"]))

                    relationships.append(self._deserialize(relationship_document))

                return relationships

            else:
                successors = _graph.succ[source_id]
                relationships = []

                for source_id, data in successors.items():
                    relationship_document = await self._collection.find_one(
                        filters={"id": {"$eq": data["id"]}},
                    )

                    if not relationship_document:
                        raise ItemNotFoundError(item_id=UniqueId(data["id"]))

                    relationships.append(self._deserialize(relationship_document))

                return relationships

        async with self._lock.reader_lock:
            graph = await self._get_relationships_graph(kind)

            if source_id:
                relationships = await get_node_relationships(source_id)
            elif target_id:
                relationships = await get_node_relationships(target_id, reversed_graph=True)

        return relationships
