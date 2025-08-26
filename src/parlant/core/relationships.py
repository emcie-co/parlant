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
from typing import NewType, Optional, Sequence, Union, cast
from typing_extensions import override, TypedDict, Self

import networkx  # type: ignore

from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import ItemNotFoundError, UniqueId, Version, IdGenerator
from parlant.core.guidelines import GuidelineId
from parlant.core.persistence.common import ObjectId, Where
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


class RelationshipKind(Enum):
    """Enumeration of relationship kinds."""

    ENTAILMENT = "entailment"
    """When SOURCE is activated, TARGET should always be activated."""

    PRIORITY = "priority"
    """When both SOURCE and TARGET are activated, only SOURCE should be activated."""

    DEPENDENCY = "dependency"
    """When SOURCE is activated, deactivate it unless T is also activated."""

    DISAMBIGUATION = "disambiguation"
    """When SOURCE is activated and two or more of the targets T ∈ {T₁, T₂, ...} are activated, ask the customer to clarify which action they want to take."""

    REEVALUATION = "reevaluation"
    """When TARGET tool is executed, re-evaluate SOURCE guideline before responding."""

    OVERLAP = "overlap"
    """When SOURCE and TARGET tools are both evaluated, they should be evaluated in the same batch to prevent conflicts."""


RelationshipEntityId = Union[GuidelineId, TagId, ToolId]


class RelationshipEntityKind(Enum):
    """Enumeration of relationship entity kinds."""

    GUIDELINE = "guideline"
    """A guideline entity."""

    TAG = "tag"
    """A tag entity."""

    TOOL = "tool"
    """A tool entity."""


@dataclass(frozen=True)
class RelationshipEntity:
    """An entity that can be part of a relationship."""

    id: RelationshipEntityId
    kind: RelationshipEntityKind

    def id_to_string(self) -> str:
        return str(self.id) if not isinstance(self.id, ToolId) else self.id.to_string()


@dataclass(frozen=True)
class Relationship:
    """A relationship between two entities."""

    id: RelationshipId
    creation_utc: datetime
    source: RelationshipEntity
    target: RelationshipEntity
    kind: RelationshipKind

    def __hash__(self) -> int:
        return hash(self.id)


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
        kind: Optional[RelationshipKind] = None,
        indirect: bool = False,
        source_id: Optional[RelationshipEntityId] = None,
        target_id: Optional[RelationshipEntityId] = None,
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
    kind: RelationshipKind


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

    def __init__(
        self,
        id_generator: IdGenerator,
        database: DocumentDatabase,
        allow_migration: bool = False,
    ) -> None:
        self._id_generator = id_generator

        self._database = database
        self._collection: DocumentCollection[RelationshipDocument]
        self._graphs: dict[RelationshipKind | RelationshipKind, networkx.DiGraph] = {}
        self._allow_migration = allow_migration
        self._lock = ReaderWriterLock()

    async def _document_loader(self, doc: BaseDocument) -> Optional[RelationshipDocument]:
        async def v0_2_0_to_v0_3_0(doc: BaseDocument) -> Optional[BaseDocument]:
            raise ValueError("Cannot load v0.2.0 relationships")

        async def v0_1_0_to_v0_2_0(doc: BaseDocument) -> Optional[BaseDocument]:
            raise ValueError("Cannot load v0.1.0 relationships")

        return await DocumentMigrationHelper[RelationshipDocument](
            self,
            {
                "0.1.0": v0_1_0_to_v0_2_0,
                "0.2.0": v0_2_0_to_v0_3_0,
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
            source_type=relationship.source.kind.value,
            target=relationship.target.id_to_string(),
            target_type=relationship.target.kind.value,
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
            entity_type = RelationshipEntityKind(entity_type_str)

            if entity_type == RelationshipEntityKind.GUIDELINE:
                return RelationshipEntity(id=GuidelineId(id), kind=RelationshipEntityKind.GUIDELINE)
            elif entity_type == RelationshipEntityKind.TAG:
                return RelationshipEntity(id=TagId(id), kind=RelationshipEntityKind.TAG)
            elif entity_type == RelationshipEntityKind.TOOL:
                return RelationshipEntity(
                    id=ToolId.from_string(id), kind=RelationshipEntityKind.TOOL
                )

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
            RelationshipKind(relationship_document["kind"])
            if source.kind in {RelationshipEntityKind.GUIDELINE, RelationshipEntityKind.TAG}
            else RelationshipKind(relationship_document["kind"])
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

            relationship_checksum = f"{source.id_to_string()}{target.id_to_string()}{kind.value}"

            relationship = Relationship(
                id=RelationshipId(self._id_generator.generate(relationship_checksum)),
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
        kind: Optional[RelationshipKind] = None,
        indirect: bool = True,
        source_id: Optional[RelationshipEntityId] = None,
        target_id: Optional[RelationshipEntityId] = None,
    ) -> Sequence[Relationship]:
        async def get_node_relationships_by_kind(
            source_id: RelationshipEntityId,
            reversed_graph: bool = False,
        ) -> Sequence[Relationship]:
            if not graph.has_node(source_id):
                return []

            _graph = graph.reverse() if reversed_graph else graph

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

        async with self._lock.reader_lock:
            if not source_id and not target_id:
                filters = {**({"kind": {"$eq": kind.value}} if kind else {})}
                return [
                    self._deserialize(d)
                    for d in await self._collection.find(filters=cast(Where, filters))
                ]

            relationships: list[Relationship] = []

            if indirect:
                for _kind in (
                    [kind]
                    if kind
                    else [
                        *list(RelationshipKind),
                        *list(RelationshipKind),
                    ]
                ):
                    graph = await self._get_relationships_graph(_kind)

                    if source_id:
                        relationships.extend(
                            await get_node_relationships_by_kind(source_id, reversed_graph=False)
                        )
                    if target_id:
                        relationships.extend(
                            await get_node_relationships_by_kind(target_id, reversed_graph=True)
                        )

                return relationships
            else:
                if source_id:
                    relationships.extend(
                        [
                            self._deserialize(d)
                            for d in await self._collection.find(
                                filters={
                                    "source": {
                                        "$eq": source_id.to_string()
                                        if isinstance(source_id, ToolId)
                                        else str(source_id)
                                    }
                                }
                            )
                        ]
                    )
                if target_id:
                    relationships.extend(
                        [
                            self._deserialize(d)
                            for d in await self._collection.find(
                                filters={
                                    "target": {
                                        "$eq": target_id.to_string()
                                        if isinstance(target_id, ToolId)
                                        else str(target_id)
                                    }
                                }
                            )
                        ]
                    )

        return relationships
