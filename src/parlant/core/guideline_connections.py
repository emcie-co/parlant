# Copyright 2024 Emcie Co Ltd.
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
from enum import Enum, auto
from typing import NewType, Optional, Sequence
from typing_extensions import override, TypedDict, Self

import networkx  # type: ignore

from parlant.core.async_utils import ReaderWriterLock
from parlant.core.common import ItemNotFoundError, UniqueId, Version, generate_id
from parlant.core.guidelines import GuidelineId
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import DocumentDatabase, DocumentCollection

GuidelineConnectionId = NewType("GuidelineConnectionId", str)


class ConnectionKind(Enum):
    ENTAILS = auto()
    SUGGESTS = auto()


@dataclass(frozen=True)
class GuidelineConnection:
    id: GuidelineConnectionId
    creation_utc: datetime
    source: GuidelineId
    target: GuidelineId
    kind: ConnectionKind


class GuidelineConnectionStore(ABC):
    @abstractmethod
    async def create_connection(
        self,
        source: GuidelineId,
        target: GuidelineId,
        kind: ConnectionKind,
    ) -> GuidelineConnection: ...

    @abstractmethod
    async def delete_connection(
        self,
        id: GuidelineConnectionId,
    ) -> None: ...

    @abstractmethod
    async def list_connections(
        self,
        indirect: bool,
        source: Optional[GuidelineId] = None,
        target: Optional[GuidelineId] = None,
    ) -> Sequence[GuidelineConnection]: ...


class _GuidelineConnectionDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    source: GuidelineId
    target: GuidelineId
    kind: str


class GuidelineConnectionDocumentStore(GuidelineConnectionStore):
    VERSION = Version.from_string("0.1.0")

    def __init__(self, database: DocumentDatabase) -> None:
        self._database = database
        self._collection: DocumentCollection[_GuidelineConnectionDocument]
        self._graph: networkx.DiGraph | None = None

        self._lock = ReaderWriterLock()

    async def __aenter__(self) -> Self:
        self._collection = await self._database.get_or_create_collection(
            name="guideline_connections",
            schema=_GuidelineConnectionDocument,
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
        guideline_connection: GuidelineConnection,
    ) -> _GuidelineConnectionDocument:
        return _GuidelineConnectionDocument(
            id=ObjectId(guideline_connection.id),
            version=self.VERSION.to_string(),
            creation_utc=guideline_connection.creation_utc.isoformat(),
            source=guideline_connection.source,
            target=guideline_connection.target,
            kind=guideline_connection.kind.name,
        )

    def _deserialize(
        self,
        guideline_connection_document: _GuidelineConnectionDocument,
    ) -> GuidelineConnection:
        return GuidelineConnection(
            id=GuidelineConnectionId(guideline_connection_document["id"]),
            creation_utc=datetime.fromisoformat(guideline_connection_document["creation_utc"]),
            source=guideline_connection_document["source"],
            target=guideline_connection_document["target"],
            kind=ConnectionKind[guideline_connection_document["kind"]],
        )

    async def _get_graph(self) -> networkx.DiGraph:
        if not self._graph:
            g = networkx.DiGraph()

            connections = [self._deserialize(d) for d in await self._collection.find(filters={})]

            nodes = set()
            edges = list()

            for c in connections:
                nodes.add(c.source)
                nodes.add(c.target)
                edges.append(
                    (
                        c.source,
                        c.target,
                        {
                            "kind": c.kind,
                            "id": c.id,
                        },
                    )
                )

            g.update(edges=edges, nodes=nodes)

            self._graph = g

        return self._graph

    @override
    async def create_connection(
        self,
        source: GuidelineId,
        target: GuidelineId,
        kind: ConnectionKind,
        creation_utc: Optional[datetime] = None,
    ) -> GuidelineConnection:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)

            guideline_connection = GuidelineConnection(
                id=GuidelineConnectionId(generate_id()),
                creation_utc=creation_utc,
                source=source,
                target=target,
                kind=kind,
            )

            result = await self._collection.update_one(
                filters={"source": {"$eq": source}, "target": {"$eq": target}},
                params=self._serialize(guideline_connection),
                upsert=True,
            )

        assert result.updated_document

        graph = await self._get_graph()

        graph.add_node(source)
        graph.add_node(target)

        graph.add_edge(
            source,
            target,
            kind=kind,
            id=guideline_connection.id,
        )

        return guideline_connection

    @override
    async def delete_connection(
        self,
        id: GuidelineConnectionId,
    ) -> None:
        async with self._lock.writer_lock:
            connection_document = await self._collection.find_one(filters={"id": {"$eq": id}})

            if not connection_document:
                raise ItemNotFoundError(item_id=UniqueId(id))

            connection = self._deserialize(connection_document)

            (await self._get_graph()).remove_edge(connection.source, connection.target)

            await self._collection.delete_one(filters={"id": {"$eq": id}})

    @override
    async def list_connections(
        self,
        indirect: bool,
        source: Optional[GuidelineId] = None,
        target: Optional[GuidelineId] = None,
    ) -> Sequence[GuidelineConnection]:
        assert (source or target) and not (source and target)

        async def get_node_connections(
            source: GuidelineId,
            reversed_graph: bool = False,
        ) -> Sequence[GuidelineConnection]:
            if not graph.has_node(source):
                return []

            _graph = graph.reverse() if reversed_graph else graph

            if indirect:
                descendant_edges = networkx.bfs_edges(_graph, source)
                connections = []

                for edge_source, edge_target in descendant_edges:
                    edge_data = _graph.get_edge_data(edge_source, edge_target)

                    connection_document = await self._collection.find_one(
                        filters={"id": {"$eq": edge_data["id"]}},
                    )

                    if not connection_document:
                        raise ItemNotFoundError(item_id=UniqueId(edge_data["id"]))

                    connections.append(self._deserialize(connection_document))

                return connections

            else:
                successors = _graph.succ[source]
                connections = []

                for source, data in successors.items():
                    connection_document = await self._collection.find_one(
                        filters={"id": {"$eq": data["id"]}},
                    )

                    if not connection_document:
                        raise ItemNotFoundError(item_id=UniqueId(data["id"]))

                    connections.append(self._deserialize(connection_document))

                return connections

        async with self._lock.reader_lock:
            graph = await self._get_graph()

            if source:
                connections = await get_node_connections(source)
            elif target:
                connections = await get_node_connections(target, reversed_graph=True)

        return connections
