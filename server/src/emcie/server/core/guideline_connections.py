from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, NewType, Optional, Sequence
import networkx  # type: ignore

from emcie.server.base_models import DefaultBaseModel
from emcie.server.core.common import generate_id
from emcie.server.core.guidelines import GuidelineId
from emcie.server.core.persistence.document_database import DocumentDatabase, DocumentCollection

GuidelineConnectionId = NewType("GuidelineConnectionId", str)
ConnectionKind = Literal["entails", "suggests"]


@dataclass(frozen=True)
class GuidelineConnection:
    id: GuidelineConnectionId
    creation_utc: datetime
    source: GuidelineId
    target: GuidelineId
    kind: ConnectionKind


class GuidelineConnectionStore(ABC):
    @abstractmethod
    async def update_connection(
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
        source: GuidelineId,
        indirect: bool,
    ) -> Sequence[GuidelineConnection]: ...


class GuidelineConnectionDocumentStore(GuidelineConnectionStore):
    class GuidelineConnectionDocument(DefaultBaseModel):
        id: GuidelineConnectionId
        creation_utc: datetime
        source: GuidelineId
        target: GuidelineId
        kind: ConnectionKind

    def __init__(self, database: DocumentDatabase) -> None:
        self._collection: DocumentCollection = database.get_or_create_collection(
            name="guideline_connections",
            schema=self.GuidelineConnectionDocument,
        )
        self._graph: networkx.DiGraph | None = None

    async def _get_graph(self) -> networkx.DiGraph:
        if not self._graph:
            g = networkx.DiGraph()

            connections = await self._collection.find(filters={})

            nodes = set()
            edges = list()

            for c in connections:
                nodes.add(c["source"])
                nodes.add(c["target"])
                edges.append(
                    (
                        c["source"],
                        c["target"],
                        {
                            "kind": c["kind"],
                            "id": c["id"],
                        },
                    )
                )

            g.update(edges=edges, nodes=nodes)

            self._graph = g

        return self._graph

    async def update_connection(
        self,
        source: GuidelineId,
        target: GuidelineId,
        kind: ConnectionKind,
        creation_utc: Optional[datetime] = None,
    ) -> GuidelineConnection:
        assert kind in ("entails", "suggests")

        creation_utc = creation_utc or datetime.now(timezone.utc)

        connection_id = await self._collection.update_one(
            filters={"source": {"$eq": source}, "target": {"$eq": target}},
            updated_document={
                "id": generate_id(),
                "creation_utc": creation_utc,
                "source": source,
                "target": target,
                "kind": kind,
            },
            upsert=True,
        )

        graph = await self._get_graph()

        graph.add_node(source)
        graph.add_node(target)

        graph.add_edge(
            source,
            target,
            kind=kind,
            id=connection_id,
        )

        return GuidelineConnection(
            id=GuidelineConnectionId(connection_id),
            creation_utc=creation_utc,
            source=source,
            target=target,
            kind=kind,
        )

    async def delete_connection(
        self,
        id: GuidelineConnectionId,
    ) -> None:
        document = await self._collection.find_one(filters={"id": {"$eq": id}})

        (await self._get_graph()).remove_edge(document["source"], document["target"])

        await self._collection.delete_one(filters={"id": {"$eq": id}})

    async def list_connections(
        self,
        source: GuidelineId,
        indirect: bool,
    ) -> Sequence[GuidelineConnection]:
        graph = await self._get_graph()

        if not graph.has_node(source):
            return []

        if indirect:
            descendant_edges = networkx.bfs_edges(graph, source)
            connections = []

            for edge_source, edge_target in descendant_edges:
                edge_data = graph.get_edge_data(edge_source, edge_target)

                connection = await self._collection.find_one(
                    filters={"id": {"$eq": edge_data["id"]}},
                )

                connections.append(
                    GuidelineConnection(
                        id=connection["id"],
                        source=connection["source"],
                        target=connection["target"],
                        kind=connection["kind"],
                        creation_utc=connection["creation_utc"],
                    )
                )

            return connections

        else:
            successors = graph.succ[source]
            connections = []

            for source, data in successors.items():
                connection = await self._collection.find_one(
                    filters={"id": {"$eq": data["id"]}},
                )

                connections.append(
                    GuidelineConnection(
                        id=connection["id"],
                        source=connection["source"],
                        target=connection["target"],
                        kind=connection["kind"],
                        creation_utc=connection["creation_utc"],
                    )
                )

            return connections
