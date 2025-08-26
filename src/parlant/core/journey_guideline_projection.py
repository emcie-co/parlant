from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Sequence, cast
from parlant.core.common import JSONSerializable
from parlant.core.engines.alpha.guideline_matching.generic.common import (
    format_journey_node_guideline_id,
)
from parlant.core.guidelines import Guideline, GuidelineStore, GuidelineContent, GuidelineId
from parlant.core.journeys import (
    JourneyEdge,
    JourneyEdgeId,
    JourneyId,
    JourneyNode,
    JourneyStore,
    JourneyNodeId,
)


def extract_node_id_from_journey_node_guideline_id(
    guideline_id: GuidelineId,
) -> JourneyNodeId:
    parts = guideline_id.split(":")
    if len(parts) < 2 or parts[0] != "journey_node":
        raise ValueError(f"Invalid guideline ID format: {guideline_id}")

    return JourneyNodeId(parts[1])


class JourneyGuidelineProjection:
    def __init__(
        self,
        journey_store: JourneyStore,
        guideline_store: GuidelineStore,
    ) -> None:
        self._journey_store = journey_store
        self._guideline_store = guideline_store

    async def project_journey_to_guidelines(
        self,
        journey_id: JourneyId,
    ) -> Sequence[Guideline]:
        guidelines: dict[GuidelineId, Guideline] = {}

        index = 0

        journey = await self._journey_store.read_journey(journey_id)

        edges_objs = await self._journey_store.list_edges(journey_id)

        nodes = {n.id: n for n in await self._journey_store.list_nodes(journey_id)}
        node_indexes: dict[JourneyNodeId, int] = {}
        edges = {e.id: e for e in edges_objs}

        node_edges: dict[JourneyNodeId, list[JourneyEdge]] = defaultdict(list)

        for edge in edges_objs:
            node_edges[edge.source].append(edge)

        def make_guideline(
            edge: JourneyEdge | None,
            node: JourneyNode,
        ) -> Guideline:
            if node.id not in node_indexes:
                nonlocal index
                index += 1
                node_indexes[node.id] = index

            base_journey_node = {
                "follow_ups": [],
                "index": str(node_indexes[node.id]),
                "journey_id": journey_id,
            }

            # Extract nested journey_node metadata from edge and node
            edge_journey_node = (
                edge.metadata.get("journey_node")
                if edge and "journey_node" in edge.metadata
                else {}
            ) or {}
            node_journey_node = node.metadata.get("journey_node", {}) or {}

            # Merge nested journey_node data
            merged_journey_node = {
                **base_journey_node,
                **cast(dict[str, JSONSerializable], node_journey_node),
                **cast(dict[str, JSONSerializable], edge_journey_node),
            }

            # Merge top-level metadata
            metadata = {
                "journey_node": merged_journey_node,
                **{k: v for k, v in node.metadata.items() if k != "journey_node"},
                **({k: v for k, v in edge.metadata.items() if k != "journey_node"} if edge else {}),
            }

            return Guideline(
                id=format_journey_node_guideline_id(node.id, edge.id if edge else None),
                content=GuidelineContent(
                    condition=edge.condition if edge and edge.condition else "",
                    action=node.action,
                ),
                creation_utc=datetime.now(timezone.utc),
                enabled=True,
                tags=[],
                metadata=metadata,
            )

        def add_edge_guideline_metadata(
            guideline_id: GuidelineId, edge_guideline_id: GuidelineId
        ) -> None:
            cast(dict[str, list[str]], guidelines[guideline_id].metadata["journey_node"])[
                "follow_ups"
            ] = list(
                set(
                    cast(dict[str, list[str]], guidelines[guideline_id].metadata["journey_node"])[
                        "follow_ups"
                    ]
                    + [edge_guideline_id]
                )
            )

        queue: deque[tuple[JourneyEdgeId | None, JourneyNodeId]] = deque()
        queue.append((None, journey.root_id))

        visited: set[tuple[JourneyEdgeId | None, JourneyNodeId]] = set()

        while queue:
            edge_id, node_id = queue.popleft()
            new_guideline = make_guideline(edges[edge_id] if edge_id else None, nodes[node_id])

            guidelines[new_guideline.id] = new_guideline

            for edge in node_edges[node_id]:
                if (edge.id, edge.target) in visited:
                    continue

                queue.append((edge.id, edge.target))

                add_edge_guideline_metadata(
                    new_guideline.id,
                    format_journey_node_guideline_id(edge.target, edge.id),
                )

            visited.add((edge_id, node_id))

        return list(guidelines.values())
