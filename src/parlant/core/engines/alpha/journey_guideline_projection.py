from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Sequence, cast
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
        root_id = journey.root_id

        edges_objs = await self._journey_store.list_edges(journey_id)

        nodes = {n.id: n for n in await self._journey_store.list_nodes(journey_id)}
        edges = {e.id: e for e in edges_objs}

        node_edges: dict[JourneyNodeId, list[JourneyEdge]] = defaultdict(list)

        for edge in edges_objs:
            node_edges[edge.source].append(edge)

        visited: set[tuple[JourneyNodeId, JourneyEdgeId | None]] = set()

        def make_guideline(
            node: JourneyNode,
            edge: JourneyEdge | None,
        ) -> Guideline:
            nonlocal index
            index += 1

            return Guideline(
                id=format_journey_node_guideline_id(node.id, edge.id if edge else None),
                content=GuidelineContent(
                    condition=edge.condition if edge and edge.condition else "",
                    action=node.action,
                ),
                creation_utc=datetime.now(timezone.utc),
                enabled=True,
                tags=[],
                metadata={
                    **{
                        "journey_node": {"follow_ups": [], "index": index, "journey_id": journey_id}
                    },
                    **({**edge.metadata, **node.metadata} if edge else node.metadata),
                },
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

        queue: deque[tuple[JourneyNodeId, JourneyEdgeId | None]] = deque()
        queue.append((root_id, None))

        while queue:
            node_id, edge_id = queue.popleft()
            new_guideline = make_guideline(nodes[node_id], edges[edge_id] if edge_id else None)

            guidelines[new_guideline.id] = new_guideline

            for edge in node_edges[node_id]:
                if (edge.target, edge.id) in visited:
                    continue

                queue.append((edge.target, edge.id))

                add_edge_guideline_metadata(
                    new_guideline.id, format_journey_node_guideline_id(edge.target, edge.id)
                )

            visited.add((node_id, edge_id))

        return list(guidelines.values())
