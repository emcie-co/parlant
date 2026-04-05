from collections import defaultdict, deque
from dataclasses import replace
from datetime import datetime, timezone
from typing import Sequence, cast
from parlant.core.common import Criticality, JSONSerializable
from parlant.core.engines.alpha.guideline_matching.generic.common import (
    format_journey_node_guideline_id,
)
from parlant.core.guidelines import Guideline, GuidelineStore, GuidelineContent, GuidelineId
from parlant.core.journeys import (
    JourneyEdge,
    JourneyEdgeId,
    JourneyId,
    JourneyLink,
    JourneyLinkId,
    JourneyNode,
    JourneyNodeId,
    JourneyStore,
)


def extract_edge_id_from_journey_node_guideline_id(
    guideline_id: GuidelineId,
) -> JourneyEdgeId | None:
    parts = guideline_id.split(":")
    if len(parts) < 2 or parts[0] != "journey_node":
        raise ValueError(f"Invalid guideline ID format: {guideline_id}")

    return JourneyEdgeId(parts[2]) if len(parts) > 2 else None


def extract_node_id_from_journey_node_guideline_id(
    guideline_id: GuidelineId,
) -> JourneyNodeId:
    parts = guideline_id.split(":")
    if len(parts) < 2 or parts[0] != "journey_node":
        raise ValueError(f"Invalid guideline ID format: {guideline_id}")

    return JourneyNodeId(parts[1])


def extract_link_id_from_journey_node_guideline_id(
    guideline_id: GuidelineId,
) -> JourneyLinkId | None:
    parts = guideline_id.split(":")
    if len(parts) < 2 or parts[0] != "journey_node":
        raise ValueError(f"Invalid guideline ID format: {guideline_id}")

    return JourneyLinkId(parts[3]) if len(parts) > 3 else None


class JourneyGuidelineProjection:
    def __init__(
        self,
        journey_store: JourneyStore,
        guideline_store: GuidelineStore,
    ) -> None:
        self._journey_store = journey_store
        self._guideline_store = guideline_store

    async def _resolve_links(
        self,
        parent_journey_id: JourneyId,
        nodes: dict[JourneyNodeId, JourneyNode],
        edges: dict[JourneyEdgeId, JourneyEdge],
        node_edges: dict[JourneyNodeId, list[JourneyEdge]],
    ) -> None:
        """Resolve sub-journey links by injecting mapped nodes and edges into the parent graph."""
        links = await self._journey_store.list_links(parent_journey_id)

        for link in links:
            await self._resolve_single_link(
                link=link,
                parent_journey_id=parent_journey_id,
                nodes=nodes,
                edges=edges,
                node_edges=node_edges,
            )

    async def _resolve_single_link(
        self,
        link: JourneyLink,
        parent_journey_id: JourneyId,
        nodes: dict[JourneyNodeId, JourneyNode],
        edges: dict[JourneyEdgeId, JourneyEdge],
        node_edges: dict[JourneyNodeId, list[JourneyEdge]],
    ) -> None:
        sub_journey = await self._journey_store.read_journey(link.sub_journey_id)
        sub_nodes_list = await self._journey_store.list_nodes(link.sub_journey_id)
        sub_edges_list = await self._journey_store.list_edges(link.sub_journey_id)
        sub_nodes = {n.id: n for n in sub_nodes_list}

        sub_node_edges: dict[JourneyNodeId, list[JourneyEdge]] = defaultdict(list)
        for edge in sub_edges_list:
            sub_node_edges[edge.source].append(edge)

        def scoped_node_id(node_id: JourneyNodeId) -> JourneyNodeId:
            """Namespace a sub-journey node ID with link ID to prevent collisions."""
            return JourneyNodeId(f"{link.id}:{node_id}")

        def scoped_edge_id(edge_id: JourneyEdgeId) -> JourneyEdgeId:
            """Namespace a sub-journey edge ID with link ID to prevent collisions."""
            return JourneyEdgeId(f"{link.id}:{edge_id}")

        link_metadata: dict[str, JSONSerializable] = {
            "link_id": link.id,
            "sub_journey_id": link.sub_journey_id,
        }

        # BFS through sub-journey, skipping the root node
        queue: deque[JourneyNodeId] = deque()
        visited: set[JourneyNodeId] = set()

        # Start from root's children
        root_edges = sub_node_edges.get(sub_journey.root_id, [])

        for root_edge in root_edges:
            if root_edge.target == JourneyStore.END_NODE_ID:
                # Root directly transitions to END — wire source to merge node
                virtual_edge = JourneyEdge(
                    id=scoped_edge_id(root_edge.id),
                    creation_utc=root_edge.creation_utc,
                    source=link.source_node_id,
                    target=link.merge_node_id,
                    condition=link.condition or root_edge.condition,
                    metadata={**root_edge.metadata, "journey_node": link_metadata},
                )
                edges[virtual_edge.id] = virtual_edge
                node_edges[virtual_edge.source].append(virtual_edge)
                continue

            target_node = sub_nodes.get(root_edge.target)
            if not target_node:
                continue

            namespaced_target = scoped_node_id(root_edge.target)
            if namespaced_target not in nodes:
                self._inject_sub_node(
                    nodes, target_node, parent_journey_id, link.sub_journey_id, link.id
                )

            condition = link.condition or root_edge.condition
            virtual_edge = JourneyEdge(
                id=scoped_edge_id(root_edge.id),
                creation_utc=root_edge.creation_utc,
                source=link.source_node_id,
                target=namespaced_target,
                condition=condition,
                metadata={**root_edge.metadata, "journey_node": link_metadata},
            )
            edges[virtual_edge.id] = virtual_edge
            node_edges[virtual_edge.source].append(virtual_edge)

            queue.append(root_edge.target)

        # BFS the rest of the sub-journey
        while queue:
            current_id = queue.popleft()
            if current_id in visited:
                continue
            visited.add(current_id)

            namespaced_current = scoped_node_id(current_id)
            current_edges = sub_node_edges.get(current_id, [])

            # If leaf node (no outgoing edges), connect to merge
            if not current_edges:
                leaf_edge_id = JourneyEdgeId(f"leaf:{link.id}:{current_id}")
                leaf_edge = JourneyEdge(
                    id=leaf_edge_id,
                    creation_utc=datetime.now(timezone.utc),
                    source=namespaced_current,
                    target=link.merge_node_id,
                    condition=None,
                    metadata={"journey_node": link_metadata},
                )
                edges[leaf_edge.id] = leaf_edge
                node_edges[namespaced_current].append(leaf_edge)
                continue

            for sub_edge in current_edges:
                if sub_edge.target == JourneyStore.END_NODE_ID:
                    # END transition — wire to merge node
                    virtual_edge = JourneyEdge(
                        id=scoped_edge_id(sub_edge.id),
                        creation_utc=sub_edge.creation_utc,
                        source=namespaced_current,
                        target=link.merge_node_id,
                        condition=sub_edge.condition,
                        metadata={**sub_edge.metadata, "journey_node": link_metadata},
                    )
                    edges[virtual_edge.id] = virtual_edge
                    node_edges[namespaced_current].append(virtual_edge)
                    continue

                target_node = sub_nodes.get(sub_edge.target)
                if not target_node:
                    continue

                namespaced_target = scoped_node_id(sub_edge.target)
                if namespaced_target not in nodes:
                    self._inject_sub_node(
                        nodes, target_node, parent_journey_id, link.sub_journey_id, link.id
                    )

                virtual_edge = JourneyEdge(
                    id=scoped_edge_id(sub_edge.id),
                    creation_utc=sub_edge.creation_utc,
                    source=namespaced_current,
                    target=namespaced_target,
                    condition=sub_edge.condition,
                    metadata={**sub_edge.metadata, "journey_node": link_metadata},
                )
                edges[virtual_edge.id] = virtual_edge
                node_edges[namespaced_current].append(virtual_edge)

                queue.append(sub_edge.target)

    @staticmethod
    def _inject_sub_node(
        nodes: dict[JourneyNodeId, JourneyNode],
        original: JourneyNode,
        parent_journey_id: JourneyId,
        sub_journey_id: JourneyId,
        link_id: JourneyLinkId,
    ) -> None:
        original_journey_node = cast(
            dict[str, JSONSerializable], original.metadata.get("journey_node", {}) or {}
        )

        metadata = dict(original.metadata)
        metadata["journey_node"] = {
            **original_journey_node,
            "journey_id": parent_journey_id,
            "sub_journey_id": sub_journey_id,
            "link_id": link_id,
            "original_node_id": original.id,
        }

        namespaced_id = JourneyNodeId(f"{link_id}:{original.id}")
        nodes[namespaced_id] = replace(
            original,
            id=namespaced_id,
            metadata=metadata,
        )

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

        # Resolve sub-journey links into the graph
        await self._resolve_links(journey_id, nodes, edges, node_edges)

        def _get_link_context(
            node: JourneyNode,
            edge: JourneyEdge | None,
        ) -> JourneyLinkId | None:
            """Extract link_id from edge or node journey_node metadata."""
            if edge:
                edge_jn = edge.metadata.get("journey_node")
                if isinstance(edge_jn, dict) and "link_id" in edge_jn:
                    return JourneyLinkId(cast(str, edge_jn["link_id"]))

            node_jn = node.metadata.get("journey_node")
            if isinstance(node_jn, dict) and "link_id" in node_jn:
                return JourneyLinkId(cast(str, node_jn["link_id"]))

            return None

        def _get_original_node_id(node: JourneyNode) -> JourneyNodeId:
            """Get original node ID for linked nodes, or node.id for parent nodes."""
            node_jn = node.metadata.get("journey_node")
            if isinstance(node_jn, dict) and "original_node_id" in node_jn:
                return JourneyNodeId(cast(str, node_jn["original_node_id"]))
            return node.id

        def _resolve_guideline_id(
            node: JourneyNode,
            edge: JourneyEdge | None,
        ) -> GuidelineId:
            """Build guideline ID using original node/edge IDs + link_id."""
            original_node_id = _get_original_node_id(node)
            link_id = _get_link_context(node, edge)
            return format_journey_node_guideline_id(
                original_node_id,
                edge.id if edge else None,
                link_id,
            )

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
                "labels": list(node.labels),
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
                id=_resolve_guideline_id(node, edge),
                content=GuidelineContent(
                    condition=edge.condition if edge and edge.condition else "",
                    action=node.action,
                    description=node.description,
                ),
                criticality=Criticality.HIGH,
                creation_utc=datetime.now(timezone.utc),
                enabled=True,
                tags=list(journey.tags),
                metadata=metadata,
                composition_mode=node.composition_mode,
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

                target_node = nodes[edge.target]
                follow_up_id = _resolve_guideline_id(target_node, edge)
                add_edge_guideline_metadata(new_guideline.id, follow_up_id)

            visited.add((edge_id, node_id))

        return list(guidelines.values())
