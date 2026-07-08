from parlant.core.engines.alpha.guideline_matching.generic.journey.journey_backtrack_node_selection import (
    JourneyBacktrackNodeSelectionSchema,
    _JourneyEdge as _AlphaJourneyEdge,
    _JourneyNode,
    build_node_wrappers,
    get_journey_transition_map_text,
)
from parlant.core.rules import Rule


class _JourneyEdge(_AlphaJourneyEdge):
    def __init__(
        self,
        target_rule: Rule | None,
        condition: str | None,
        source_node_index: str,
        target_node_index: str,
    ) -> None:
        super().__init__(
            target_guideline=target_rule,
            condition=condition,
            source_node_index=source_node_index,
            target_node_index=target_node_index,
        )

    @property
    def target_rule(self) -> Rule | None:
        return self.target_guideline


__all__ = [
    "JourneyBacktrackNodeSelectionSchema",
    "_JourneyEdge",
    "_JourneyNode",
    "build_node_wrappers",
    "get_journey_transition_map_text",
]
