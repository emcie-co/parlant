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

"""Engine-specific tracing helpers.

Wraps the generic ``Tracer`` and exposes typed methods for each kind of
event the alpha engine emits, so call sites stay short and the event
schema lives in one place.
"""

from enum import Enum
import json
from typing import Mapping, Optional, Sequence, cast

from parlant.core.common import JSONSerializable
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.guideline_matching.guideline_match import GuidelineMatch
from parlant.core.engines.alpha.relational_resolver import (
    RelationalResolverResult,
    ResolutionKind,
)
from parlant.core.engines.alpha.tool_calling.tool_caller import ToolInsights
from parlant.core.glossary import Term
from parlant.core.guidelines import GuidelineId
from parlant.core.journey_guideline_projection import (
    extract_edge_id_from_journey_node_guideline_id,
    extract_node_id_from_journey_node_guideline_id,
)
from parlant.core.journeys import Journey, JourneyId
from parlant.core.sessions import ToolEventData
from parlant.core.tracer import Tracer


class MatchReason(str, Enum):
    """The reason a guideline appears in a match-tracer event.

    Mirrors :class:`ResolutionKind` for tracing — ``NONE`` is renamed to
    ``COMPLETION`` since "no relational changes" reads, from the trace
    consumer's perspective, as "selected via matcher completion".
    """

    COMPLETION = "completion"
    UNMET_DEPENDENCY_ALL = "unmet_dependency_all"
    UNMET_DEPENDENCY_ANY = "unmet_dependency_any"
    DEPRIORITIZED = "deprioritized"
    ENTAILED = "entailed"


def _resolution_kind_to_match_reason(kind: ResolutionKind) -> MatchReason:
    match kind:
        case ResolutionKind.NONE:
            return MatchReason.COMPLETION
        case ResolutionKind.UNMET_DEPENDENCY_ALL:
            return MatchReason.UNMET_DEPENDENCY_ALL
        case ResolutionKind.UNMET_DEPENDENCY_ANY:
            return MatchReason.UNMET_DEPENDENCY_ANY
        case ResolutionKind.DEPRIORITIZED:
            return MatchReason.DEPRIORITIZED
        case ResolutionKind.ENTAILED:
            return MatchReason.ENTAILED


class EngineTracer:
    """Typed wrapper around :class:`Tracer` for engine events."""

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer

    def context_variable_loaded(
        self,
        variable: ContextVariable,
        value: ContextVariableValue,
    ) -> None:
        self._tracer.add_event(
            "ctx.variable_loaded",
            attributes={
                "variable_id": variable.id,
                "name": variable.name,
                "value": str(value.data),
            },
        )

    def glossary_term_loaded(self, term: Term) -> None:
        self._tracer.add_event(
            "glossary.term_loaded",
            attributes={
                "term_id": term.id,
                "last_modified": term.last_modified_utc.isoformat(),
                "name": term.name,
            },
        )

    def tool_calls(self, tool_events: Sequence[EmittedEvent]) -> None:
        for tool_event in tool_events:
            tool_calls = cast(ToolEventData, tool_event.data)["tool_calls"]
            for tool_call in tool_calls:
                self._tracer.add_event(
                    "tc.result",
                    attributes={
                        "tool_id": tool_call["tool_id"],
                        "rationale": tool_call["rationale"],
                        "arguments": json.dumps(tool_call["arguments"]),
                        "result": json.dumps(tool_call["result"]),
                    },
                )

    def tool_insights(self, tool_insights: ToolInsights) -> None:
        for tool_id, tc_missing in tool_insights.missing_data.items():
            for missing_tc_id, missing_items in tc_missing.items():
                for missing_item in missing_items:
                    self._tracer.add_event(
                        "tc.missing",
                        attributes={
                            "tool_id": tool_id.to_string(),
                            "tool_call_id": missing_tc_id,
                            "parameter": missing_item.parameter,
                        },
                    )

        for tool_id, tc_invalid in tool_insights.invalid_data.items():
            for invalid_tc_id, invalid_items in tc_invalid.items():
                for invalid_item in invalid_items:
                    self._tracer.add_event(
                        "tc.invalid",
                        attributes={
                            "tool_id": tool_id.to_string(),
                            "tool_call_id": invalid_tc_id,
                            "parameter": invalid_item.parameter,
                            "invalid_value": invalid_item.invalid_value,
                        },
                    )

    def matches(
        self,
        matcher_matched: Sequence[GuidelineMatch],
        matcher_ruled_out: Sequence[GuidelineMatch],
        resolver_result: RelationalResolverResult,
        journeys: Optional[Mapping[JourneyId, Journey]] = None,
    ) -> None:
        matcher_match_ids = {m.guideline.id for m in matcher_matched}
        final_match_ids = {m.guideline.id for m in resolver_result.matches}

        matcher_by_id = {m.guideline.id: m for m in matcher_matched}
        final_by_id = {m.guideline.id: m for m in resolver_result.matches}

        def _resolutions_attr(gid: GuidelineId) -> Mapping[str, str]:
            items: list[dict[str, JSONSerializable]] = [
                {
                    "reason": _resolution_kind_to_match_reason(r.kind).value,
                    "description": r.details.description,
                    "target_ids": list(r.details.target_ids),
                    **(
                        {"relationship_id": r.details.relationship_id}
                        if r.details.relationship_id
                        else {}
                    ),
                }
                for r in resolver_result.resolutions.get(gid, [])
            ]
            return {"resolutions": json.dumps(items)} if items else {}

        def _sub_journey_attrs(match: GuidelineMatch) -> Mapping[str, str]:
            journey_node = cast(
                dict[str, JSONSerializable],
                match.guideline.metadata["journey_node"],
            )
            if "sub_journey_id" not in journey_node:
                return {}

            attrs: dict[str, str] = {
                "sub_journey_id": cast(str, journey_node["sub_journey_id"]),
            }
            if "sub_journey_last_modified_utc" in journey_node:
                attrs["sub_journey_last_modified_utc"] = cast(
                    str, journey_node["sub_journey_last_modified_utc"]
                )
            return attrs

        def _emit_selected(match: GuidelineMatch, rationale: str) -> None:
            gid = match.guideline.id
            if match.guideline.metadata.get("journey_node"):
                edge_id = extract_edge_id_from_journey_node_guideline_id(gid)
                journey_id = cast(str, match.metadata.get("step_selection_journey_id"))
                journey_last_modified_utc = ""
                if journeys and journey_id:
                    j = journeys.get(JourneyId(journey_id))
                    if j:
                        journey_last_modified_utc = j.last_modified_utc.isoformat()

                self._tracer.add_event(
                    "journey.state.selected",
                    attributes={
                        **({"edge_id": edge_id} if edge_id else {}),
                        "node_id": extract_node_id_from_journey_node_guideline_id(gid),
                        "journey_path": json.dumps(
                            match.metadata.get("journey_path_guideline_ids", [])
                        ),
                        "rationale": rationale,
                        "journey_id": journey_id,
                        **(
                            {"last_modified": journey_last_modified_utc}
                            if journey_last_modified_utc
                            else {}
                        ),
                        **_sub_journey_attrs(match),
                        **_resolutions_attr(gid),
                    },
                )
            else:
                self._tracer.add_event(
                    "gm.selected",
                    attributes={
                        "guideline_id": gid,
                        "last_modified": match.guideline.last_modified_utc.isoformat(),
                        "rationale": rationale,
                        **_resolutions_attr(gid),
                    },
                )

        def _emit_ruled_out(match: GuidelineMatch, rationale: str) -> None:
            gid = match.guideline.id
            if match.guideline.metadata.get("journey_node"):
                edge_id = extract_edge_id_from_journey_node_guideline_id(gid)
                journey_id = cast(str, match.metadata.get("step_selection_journey_id"))
                journey_last_modified_utc = ""
                if journeys and journey_id:
                    j = journeys.get(JourneyId(journey_id))
                    if j:
                        journey_last_modified_utc = j.last_modified_utc.isoformat()

                self._tracer.add_event(
                    "journey.state.ruled_out",
                    attributes={
                        **({"edge_id": edge_id} if edge_id else {}),
                        "node_id": extract_node_id_from_journey_node_guideline_id(gid),
                        "journey_path": json.dumps(
                            match.guideline.metadata.get("journey_path_guideline_ids", [])
                        ),
                        "rationale": rationale,
                        "journey_id": journey_id,
                        **(
                            {"last_modified": journey_last_modified_utc}
                            if journey_last_modified_utc
                            else {}
                        ),
                        **_sub_journey_attrs(match),
                        **_resolutions_attr(gid),
                    },
                )
            else:
                self._tracer.add_event(
                    "gm.ruled_out",
                    attributes={
                        "guideline_id": gid,
                        "last_modified": match.guideline.last_modified_utc.isoformat(),
                        "rationale": rationale,
                        **_resolutions_attr(gid),
                    },
                )

        # gm.selected: every guideline in the final match set.
        # Rationale comes from the matcher when the guideline was originally matched;
        # for entailed guidelines, fall back to the first non-NONE resolution description.
        for gid in final_match_ids:
            match = final_by_id[gid]
            if gid in matcher_match_ids:
                rationale = matcher_by_id[gid].rationale
            else:
                rationale = next(
                    (
                        r.details.description
                        for r in resolver_result.resolutions.get(gid, [])
                        if r.kind != ResolutionKind.NONE
                    ),
                    "",
                )
            _emit_selected(match, rationale)

        # gm.ruled_out: matcher's ruled out, plus matched-but-dropped-by-resolver.
        for ruled_out in matcher_ruled_out:
            _emit_ruled_out(ruled_out, ruled_out.rationale)

        for gid in matcher_match_ids - final_match_ids:
            match = matcher_by_id[gid]
            # Rationale here describes WHY the resolver dropped it (e.g. "Removed:
            # dependency X unmet"), not why the matcher selected it — otherwise
            # the event would carry a "why I matched it" reason on a "ruled_out" event.
            rationale = next(
                (
                    r.details.description
                    for r in resolver_result.resolutions.get(gid, [])
                    if r.kind != ResolutionKind.NONE
                ),
                "",
            )
            _emit_ruled_out(match, rationale)
