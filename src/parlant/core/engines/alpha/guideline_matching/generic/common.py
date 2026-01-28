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

from dataclasses import dataclass
import json
from typing import Optional, cast

from parlant.core.guidelines import Guideline, GuidelineId
from parlant.core.journeys import JourneyEdgeId, JourneyNodeId


@dataclass
class GuidelineInternalRepresentation:
    condition: str
    action: Optional[str]
    description: Optional[str]


def escape_json_string(s: str) -> str:
    return json.dumps(s)[1:-1]


def internal_representation(g: Guideline) -> GuidelineInternalRepresentation:
    action, condition = g.content.action, g.content.condition
    description = g.content.description

    # Escape special characters (newlines, quotes, etc.) for valid JSON outputs
    condition = escape_json_string(condition)
    action = escape_json_string(action) if action else None

    if agent_intention_condition := g.metadata.get("agent_intention_condition"):
        condition = cast(str, agent_intention_condition) or condition

    if internal_action := g.metadata.get("internal_action"):
        action = cast(str, internal_action) or action

    return GuidelineInternalRepresentation(condition, action, description)


def format_journey_node_guideline_id(
    node_id: JourneyNodeId,
    edge_id: Optional[JourneyEdgeId] = None,
) -> GuidelineId:
    if edge_id:
        return GuidelineId(f"journey_node:{node_id}:{edge_id}")

    return GuidelineId(f"journey_node:{node_id}")
