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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Sequence

from parlant.testing.steps import AgentMessage, CustomerMessage, StatusEvent, Step


@dataclass
class PrefabEvent:
    """Represents an event to be created in a session."""

    source: str  # "customer", "ai_agent", "system"
    kind: str  # "message", "tool", "status"
    data: Mapping[str, Any]


class InteractionBuilder:
    """Builder for creating prefab session history.

    Supports fluent API with .step() method for chaining.
    Auto-inserts status events (typing before agent, ready after agent) on build().
    """

    def __init__(self) -> None:
        self._steps: List[Step] = []

    def step(self, step: Step) -> "InteractionBuilder":
        """Add a step to the builder. Returns self for fluent chaining."""
        self._steps.append(step)
        return self

    def build(self) -> List[PrefabEvent]:
        """Build the list of events, auto-inserting status events as needed.

        Automatically inserts:
        - "typing" status before each AgentMessage (if not already present)
        - "ready" status after each AgentMessage (if not already present)
        """
        events: List[PrefabEvent] = []
        i = 0

        while i < len(self._steps):
            current = self._steps[i]

            if isinstance(current, CustomerMessage):
                events.append(
                    PrefabEvent(
                        source="customer",
                        kind="message",
                        data={"message": current.message},
                    )
                )

            elif isinstance(current, AgentMessage):
                # Check if previous step is a typing status
                prev_step = self._steps[i - 1] if i > 0 else None
                prev_is_typing = isinstance(prev_step, StatusEvent) and prev_step.status == "typing"

                # Use human_agent_on_behalf_of_ai_agent since API doesn't allow
                # creating ai_agent messages directly
                agent_source = "human_agent_on_behalf_of_ai_agent"

                # Insert typing if not already there
                if not prev_is_typing:
                    events.append(
                        PrefabEvent(
                            source=agent_source,
                            kind="status",
                            data={"status": "typing", "data": {}},
                        )
                    )

                # Add the agent message
                events.append(
                    PrefabEvent(
                        source=agent_source,
                        kind="message",
                        data={"message": current.ideal},
                    )
                )

                # Check if next step is a ready status
                next_step = self._steps[i + 1] if i + 1 < len(self._steps) else None
                next_is_ready = isinstance(next_step, StatusEvent) and next_step.status == "ready"

                # Insert ready if not already there
                if not next_is_ready:
                    events.append(
                        PrefabEvent(
                            source=agent_source,
                            kind="status",
                            data={"status": "ready", "data": {}},
                        )
                    )

            elif isinstance(current, StatusEvent):
                events.append(
                    PrefabEvent(
                        source="human_agent_on_behalf_of_ai_agent",
                        kind="status",
                        data={"status": current.status, "data": {}},
                    )
                )

            i += 1

        return events

    @classmethod
    def from_steps(cls, steps: Sequence[Step]) -> "InteractionBuilder":
        """Create a builder from an existing sequence of steps."""
        builder = cls()
        for s in steps:
            builder.step(s)
        return builder
