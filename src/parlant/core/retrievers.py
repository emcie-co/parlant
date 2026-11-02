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

from dataclasses import dataclass, replace
from typing import Literal

from parlant.core.common import generate_id


@dataclass(frozen=True)
class RetrieverOwner:
    kind: Literal["agent", "rule"]
    id: str


@dataclass(frozen=True)
class RetrieverRegistration:
    id: str
    name: str
    title: str | None
    description: str | None
    owner: RetrieverOwner


class RetrieverRegistry:
    """Process-local metadata for executable attachments owned by SDK code.

    Callbacks stay in the SDK's maps and hooks. Agent/name replacement retains
    identity; each rule hook remains a distinct attachment, even for equal names.
    A catalog is a snapshot of this container, not historical execution evidence.
    """

    def __init__(self) -> None:
        self._registrations: dict[str, RetrieverRegistration] = {}
        self._agent_ids: dict[tuple[str, str], str] = {}

    def register_agent(
        self, agent_id: str, name: str, *, title: str | None, description: str | None
    ) -> RetrieverRegistration:
        key = (agent_id, name)
        if registration_id := self._agent_ids.get(key):
            registration = replace(
                self._registrations[registration_id], title=title, description=description
            )
            self._registrations[registration_id] = registration
            return registration
        registration = self._register(RetrieverOwner("agent", agent_id), name, title, description)
        self._agent_ids[key] = registration.id
        return registration

    def register_rule(
        self, rule_id: str, name: str, *, title: str | None, description: str | None
    ) -> RetrieverRegistration:
        return self._register(RetrieverOwner("rule", rule_id), name, title, description)

    def _register(
        self, owner: RetrieverOwner, name: str, title: str | None, description: str | None
    ) -> RetrieverRegistration:
        registration = RetrieverRegistration(generate_id(), name, title, description, owner)
        self._registrations[registration.id] = registration
        return registration

    def list(self) -> tuple[RetrieverRegistration, ...]:
        return tuple(self._registrations.values())
