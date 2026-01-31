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
from typing import Optional, Sequence

from parlant.core.loggers import Logger
from parlant.core.playbooks import (
    DisabledRuleRef,
    Playbook,
    PlaybookId,
    PlaybookStore,
    PlaybookUpdateParams,
)
from parlant.core.tags import TagId, TagStore


# Sentinel value to distinguish "not provided" from "explicitly set to None"
_UNSET: object = object()


@dataclass(frozen=True)
class PlaybookTagUpdateParamsModel:
    add: list[TagId] | None = None
    remove: list[TagId] | None = None


@dataclass(frozen=True)
class PlaybookDisabledRulesUpdateParamsModel:
    add: list[DisabledRuleRef] | None = None
    remove: list[DisabledRuleRef] | None = None


class PlaybookModule:
    def __init__(
        self,
        logger: Logger,
        playbook_store: PlaybookStore,
        tag_store: TagStore,
    ):
        self._logger = logger
        self._playbook_store = playbook_store
        self._tag_store = tag_store

    async def _ensure_tag(self, tag_id: TagId) -> None:
        await self._tag_store.read_tag(tag_id)

    async def _validate_parent(self, parent_id: Optional[PlaybookId]) -> None:
        if parent_id:
            await self._playbook_store.read_playbook(parent_id)

    async def _detect_circular_inheritance(
        self, playbook_id: PlaybookId, new_parent_id: PlaybookId
    ) -> bool:
        """Returns True if setting new_parent_id would create a cycle."""
        seen = {playbook_id}
        current_id: Optional[PlaybookId] = new_parent_id

        while current_id:
            if current_id in seen:
                return True
            seen.add(current_id)
            playbook = await self._playbook_store.read_playbook(current_id)
            current_id = playbook.parent_id

        return False

    async def create(
        self,
        name: str,
        description: str | None = None,
        parent_id: PlaybookId | None = None,
        tags: list[TagId] | None = None,
        id: PlaybookId | None = None,
    ) -> Playbook:
        await self._validate_parent(parent_id)

        if tags:
            for tag_id in tags:
                await self._ensure_tag(tag_id)
            tags = list(set(tags))

        playbook = await self._playbook_store.create_playbook(
            name=name,
            description=description,
            parent_id=parent_id,
            tags=tags,
            id=id,
        )
        return playbook

    async def read(self, playbook_id: PlaybookId) -> Playbook:
        playbook = await self._playbook_store.read_playbook(playbook_id=playbook_id)
        return playbook

    async def find(self) -> Sequence[Playbook]:
        playbooks = await self._playbook_store.list_playbooks()
        return playbooks

    async def update(
        self,
        playbook_id: PlaybookId,
        name: str | None = None,
        description: str | None = None,
        parent_id: PlaybookId | None = _UNSET,  # type: ignore[assignment]
        tags: PlaybookTagUpdateParamsModel | None = None,
        disabled_rules: PlaybookDisabledRulesUpdateParamsModel | None = None,
    ) -> Playbook:
        if parent_id is not _UNSET and parent_id is not None:
            await self._validate_parent(parent_id)
            if await self._detect_circular_inheritance(playbook_id, parent_id):
                raise ValueError("Circular inheritance detected")

        update_params: PlaybookUpdateParams = {}

        if name:
            update_params["name"] = name

        if description is not None:
            update_params["description"] = description

        # Handle parent_id: _UNSET means "don't change", None means "clear it"
        if parent_id is not _UNSET:
            update_params["parent_id"] = parent_id

        if update_params:
            await self._playbook_store.update_playbook(
                playbook_id=playbook_id, params=update_params
            )

        if tags:
            if tags.add:
                for tag_id in tags.add:
                    await self._ensure_tag(tag_id)
                    await self._playbook_store.upsert_tag(
                        playbook_id=playbook_id,
                        tag_id=tag_id,
                    )

            if tags.remove:
                for tag_id in tags.remove:
                    await self._playbook_store.remove_tag(
                        playbook_id=playbook_id,
                        tag_id=tag_id,
                    )

        if disabled_rules:
            if disabled_rules.add:
                for rule_ref in disabled_rules.add:
                    await self._playbook_store.add_disabled_rule(
                        playbook_id=playbook_id,
                        rule_ref=rule_ref,
                    )

            if disabled_rules.remove:
                for rule_ref in disabled_rules.remove:
                    await self._playbook_store.remove_disabled_rule(
                        playbook_id=playbook_id,
                        rule_ref=rule_ref,
                    )

        playbook = await self._playbook_store.read_playbook(playbook_id)

        return playbook

    async def delete(self, playbook_id: PlaybookId) -> None:
        # Check for child playbooks
        all_playbooks = await self._playbook_store.list_playbooks()
        children = [p for p in all_playbooks if p.parent_id == playbook_id]
        if children:
            raise ValueError(f"Cannot delete playbook with children: {[c.name for c in children]}")

        await self._playbook_store.delete_playbook(playbook_id=playbook_id)

    async def get_inheritance_chain(self, playbook_id: PlaybookId) -> Sequence[Playbook]:
        """Returns chain from root to child (least to most specific)."""
        chain: list[Playbook] = []
        current_id: Optional[PlaybookId] = playbook_id
        seen: set[PlaybookId] = set()

        while current_id and current_id not in seen:
            seen.add(current_id)
            playbook = await self._playbook_store.read_playbook(current_id)
            chain.append(playbook)
            current_id = playbook.parent_id

        return list(reversed(chain))
