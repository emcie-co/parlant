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

"""
Store-specific tests for playbooks.

Basic CRUD operations are tested via API tests in tests/api/test_playbooks.py.
This file contains only store-specific behavior tests:
- Tag helper utility functions
- Idempotency behavior (return values for duplicate operations)
"""

from typing import AsyncIterator

from pytest import fixture

from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.core.common import IdGenerator
from parlant.core.playbooks import (
    DisabledRuleRef,
    PlaybookDocumentStore,
    PlaybookStore,
)
from parlant.core.tags import TagId, Tag


@fixture
async def playbook_store(
    id_generator: IdGenerator,
) -> AsyncIterator[PlaybookStore]:
    async with PlaybookDocumentStore(
        id_generator=id_generator,
        database=TransientDocumentDatabase(),
    ) as store:
        yield store


@fixture
def id_generator() -> IdGenerator:
    return IdGenerator()


async def test_that_adding_duplicate_disabled_rule_returns_false(
    playbook_store: PlaybookStore,
) -> None:
    playbook = await playbook_store.create_playbook(name="Playbook with disabled rules")
    rule_ref = DisabledRuleRef("guideline:abc123")

    first_add = await playbook_store.add_disabled_rule(playbook.id, rule_ref)
    second_add = await playbook_store.add_disabled_rule(playbook.id, rule_ref)

    assert first_add is True
    assert second_add is False

    updated_playbook = await playbook_store.read_playbook(playbook.id)
    # Should only have one instance
    assert updated_playbook.disabled_rules.count(rule_ref) == 1


async def test_that_upserting_existing_tag_returns_false(
    playbook_store: PlaybookStore,
) -> None:
    tag_id = TagId("existing-tag")
    playbook = await playbook_store.create_playbook(
        name="Tagged Playbook",
        tags=[tag_id],
    )

    result = await playbook_store.upsert_tag(playbook.id, tag_id)

    assert result is False


async def test_that_playbook_tag_helper_creates_correct_format() -> None:
    playbook_id = "pb_123"
    tag_id = Tag.for_playbook_id(playbook_id)

    assert tag_id == TagId("playbook:pb_123")


async def test_that_playbook_id_can_be_extracted_from_tag() -> None:
    tag_id = TagId("playbook:pb_456")
    extracted = Tag.extract_playbook_id(tag_id)

    assert extracted == "pb_456"


async def test_that_extract_playbook_id_returns_none_for_non_playbook_tag() -> None:
    tag_id = TagId("agent:ag_123")
    extracted = Tag.extract_playbook_id(tag_id)

    assert extracted is None
