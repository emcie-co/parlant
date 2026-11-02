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

from contextlib import AsyncExitStack
from pathlib import Path
from typing import AsyncIterator

import pytest

from parlant.adapters.db.json_file import JSONFileDocumentDatabase
from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.core.common import IdGenerator
from parlant.core.context_variables import ContextVariableDocumentStore
from parlant.core.loggers import CompositeLogger


@pytest.fixture(params=["transient", "json"])
async def store(
    request: pytest.FixtureRequest, tmp_path: Path
) -> AsyncIterator[ContextVariableDocumentStore]:
    async with AsyncExitStack() as stack:
        database = (
            await stack.enter_async_context(
                JSONFileDocumentDatabase(CompositeLogger([]), tmp_path / "context.json")
            )
            if request.param == "json"
            else TransientDocumentDatabase()
        )
        yield await stack.enter_async_context(ContextVariableDocumentStore(IdGenerator(), database))


@pytest.mark.parametrize("rule", ["not a cron expression", "61 * * * *"])
async def test_that_invalid_freshness_rules_are_rejected_on_creation(
    store: ContextVariableDocumentStore, rule: str
) -> None:
    with pytest.raises(ValueError, match="(?i)freshness|cron"):
        await store.create_variable(name="Invalid", freshness_rules=rule)


@pytest.mark.parametrize("rule", ["not a cron expression", "61 * * * *"])
async def test_that_invalid_freshness_rules_are_rejected_before_an_update_changes_data(
    store: ContextVariableDocumentStore, rule: str
) -> None:
    variable = await store.create_variable(name="Original", freshness_rules="*/5 * * * *")
    with pytest.raises(ValueError, match="(?i)freshness|cron"):
        await store.update_variable(variable.id, {"name": "Changed", "freshness_rules": rule})
    assert await store.read_variable(variable.id) == variable


@pytest.mark.parametrize("rule", [None, "", "*/5 * * * *"])
async def test_that_valid_or_unset_freshness_rules_keep_existing_behavior(
    store: ContextVariableDocumentStore, rule: str | None
) -> None:
    variable = await store.create_variable(name="Original")
    updated = await store.update_variable(variable.id, {"freshness_rules": rule})
    assert updated.freshness_rules == (rule or None)
