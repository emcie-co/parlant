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

import pytest

from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.core.groups import GroupId
from parlant.core.indexing import (
    IndexingDocumentStore,
    IndexingTaskDocument,
    IndexingTaskTagAssociationDocument,
    PayloadDescriptor,
    PayloadKind,
    PayloadOperation,
    RulePayload,
)


async def test_that_indexing_store_uses_only_indexing_named_collections() -> None:
    database = TransientDocumentDatabase()

    async with IndexingDocumentStore(database) as store:
        task = await store.create_indexing_task(
            payload_descriptors=[
                PayloadDescriptor(
                    PayloadKind.RULE,
                    RulePayload(
                        condition="condition",
                        content="description",
                        operation=PayloadOperation.ADD,
                        signal_proposition=True,
                    ),
                )
            ],
            groups=[GroupId("group-1")],
        )

        tasks = await database.get_collection(
            "indexing_tasks",
            IndexingTaskDocument,
            store.document_loader,
        )
        associations = await database.get_collection(
            "indexing_task_tag_associations",
            IndexingTaskTagAssociationDocument,
            store.tag_association_document_loader,
        )

        assert await tasks.find_one({"id": {"$eq": task.id}})
        association = await associations.find_one({"indexing_task_id": {"$eq": task.id}})
        assert association
        assert association["indexing_task_id"] == task.id

        with pytest.raises(ValueError, match="does not exist"):
            await database.get_collection(
                "evaluations",
                IndexingTaskDocument,
                store.document_loader,
            )
