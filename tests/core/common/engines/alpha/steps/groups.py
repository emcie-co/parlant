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

from pytest_bdd import given, parsers
from parlant.core.groups import GroupStore, GroupId

from tests.core.common.engines.alpha.utils import step
from tests.core.common.utils import ContextOfTest


@step(given, parsers.parse('a group "{group_name}"'))
def given_a_tag(
    context: ContextOfTest,
    group_name: str,
) -> GroupId:
    group_store = context.container[GroupStore]

    group = context.sync_await(group_store.create_group(group_name))

    return group.id
