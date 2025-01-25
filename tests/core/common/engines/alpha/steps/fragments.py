# Copyright 2024 Emcie Co Ltd.
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

import re
from pytest_bdd import given, parsers
from parlant.core.fragments import FragmentStore, FragmentId, Slot

from tests.core.common.engines.alpha.utils import step
from tests.core.common.utils import ContextOfTest


@step(given, parsers.parse('a fragment, "{text}"'))
def given_a_fragment(
    context: ContextOfTest,
    text: str,
) -> FragmentId:
    fragment_store = context.container[FragmentStore]

    slot_pattern = r"\{(.*?)\}"
    slot_names = re.findall(slot_pattern, text)

    fragment = context.sync_await(
        fragment_store.create_fragment(
            value=text,
            slots=[
                Slot(
                    name=slot_name,
                    description="",
                    examples=[],
                )
                for slot_name in slot_names
            ],
        )
    )

    return fragment.id
