# Copyright 2025 Emcie Co Ltd.
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
from parlant.core.canned_responses import CannedResponseStore, CannedResponseId, CannedResponseField

from tests.core.common.engines.alpha.utils import step
from tests.core.common.utils import ContextOfTest


@step(given, parsers.parse('a canned response, "{text}"'))
def given_a_canned_response(
    context: ContextOfTest,
    text: str,
) -> CannedResponseId:
    can_rep_store = context.container[CannedResponseStore]

    can_rep_field_pattern = r"\{(.*?)\}"
    field_names = re.findall(can_rep_field_pattern, text)

    can_rep = context.sync_await(
        can_rep_store.create_can_rep(
            value=text,
            fields=[
                CannedResponseField(
                    name=can_rep_field_name,
                    description="",
                    examples=[],
                )
                for can_rep_field_name in field_names
            ],
        )
    )

    return can_rep.id
