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

"""How a rule is rendered into model-facing prompts: JSON-safe escaping and
context-variable serialization helpers."""

import json
from typing import Sequence

from parlant.core.common import to_prompt_json
from parlant.core.context_variables import ContextVariable, ContextVariableValue


def escape_json_string(s: str) -> str:
    return json.dumps(s)[1:-1]


def context_variables_to_json(
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
) -> str:
    context_values = {
        variable.name: {
            "value": value.data,
            **({"description": variable.description} if variable.description else {}),
        }
        for variable, value in context_variables
    }

    return to_prompt_json(context_values)
