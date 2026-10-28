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

from pydantic import BaseModel, Field

from parlant.core.services.tools.plugins import tool
from parlant.core.tools import ToolContext, ToolResult


class Address(BaseModel):
    city: str = Field(description="客户所在城市")


def test_that_model_parameter_schemas_preserve_unicode_for_tool_prompts() -> None:
    @tool
    def locate_customer(context: ToolContext, address: Address) -> ToolResult:
        return ToolResult(data=address.city)

    descriptor, _ = locate_customer.tool.parameters["address"]
    description = descriptor["description"]

    assert "客户所在城市" in description
    assert "\\u5ba2" not in description
