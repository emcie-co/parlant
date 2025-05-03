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

from lagom import Container

from parlant.core.agents import Agent
from parlant.core.guidelines import GuidelineContent
from parlant.core.services.indexing.guideline_action_proposer import GuidelineActionProposer
from parlant.core.tools import LocalToolService, ToolId


async def test_that_no_action_is_proposed_when_guideline_already_contains_action_or_no_tools(
    container: Container,
    agent: Agent,
) -> None:
    action_proposer = container[GuidelineActionProposer]

    guideline = GuidelineContent(
        condition="the customer greets the agent",
        action="reply with a greeting",
    )

    result = await action_proposer.propose_action(
        agent=agent,
        guideline=guideline,
        tool_ids=[],
    )

    assert result.content == guideline
    assert result.rationale == "No action proposed"


async def test_that_action_is_proposed_when_guideline_lacks_action_and_tools_are_supplied(
    container: Container,
    agent: Agent,
) -> None:
    local_tool_service = container[LocalToolService]

    dummy_tool = await local_tool_service.create_tool(
        name="dummy_tool",
        module_path="dummy.module",
        description="A dummy testing tool",
        parameters={},
        required=[],
    )

    guideline_without_action = GuidelineContent(
        condition="customer asks for something",
        action=None,
    )

    tool_id = ToolId(service_name="local", tool_name=dummy_tool.name)

    action_proposer = container[GuidelineActionProposer]

    result = await action_proposer.propose_action(
        agent=agent,
        guideline=guideline_without_action,
        tool_ids=[tool_id],
    )

    # Assertions: an action was proposed and it references the tool name
    assert result.content.action is not None
    assert dummy_tool.name in result.content.action
    assert result.content.condition == guideline_without_action.condition
