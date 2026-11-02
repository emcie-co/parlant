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

from typing import Annotated, Sequence, TypeAlias
from fastapi import APIRouter, Path, Request, status

from parlant.api.authorization import AuthorizationPolicy, Operation
from parlant.api.common import ExampleJson, apigen_config, example_json_content
from parlant.core.agents import AgentId
from parlant.core.application import Application
from parlant.core.common import DefaultBaseModel
from pydantic import Field

API_GROUP = "agents"

AgentIdPath: TypeAlias = Annotated[
    AgentId,
    Path(
        description="Unique identifier for the agent",
        examples=["IUCGT-lvpS"],
        min_length=1,
    ),
]

AgentNameField: TypeAlias = Annotated[
    str,
    Field(
        description="The display name of the agent",
        examples=["Haxon", "Alfred J. Quack"],
        min_length=1,
        max_length=100,
    ),
]

agent_example: ExampleJson = {
    "id": "IUCGT-lvpS",
    "name": "Haxon",
}


class AgentDTO(
    DefaultBaseModel,
    json_schema_extra={"example": agent_example},
):
    """The public view of an agent: identity only.

    Agents are configured in code; this API serves frontends that need to
    pick an agent to talk to. It deliberately exposes nothing about the
    agent's configuration.
    """

    id: AgentIdPath
    name: AgentNameField


def create_router(
    policy: AuthorizationPolicy,
    app: Application,
) -> APIRouter:
    router = APIRouter()

    @router.get(
        "",
        operation_id="list_agents",
        response_model=Sequence[AgentDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "List of all agents in the system",
                "content": example_json_content([agent_example]),
            }
        },
        **apigen_config(group_name=API_GROUP, method_name="list"),
    )
    async def list_agents(request: Request) -> Sequence[AgentDTO]:
        """
        Retrieves the id and name of every agent in the system.

        Returns an empty list if no agents exist.
        Agents are returned in no guaranteed order.
        """
        await policy.authorize(
            request=request,
            operation=Operation.LIST_AGENTS,
        )

        agents = await app.agents.find()

        return [AgentDTO(id=a.id, name=a.name) for a in agents]

    @router.get(
        "/{agent_id}",
        operation_id="read_agent",
        response_model=AgentDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "The agent's id and name.",
                "content": example_json_content(agent_example),
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Agent not found. the specified `agent_id` does not exist"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve"),
    )
    async def read_agent(
        request: Request,
        agent_id: AgentIdPath,
    ) -> AgentDTO:
        """
        Retrieves the id and name of a specific agent.
        """
        await policy.authorize(
            request=request,
            operation=Operation.READ_AGENT,
        )

        agent = await app.agents.read(agent_id=agent_id)

        return AgentDTO(id=agent.id, name=agent.name)

    return router
