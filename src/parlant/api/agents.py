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

from datetime import datetime
from enum import Enum
import dateutil
import dateutil.parser
from fastapi import APIRouter, Path, status
from pydantic import Field
from typing import Annotated, Optional, Sequence, TypeAlias

from parlant.api.common import ExampleJson, apigen_config, example_json_content
from parlant.core.agents import AgentId, AgentStore, AgentUpdateParams
from parlant.core.common import DefaultBaseModel

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
        description="The display name of the agent, mainly for management purposes",
        examples=["Haxon", "Alfred J. Quack"],
        min_length=1,
        max_length=100,
    ),
]

AgentDescriptionField: TypeAlias = Annotated[
    str,
    Field(
        default=None,
        description="Detailed description of the agent's purpose and capabilities",
        examples=["Technical Support Assistant"],
    ),
]

AgentCreationUTCField: TypeAlias = Annotated[
    datetime,
    Field(
        description="UTC timestamp of when the agent was created",
        examples=[dateutil.parser.parse("2024-03-24T12:00:00Z")],
    ),
]

AgentMaxEngineIterationsField: TypeAlias = Annotated[
    int,
    Field(
        description="Maximum number of processing iterations the agent can perform per request",
        ge=1,
        examples=[1, 3],
    ),
]

agent_example: ExampleJson = {
    "id": "IUCGT-lvpS",
    "name": "Haxon",
    "description": "Technical Support Assistant",
    "creation_utc": "2024-03-24T12:00:00Z",
    "max_engine_iterations": 3,
}


class CompositionModeDTO(Enum):
    """
    Defines the composition mode for an entity.

    Available options:
    - fluid
    - strict_assembly
    - composited_assembly
    - fluid_assembly
    """

    FLUID = "fluid"
    STRICT_ASSEMBLY = "strict_assembly"
    COMPOSITED_ASSEMBLY = "composited_assembly"
    FLUID_ASSEMBLY = "fluid_assembly"


class AgentDTO(
    DefaultBaseModel,
    json_schema_extra={"example": agent_example},
):
    """
    An agent is a specialized AI personality crafted for a specific service role.

    Agents form the basic unit of conversational customization: all behavioral configurations
    are made at the agent level.

    Use this model for representing complete agent information in API responses.
    """

    id: AgentIdPath
    name: AgentNameField
    description: Optional[AgentDescriptionField] = None
    creation_utc: AgentCreationUTCField
    max_engine_iterations: AgentMaxEngineIterationsField
    composition_mode: CompositionModeDTO


agent_creation_params_example: ExampleJson = {
    "name": "Haxon",
    "description": "Technical Support Assistant",
    "max_engine_iterations": 3,
}


class AgentCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": agent_creation_params_example},
):
    """
    Parameters for creating a new agent.

    Optional fields:
    - `description`: Detailed explanation of the agent's purpose
    - `max_engine_iterations`: Processing limit per request

    Note: Agents must be created via the API before they can be used.
    """

    name: AgentNameField
    description: Optional[AgentDescriptionField] = None
    max_engine_iterations: Optional[AgentMaxEngineIterationsField] = None


agent_update_params_example: ExampleJson = {
    "name": "Haxon",
    "description": "Technical Support Assistant",
    "max_engine_iterations": 3,
}


class AgentUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": agent_update_params_example},
):
    """
    Parameters for updating an existing agent.

    All fields are optional. only provided fields will be updated.
    The agent's ID and creation timestamp cannot be modified.
    """

    name: Optional[AgentNameField] = None
    description: Optional[AgentDescriptionField] = None
    max_engine_iterations: Optional[AgentMaxEngineIterationsField] = None
    composition_mode: Optional[CompositionModeDTO] = None


def create_router(
    agent_store: AgentStore,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_agent",
        response_model=AgentDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Agent successfully created. Returns the complete agent object including generated ID.",
                "content": example_json_content(agent_example),
            },
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create"),
    )
    async def create_agent(
        params: AgentCreationParamsDTO,
    ) -> AgentDTO:
        """
        Creates a new agent in the system.

        The agent will be initialized with the provided name and optional settings.
        A unique identifier will be automatically generated.

        Default behaviors:
        - `name` defaults to `"Unnamed Agent"` if not provided
        - `description` defaults to `None`
        - `max_engine_iterations` defaults to `None` (uses system default)
        """
        agent = await agent_store.create_agent(
            name=params and params.name or "Unnamed Agent",
            description=params and params.description or None,
            max_engine_iterations=params and params.max_engine_iterations or None,
        )

        return AgentDTO(
            id=agent.id,
            name=agent.name,
            description=agent.description,
            creation_utc=agent.creation_utc,
            max_engine_iterations=agent.max_engine_iterations,
            composition_mode=CompositionModeDTO(agent.composition_mode),
        )

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
    async def list_agents() -> Sequence[AgentDTO]:
        """
        Retrieves a list of all agents in the system.

        Returns an empty list if no agents exist.
        Agents are returned in no guaranteed order.
        """
        agents = await agent_store.list_agents()

        return [
            AgentDTO(
                id=a.id,
                name=a.name,
                description=a.description,
                creation_utc=a.creation_utc,
                max_engine_iterations=a.max_engine_iterations,
                composition_mode=CompositionModeDTO(a.composition_mode),
            )
            for a in agents
        ]

    @router.get(
        "/{agent_id}",
        operation_id="read_agent",
        response_model=AgentDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Agent details successfully retrieved. Returns the complete agent object.",
                "content": example_json_content(agent_example),
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Agent not found. the specified `agent_id` does not exist"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve"),
    )
    async def read_agent(
        agent_id: AgentIdPath,
    ) -> AgentDTO:
        """
        Retrieves details of a specific agent by ID.
        """
        agent = await agent_store.read_agent(agent_id=agent_id)

        return AgentDTO(
            id=agent.id,
            name=agent.name,
            description=agent.description,
            creation_utc=agent.creation_utc,
            max_engine_iterations=agent.max_engine_iterations,
            composition_mode=CompositionModeDTO(agent.composition_mode),
        )

    @router.patch(
        "/{agent_id}",
        operation_id="update_agent",
        response_model=AgentDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Agent successfully updated. Returns the updated agent.",
                "content": example_json_content(agent_example),
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Agent not found. the specified `agent_id` does not exist"
            },
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in update parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="update"),
    )
    async def update_agent(
        agent_id: AgentIdPath,
        params: AgentUpdateParamsDTO,
    ) -> AgentDTO:
        """
        Updates an existing agent's attributes.

        Only the provided attributes will be updated; others will remain unchanged.
        The agent's ID and creation timestamp cannot be modified.
        """

        def from_dto(dto: AgentUpdateParamsDTO) -> AgentUpdateParams:
            params: AgentUpdateParams = {}

            if dto.name:
                params["name"] = dto.name

            if dto.description:
                params["description"] = dto.description

            if dto.max_engine_iterations:
                params["max_engine_iterations"] = dto.max_engine_iterations

            if dto.composition_mode:
                params["composition_mode"] = dto.composition_mode.value

            return params

        agent = await agent_store.update_agent(
            agent_id=agent_id,
            params=from_dto(params),
        )
        return AgentDTO(
            id=agent.id,
            name=agent.name,
            description=agent.description,
            creation_utc=agent.creation_utc,
            max_engine_iterations=agent.max_engine_iterations,
            composition_mode=CompositionModeDTO(agent.composition_mode),
        )

    @router.delete(
        "/{agent_id}",
        operation_id="delete_agent",
        status_code=status.HTTP_204_NO_CONTENT,
        responses={
            status.HTTP_204_NO_CONTENT: {
                "description": "Agent successfully deleted. No content returned."
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Agent not found. The specified `agent_id` does not exist"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="delete"),
    )
    async def delete_agent(
        agent_id: AgentIdPath,
    ) -> None:
        """
        Deletes an agent from the agent.

        Deleting a non-existent agent will return 404.
        No content will be returned from a successful deletion.
        """
        await agent_store.read_agent(agent_id=agent_id)

        await agent_store.delete_agent(agent_id=agent_id)

    return router
