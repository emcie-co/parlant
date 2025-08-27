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

from enum import Enum
from fastapi import APIRouter, Path, Request, status
from pydantic import Field
from typing import Annotated, Optional, Sequence, TypeAlias

from parlant.api.authorization import AuthorizationPolicy, Operation
from parlant.api.common import ExampleJson, apigen_config, example_json_content
from parlant.core.agents import AgentId, AgentStore, AgentUpdateParams, CompositionMode
from parlant.core.common import DefaultBaseModel
from parlant.core.tags import TagId, TagStore

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

AgentMaxEngineIterationsField: TypeAlias = Annotated[
    int,
    Field(
        default=1,
        description="Maximum number of processing iterations the agent can perform per request",
        ge=1,
        examples=[1, 3],
    ),
]

AgentTagsField: TypeAlias = Annotated[
    list[TagId],
    Field(
        default=None,
        description="List of tag IDs associated with the agent",
        examples=[["tag1", "tag2"]],
    ),
]

AgentTagUpdateAddField: TypeAlias = Annotated[
    list[TagId],
    Field(
        default=None,
        description="List of tag IDs to add to the agent",
        examples=[["tag1", "tag2"]],
    ),
]

AgentTagUpdateRemoveField: TypeAlias = Annotated[
    list[TagId],
    Field(
        default=None,
        description="List of tag IDs to remove from the agent",
        examples=[["tag1", "tag2"]],
    ),
]
agent_example: ExampleJson = {
    "id": "IUCGT-lvpS",
    "name": "Haxon",
    "description": "Technical Support Assistant",
    "creation_utc": "2024-03-24T12:00:00Z",
    "max_engine_iterations": 3,
    "composition_mode": "fluid",
    "tags": ["tag1", "tag2"],
}


class CompositionModeDTO(Enum):
    """
    Defines the composition mode for an entity.

    Available options:
    - fluid
    - canned_fluid
    - composited_canned
    - strict_canned
    """

    FLUID = "fluid"
    CANNED_FLUID = "canned_fluid"
    CANNED_COMPOSITED = "composited_canned"
    CANNED_STRICT = "strict_canned"


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
    max_engine_iterations: AgentMaxEngineIterationsField
    composition_mode: CompositionModeDTO
    tags: AgentTagsField


agent_creation_params_example: ExampleJson = {
    "name": "Haxon",
    "description": "Technical Support Assistant",
    "max_engine_iterations": 3,
    "composition_mode": "fluid",
    "tags": ["tag1", "tag2"],
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
    composition_mode: Optional[CompositionModeDTO] = None
    tags: Optional[AgentTagsField] = None


agent_update_params_example: ExampleJson = {
    "name": "Haxon",
    "description": "Technical Support Assistant",
    "max_engine_iterations": 3,
    "composition_mode": "fluid",
}


tags_update_params_example: ExampleJson = {
    "add": [
        "t9a8g703f4",
        "tag_456abc",
    ],
    "remove": [
        "tag_789def",
        "tag_012ghi",
    ],
}


class AgentTagUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": tags_update_params_example},
):
    """
    Parameters for updating an existing agent's tags.
    """

    add: Optional[AgentTagUpdateAddField] = None
    remove: Optional[AgentTagUpdateRemoveField] = None


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
    tags: Optional[AgentTagUpdateParamsDTO] = None


def _composition_mode_dto_to_composition_mode(dto: CompositionModeDTO) -> CompositionMode:
    match dto:
        case CompositionModeDTO.FLUID:
            return CompositionMode.FLUID
        case CompositionModeDTO.CANNED_STRICT:
            return CompositionMode.CANNED_STRICT
        case CompositionModeDTO.CANNED_COMPOSITED:
            return CompositionMode.CANNED_COMPOSITED
        case CompositionModeDTO.CANNED_FLUID:
            return CompositionMode.CANNED_FLUID


def _composition_mode_to_composition_mode_dto(
    composition_mode: CompositionMode,
) -> CompositionModeDTO:
    match composition_mode:
        case CompositionMode.FLUID:
            return CompositionModeDTO.FLUID
        case CompositionMode.CANNED_STRICT:
            return CompositionModeDTO.CANNED_STRICT
        case CompositionMode.CANNED_COMPOSITED:
            return CompositionModeDTO.CANNED_COMPOSITED
        case CompositionMode.CANNED_FLUID:
            return CompositionModeDTO.CANNED_FLUID


def create_router(
    policy: AuthorizationPolicy,
    agent_store: AgentStore,
    tag_store: TagStore,
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
        request: Request,
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
        await policy.authorize(
            request=request,
            operation=Operation.CREATE_AGENT,
        )

        tags = []

        if params.tags:
            for tag_id in params.tags:
                _ = await tag_store.read_tag(tag_id=tag_id)

            tags = list(set(params.tags))

        agent = await agent_store.create_agent(
            name=params and params.name or "Unnamed Agent",
            description=params and params.description or None,
            max_engine_iterations=params and params.max_engine_iterations or None,
            composition_mode=_composition_mode_dto_to_composition_mode(params.composition_mode)
            if params and params.composition_mode
            else None,
            tags=tags or None,
        )

        return AgentDTO(
            id=agent.id,
            name=agent.name,
            description=agent.description,
            creation_utc=agent.creation_utc,
            max_engine_iterations=agent.max_engine_iterations,
            composition_mode=_composition_mode_to_composition_mode_dto(agent.composition_mode),
            tags=agent.tags,
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
    async def list_agents(request: Request) -> Sequence[AgentDTO]:
        """
        Retrieves a list of all agents in the system.

        Returns an empty list if no agents exist.
        Agents are returned in no guaranteed order.
        """
        await policy.authorize(
            request=request,
            operation=Operation.LIST_AGENTS,
        )

        agents = await agent_store.list_agents()

        return [
            AgentDTO(
                id=a.id,
                name=a.name,
                description=a.description,
                creation_utc=a.creation_utc,
                max_engine_iterations=a.max_engine_iterations,
                composition_mode=_composition_mode_to_composition_mode_dto(a.composition_mode),
                tags=a.tags,
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
        request: Request,
        agent_id: AgentIdPath,
    ) -> AgentDTO:
        """
        Retrieves details of a specific agent by ID.
        """
        await policy.authorize(
            request=request,
            operation=Operation.READ_AGENT,
        )

        agent = await agent_store.read_agent(agent_id=agent_id)

        if await policy.check_permission(request, Operation.READ_AGENT_DESCRIPTION):
            description = agent.description
        else:
            description = None

        return AgentDTO(
            id=agent.id,
            name=agent.name,
            description=description,
            creation_utc=agent.creation_utc,
            max_engine_iterations=agent.max_engine_iterations,
            composition_mode=_composition_mode_to_composition_mode_dto(agent.composition_mode),
            tags=agent.tags,
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
        request: Request,
        agent_id: AgentIdPath,
        params: AgentUpdateParamsDTO,
    ) -> AgentDTO:
        """
        Updates an existing agent's attributes.

        Only the provided attributes will be updated; others will remain unchanged.
        The agent's ID and creation timestamp cannot be modified.
        """
        await policy.authorize(
            request=request,
            operation=Operation.UPDATE_AGENT,
        )

        def from_dto(dto: AgentUpdateParamsDTO) -> AgentUpdateParams:
            params: AgentUpdateParams = {}

            if dto.name:
                params["name"] = dto.name

            if dto.description:
                params["description"] = dto.description

            if dto.max_engine_iterations:
                params["max_engine_iterations"] = dto.max_engine_iterations

            if dto.composition_mode:
                params["composition_mode"] = _composition_mode_dto_to_composition_mode(
                    dto.composition_mode
                )

            return params

        if params.tags:
            if params.tags.add:
                for tag_id in params.tags.add:
                    _ = await tag_store.read_tag(tag_id=tag_id)

                    await agent_store.upsert_tag(
                        agent_id=agent_id,
                        tag_id=tag_id,
                    )

            if params.tags.remove:
                for tag_id in params.tags.remove:
                    await agent_store.remove_tag(
                        agent_id=agent_id,
                        tag_id=tag_id,
                    )

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
            composition_mode=_composition_mode_to_composition_mode_dto(agent.composition_mode),
            tags=agent.tags,
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
        request: Request,
        agent_id: AgentIdPath,
    ) -> None:
        """
        Deletes an agent from the agent.

        Deleting a non-existent agent will return 404.
        No content will be returned from a successful deletion.
        """
        await policy.authorize(
            request=request,
            operation=Operation.DELETE_AGENT,
        )

        await agent_store.read_agent(agent_id=agent_id)

        await agent_store.delete_agent(agent_id=agent_id)

    return router
