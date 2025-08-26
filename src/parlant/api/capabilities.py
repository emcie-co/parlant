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

from fastapi import APIRouter, Path, Query, Request, status
from pydantic import Field
from typing import Annotated, Optional, Sequence, TypeAlias

from parlant.api.authorization import AuthorizationPolicy, Operation
from parlant.core.agents import AgentId, AgentStore
from parlant.core.common import DefaultBaseModel
from parlant.api.common import ExampleJson, apigen_config, example_json_content
from parlant.core.capabilities import (
    CapabilityId,
    CapabilityStore,
    CapabilityUpdateParams,
)
from parlant.core.journeys import JourneyId, JourneyStore
from parlant.core.tags import Tag, TagId, TagStore

API_GROUP = "capabilities"

CapabilityIdPath: TypeAlias = Annotated[
    CapabilityId,
    Path(
        description="Unique identifier for the capability",
        examples=["cap_123abc"],
        min_length=1,
    ),
]

CapabilityTitleField: TypeAlias = Annotated[
    str,
    Field(
        description="The title of the capability",
        examples=["Reset password", "Replace phone"],
        min_length=1,
        max_length=100,
    ),
]

CapabilityDescriptionField: TypeAlias = Annotated[
    str,
    Field(
        description="Detailed description of the capability's purpose",
        examples=["Provide a weather update"],
    ),
]

CapabilitySignalsField: TypeAlias = Annotated[
    Sequence[str],
    Field(
        description="Example signals that this capability can handle",
        examples=[["I thought I remembered my password", "My phone just broke"]],
    ),
]

CapabilityTagsField: TypeAlias = Annotated[
    list[TagId],
    Field(
        default=None,
        description="List of tag IDs associated with the capability",
        examples=[["tag1", "tag2"]],
    ),
]

capability_example: ExampleJson = {
    "id": "cap_123abc",
    "title": "Provide Replacement Phone",
    "description": "Provide a replacement phone when a customer needs repair for their phone.",
    "signals": ["My phone is broken", "I need a replacement while my phone is being repaired"],
    "tags": ["tag1", "tag2"],
}


class CapabilityDTO(
    DefaultBaseModel,
    json_schema_extra={"example": capability_example},
):
    """
    A capability represents a functional feature or skill of the agent.
    """

    id: CapabilityIdPath
    title: CapabilityTitleField
    description: CapabilityDescriptionField
    signals: CapabilitySignalsField
    tags: CapabilityTagsField


class CapabilityCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": capability_example},
):
    """
    Parameters for creating a new capability.
    """

    title: CapabilityTitleField
    description: CapabilityDescriptionField
    signals: CapabilitySignalsField
    tags: Optional[CapabilityTagsField] = None


CapabilityTagUpdateAddField: TypeAlias = Annotated[
    list[TagId],
    Field(
        default=None,
        description="List of tag IDs to add to the capability",
        examples=[["tag1", "tag2"]],
    ),
]

CapabilityTagUpdateRemoveField: TypeAlias = Annotated[
    list[TagId],
    Field(
        default=None,
        description="List of tag IDs to remove from the capability",
        examples=[["tag1", "tag2"]],
    ),
]

capability_tag_update_params_example: ExampleJson = {
    "add": ["tag1", "tag2"],
    "remove": ["tag3"],
}


class CapabilityTagUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": capability_tag_update_params_example},
):
    """
    Parameters for updating an existing capability's tags.
    """

    add: Optional[CapabilityTagUpdateAddField] = None
    remove: Optional[CapabilityTagUpdateRemoveField] = None


class CapabilityUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": capability_example},
):
    """
    Parameters for updating an existing capability.
    All fields are optional. Only provided fields will be updated.
    """

    title: Optional[CapabilityTitleField] = None
    description: Optional[CapabilityDescriptionField] = None
    signals: Optional[CapabilitySignalsField] = None
    tags: Optional[CapabilityTagUpdateParamsDTO] = None


TagIdQuery: TypeAlias = Annotated[
    Optional[TagId],
    Query(
        description="The tag ID to filter capabilities by",
        examples=["tag:123"],
    ),
]


def create_router(
    authorization_policy: AuthorizationPolicy,
    capability_store: CapabilityStore,
    tag_store: TagStore,
    agent_store: AgentStore,
    journey_store: JourneyStore,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_capability",
        response_model=CapabilityDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Capability successfully created. Returns the complete capability object including generated ID.",
                "content": example_json_content(capability_example),
            },
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create"),
    )
    async def create_capability(
        request: Request,
        params: CapabilityCreationParamsDTO,
    ) -> CapabilityDTO:
        """
        Creates a new capability in the system.

        The capability will be initialized with the provided title, description, signals, and optional tags.
        A unique identifier will be automatically generated.

        Default behaviors:
        - `signals` defaults to an empty list if not provided
        """
        await authorization_policy.authorize(request, Operation.CREATE_CAPABILITY)

        if params.tags:
            for tag_id in params.tags:
                if agent_id := Tag.extract_agent_id(tag_id):
                    _ = await agent_store.read_agent(agent_id=AgentId(agent_id))
                elif journey_id := Tag.extract_journey_id(tag_id):
                    _ = await journey_store.read_journey(journey_id=JourneyId(journey_id))
                else:
                    _ = await tag_store.read_tag(tag_id=tag_id)

        capability = await capability_store.create_capability(
            title=params.title,
            description=params.description,
            signals=params.signals,
            tags=params.tags if params.tags else None,
        )

        return CapabilityDTO(
            id=capability.id,
            title=capability.title,
            description=capability.description,
            signals=capability.signals,
            tags=capability.tags,
        )

    @router.get(
        "",
        operation_id="list_capabilities",
        response_model=Sequence[CapabilityDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "List of all capabilities in the system",
                "content": example_json_content([capability_example]),
            }
        },
        **apigen_config(group_name=API_GROUP, method_name="list"),
    )
    async def list_capabilities(
        request: Request,
        tag_id: TagIdQuery = None,
    ) -> Sequence[CapabilityDTO]:
        """
        Retrieves a list of all capabilities in the system.

        Returns an empty list if no capabilities exist.
        Capabilities are returned in no guaranteed order.
        """
        await authorization_policy.authorize(request, Operation.LIST_CAPABILITIES)

        if tag_id:
            capabilities = await capability_store.list_capabilities(
                tags=[tag_id],
            )
        else:
            capabilities = await capability_store.list_capabilities()

        return [
            CapabilityDTO(
                id=capability.id,
                title=capability.title,
                description=capability.description,
                signals=capability.signals,
                tags=capability.tags,
            )
            for capability in capabilities
        ]

    @router.get(
        "/{capability_id}",
        operation_id="read_capability",
        response_model=CapabilityDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Capability details successfully retrieved. Returns the complete capability object.",
                "content": example_json_content(capability_example),
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Capability not found. The specified `capability_id` does not exist"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve"),
    )
    async def read_capability(
        request: Request,
        capability_id: CapabilityIdPath,
    ) -> CapabilityDTO:
        """
        Retrieves details of a specific capability by ID.

        Returns the complete capability object.
        """
        await authorization_policy.authorize(request, Operation.READ_CAPABILITY)

        capability = await capability_store.read_capability(capability_id=capability_id)

        return CapabilityDTO(
            id=capability.id,
            title=capability.title,
            description=capability.description,
            signals=capability.signals,
            tags=capability.tags,
        )

    @router.patch(
        "/{capability_id}",
        operation_id="update_capability",
        response_model=CapabilityDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Capability successfully updated. Returns the updated capability.",
                "content": example_json_content(capability_example),
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Capability not found. The specified `capability_id` does not exist"
            },
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in update parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="update"),
    )
    async def update_capability(
        request: Request,
        capability_id: CapabilityIdPath,
        params: CapabilityUpdateParamsDTO,
    ) -> CapabilityDTO:
        """
        Updates an existing capability's attributes.

        Only the provided attributes will be updated; others will remain unchanged.
        The capability's ID and creation timestamp cannot be modified.
        """
        await authorization_policy.authorize(request, Operation.UPDATE_CAPABILITY)

        update_params: CapabilityUpdateParams = {}
        if params.title:
            update_params["title"] = params.title
        if params.description:
            update_params["description"] = params.description
        if params.signals:
            update_params["signals"] = params.signals

        if update_params:
            capability = await capability_store.update_capability(
                capability_id=capability_id,
                params=update_params,
            )

        else:
            capability = await capability_store.read_capability(capability_id=capability_id)

        if params.tags:
            if params.tags.add:
                for tag_id in params.tags.add:
                    if agent_id := Tag.extract_agent_id(tag_id):
                        _ = await agent_store.read_agent(agent_id=AgentId(agent_id))
                    elif journey_id := Tag.extract_journey_id(tag_id):
                        _ = await journey_store.read_journey(journey_id=JourneyId(journey_id))
                    else:
                        _ = await tag_store.read_tag(tag_id=tag_id)

                    await capability_store.upsert_tag(capability_id=capability_id, tag_id=tag_id)

            if params.tags.remove:
                for tag_id in params.tags.remove:
                    await capability_store.remove_tag(capability_id=capability_id, tag_id=tag_id)

        capability = await capability_store.read_capability(capability_id=capability_id)
        return CapabilityDTO(
            id=capability.id,
            title=capability.title,
            description=capability.description,
            signals=capability.signals,
            tags=capability.tags,
        )

    @router.delete(
        "/{capability_id}",
        operation_id="delete_capability",
        status_code=status.HTTP_204_NO_CONTENT,
        responses={
            status.HTTP_204_NO_CONTENT: {
                "description": "Capability successfully deleted. No content returned."
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Capability not found. The specified `capability_id` does not exist"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="delete"),
    )
    async def delete_capability(
        request: Request,
        capability_id: CapabilityIdPath,
    ) -> None:
        """
        Deletes a capability from the system.

        Deleting a non-existent capability will return 404.
        No content will be returned from a successful deletion.
        """
        await authorization_policy.authorize(request, Operation.DELETE_CAPABILITY)

        await capability_store.delete_capability(capability_id=capability_id)

    return router
