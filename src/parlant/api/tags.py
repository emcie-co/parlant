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

from typing import Annotated, TypeAlias
from fastapi import APIRouter, Path, Request, status

from parlant.api.authorization import AuthorizationPolicy, Operation
from parlant.api.common import TagDTO, TagNameField, apigen_config, ExampleJson, tag_example
from parlant.core.common import DefaultBaseModel
from parlant.core.tags import TagId, TagStore

API_GROUP = "tags"


tag_creation_params_example: ExampleJson = {"name": "premium-customer"}


class TagCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": tag_creation_params_example},
):
    """
    Parameters for creating a new tag.

    Only requires a name - the ID and creation timestamp are automatically generated.
    Names should be kebab-case and unique within the system.
    """

    name: TagNameField


tag_update_params_example: ExampleJson = {"name": "enterprise-customer"}


class TagUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": tag_update_params_example},
):
    """
    Parameters for updating an existing tag.

    Currently only supports updating the tag's name.
    The ID and creation timestamp cannot be modified.
    """

    name: TagNameField


TagIdPath: TypeAlias = Annotated[
    TagId,
    Path(
        description="Unique identifier for the tag to operate on",
        examples=["tag_123xyz"],
    ),
]

tag_list_example: ExampleJson = [
    tag_example,
    {
        "id": "tag_456abc",
        "name": "enterprise",
        "creation_utc": "2024-03-24T12:30:00Z",
    },
]


def create_router(
    authorization_policy: AuthorizationPolicy,
    tag_store: TagStore,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_tag",
        response_model=TagDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Tag successfully created. Returns the complete tag object with generated ID.",
                "content": {"application/json": {"example": tag_example}},
            },
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Invalid tag parameters. Ensure name follows required format."
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create"),
    )
    async def create_tag(
        request: Request,
        params: TagCreationParamsDTO,
    ) -> TagDTO:
        """
        Creates a new tag with the specified name.

        The tag ID is automatically generated and the creation timestamp is set to the current time.
        Tag names must be unique and follow the kebab-case format.
        """
        await authorization_policy.authorize(request=request, operation=Operation.CREATE_TAG)

        tag = await tag_store.create_tag(
            name=params.name,
        )

        return TagDTO(id=tag.id, creation_utc=tag.creation_utc, name=tag.name)

    @router.get(
        "/{tag_id}",
        operation_id="read_tag",
        response_model=TagDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Tag details successfully retrieved",
                "content": {"application/json": {"example": tag_example}},
            },
            status.HTTP_404_NOT_FOUND: {"description": "No tag found with the specified ID"},
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve"),
    )
    async def read_tag(
        request: Request,
        tag_id: TagIdPath,
    ) -> TagDTO:
        """
        Retrieves details of a specific tag by ID.

        Returns a 404 error if no tag exists with the specified ID.
        """
        await authorization_policy.authorize(request=request, operation=Operation.READ_TAG)

        tag = await tag_store.read_tag(tag_id=tag_id)

        return TagDTO(id=tag.id, creation_utc=tag.creation_utc, name=tag.name)

    @router.get(
        "",
        operation_id="list_tags",
        response_model=list[TagDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "List of all tags in the system",
                "content": {"application/json": {"example": tag_list_example}},
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="list"),
    )
    async def list_tags(request: Request) -> list[TagDTO]:
        """
        Lists all tags in the system.

        Returns an empty list if no tags exist.
        Tags are returned in no particular order.
        """
        await authorization_policy.authorize(request=request, operation=Operation.LIST_TAGS)

        tags = await tag_store.list_tags()

        return [TagDTO(id=tag.id, creation_utc=tag.creation_utc, name=tag.name) for tag in tags]

    @router.patch(
        "/{tag_id}",
        operation_id="update_tag",
        response_model=TagDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Tag successfully updated. Returns the updated tag.",
                "content": {"application/json": {"example": tag_example}},
            },
            status.HTTP_404_NOT_FOUND: {"description": "No tag found with the specified ID"},
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Invalid update parameters. Ensure name follows required format."
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="update"),
    )
    async def update_tag(
        request: Request,
        tag_id: TagIdPath,
        params: TagUpdateParamsDTO,
    ) -> TagDTO:
        """
        Updates an existing tag's name.

        Only the name can be modified,
        The tag's ID and creation timestamp cannot be modified.
        """
        await authorization_policy.authorize(request=request, operation=Operation.UPDATE_TAG)

        tag = await tag_store.update_tag(
            tag_id=tag_id,
            params={"name": params.name},
        )

        return TagDTO(id=tag.id, creation_utc=tag.creation_utc, name=tag.name)

    @router.delete(
        "/{tag_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        operation_id="delete_tag",
        responses={
            status.HTTP_204_NO_CONTENT: {"description": "Tag successfully deleted"},
            status.HTTP_404_NOT_FOUND: {"description": "No tag found with the specified ID"},
        },
        **apigen_config(group_name=API_GROUP, method_name="delete"),
    )
    async def delete_tag(
        request: Request,
        tag_id: TagId,
    ) -> None:
        """
        Permanently deletes a tag.

        This operation cannot be undone. Returns a 404 error if no tag exists with the specified ID.
        Note that deleting a tag does not affect resources that were previously tagged with it.
        """
        await authorization_policy.authorize(request=request, operation=Operation.DELETE_TAG)

        await tag_store.delete_tag(tag_id=tag_id)

    return router
