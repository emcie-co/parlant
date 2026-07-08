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

from typing import Annotated, TypeAlias
from fastapi import APIRouter, Path, Query, Request, status

from parlant.api.authorization import AuthorizationPolicy, Operation
from parlant.api.common import GroupDTO, GroupNameField, apigen_config, ExampleJson, group_example
from parlant.core.application import Application
from parlant.core.common import DefaultBaseModel
from parlant.core.groups import GroupId

API_GROUP = "groups"


group_creation_params_example: ExampleJson = {"name": "premium-customer"}


class GroupCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": group_creation_params_example},
):
    """
    Parameters for creating a new group.

    Only requires a name - the ID and creation timestamp are automatically generated.
    Names should be kebab-case and unique within the system.
    """

    name: GroupNameField


group_update_params_example: ExampleJson = {"name": "enterprise-customer"}


class GroupUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": group_update_params_example},
):
    """
    Parameters for updating an existing group.

    Currently only supports updating the group's name.
    The ID and creation timestamp cannot be modified.
    """

    name: GroupNameField


GroupIdPath: TypeAlias = Annotated[
    GroupId,
    Path(
        description="Unique identifier for the group to operate on",
        examples=["group_123xyz"],
    ),
]

group_list_example: ExampleJson = [
    group_example,
    {
        "id": "group_456abc",
        "name": "enterprise",
        "creation_utc": "2024-03-24T12:30:00Z",
    },
]


def create_router(
    authorization_policy: AuthorizationPolicy,
    app: Application,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_group",
        response_model=GroupDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Group successfully created. Returns the complete group object with generated ID.",
                "content": {"application/json": {"example": group_example}},
            },
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Invalid group parameters. Ensure name follows required format."
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create"),
    )
    async def create_group(
        request: Request,
        params: GroupCreationParamsDTO,
    ) -> GroupDTO:
        """
        Creates a new group with the specified name.

        The group ID is automatically generated and the creation timestamp is set to the current time.
        Group names must be unique and follow the kebab-case format.
        """
        await authorization_policy.authorize(request=request, operation=Operation.CREATE_GROUP)

        group = await app.groups.create(
            name=params.name,
        )

        return GroupDTO(id=group.id, creation_utc=group.creation_utc, name=group.name)

    @router.get(
        "/{group_id}",
        operation_id="read_group",
        response_model=GroupDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Group details successfully retrieved",
                "content": {"application/json": {"example": group_example}},
            },
            status.HTTP_404_NOT_FOUND: {"description": "No group found with the specified ID"},
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve"),
    )
    async def read_group(
        request: Request,
        group_id: GroupIdPath,
    ) -> GroupDTO:
        """
        Retrieves details of a specific group by ID.

        Returns a 404 error if no group exists with the specified ID.
        """
        await authorization_policy.authorize(request=request, operation=Operation.READ_GROUP)

        group = await app.groups.read(group_id=group_id)

        return GroupDTO(id=group.id, creation_utc=group.creation_utc, name=group.name)

    @router.get(
        "",
        operation_id="list_groups",
        response_model=list[GroupDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "List of all groups in the system",
                "content": {"application/json": {"example": group_list_example}},
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="list"),
    )
    async def list_groups(
        request: Request,
        name: Annotated[
            str | None,
            Query(
                description="Filter groups by name",
                examples=["premium-customer"],
            ),
        ] = None,
    ) -> list[GroupDTO]:
        """
        Lists all groups in the system, optionally filtered by name.

        Returns an empty list if no groups exist or none match the filter.
        Groups are returned in no particular order.
        """
        await authorization_policy.authorize(request=request, operation=Operation.LIST_GROUPS)

        groups = await app.groups.find(name=name)

        return [
            GroupDTO(id=group.id, creation_utc=group.creation_utc, name=group.name)
            for group in groups
        ]

    @router.patch(
        "/{group_id}",
        operation_id="update_group",
        response_model=GroupDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Group successfully updated. Returns the updated group.",
                "content": {"application/json": {"example": group_example}},
            },
            status.HTTP_404_NOT_FOUND: {"description": "No group found with the specified ID"},
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Invalid update parameters. Ensure name follows required format."
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="update"),
    )
    async def update_group(
        request: Request,
        group_id: GroupIdPath,
        params: GroupUpdateParamsDTO,
    ) -> GroupDTO:
        """
        Updates an existing group's name.

        Only the name can be modified,
        The group's ID and creation timestamp cannot be modified.
        """
        await authorization_policy.authorize(request=request, operation=Operation.UPDATE_GROUP)

        group = await app.groups.update(
            group_id=group_id,
            params={"name": params.name},
        )

        return GroupDTO(id=group.id, creation_utc=group.creation_utc, name=group.name)

    @router.delete(
        "/{group_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        operation_id="delete_group",
        responses={
            status.HTTP_204_NO_CONTENT: {"description": "Group successfully deleted"},
            status.HTTP_404_NOT_FOUND: {"description": "No group found with the specified ID"},
        },
        **apigen_config(group_name=API_GROUP, method_name="delete"),
    )
    async def delete_group(
        request: Request,
        group_id: GroupId,
    ) -> None:
        """
        Permanently deletes a group.

        This operation cannot be undone. Returns a 404 error if no group exists with the specified ID.
        Note that deleting a group does not affect resources that were previously grouped with it.
        """
        await authorization_policy.authorize(request=request, operation=Operation.DELETE_GROUP)

        await app.groups.delete(group_id=group_id)

    return router
