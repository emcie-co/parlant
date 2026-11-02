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

from typing import Annotated
from fastapi import APIRouter, Query, Request, status

from parlant.api.authorization import AuthorizationPolicy, Operation
from parlant.api.common import GroupDTO, apigen_config, ExampleJson, group_example
from parlant.core.application import Application

API_GROUP = "groups"


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

        Groups are defined in the agent's code; this read-only view lets
        backend integrations resolve group ids (for example, to assign a
        customer to a group by id). Returns an empty list if no groups exist
        or none match the filter. Groups are returned in no particular order.
        """
        await authorization_policy.authorize(request=request, operation=Operation.LIST_GROUPS)

        groups = await app.groups.find(name=name)

        return [
            GroupDTO(id=group.id, creation_utc=group.creation_utc, name=group.name)
            for group in groups
        ]

    return router
