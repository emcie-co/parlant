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

from pydantic import Field, field_validator
from datetime import datetime
from croniter import croniter
from fastapi import HTTPException, Path, Query, Request, status
from typing import Annotated, Sequence, TypeAlias, cast

from fastapi import APIRouter
from parlant.api import common
from parlant.api.authorization import AuthorizationPolicy, Operation
from parlant.api.common import (
    ToolIdDTO,
    JSONSerializableDTO,
    apigen_config,
    ExampleJson,
)
from parlant.core.app_modules.context_variables import (
    ContextVariableTagsUpdateParams,
)
from parlant.core.agents import AgentId
from parlant.core.application import Application
from parlant.core.common import DefaultBaseModel
from parlant.core.context_variables import (
    ContextVariableId,
    ContextVariableValueId,
)
from parlant.core.tags import TagId
from parlant.core.tools import ToolId

API_GROUP = "context-variables"


FreshnessRulesField: TypeAlias = Annotated[
    str,
    Field(
        description="Cron expression defining the freshness rules",
    ),
]

ContextVariableIdPath: TypeAlias = Annotated[
    ContextVariableId,
    Path(
        description="Unique identifier for the context variable",
        examples=["v9a8r7i6b5"],
    ),
]


ContextVariableNameField: TypeAlias = Annotated[
    str,
    Field(
        description="Name of the context variable",
        examples=["balance"],
        min_length=1,
    ),
]

ContextVariableDescriptionField: TypeAlias = Annotated[
    str,
    Field(
        description="Description of the context variable's purpose",
        examples=["Stores user preferences for customized interactions"],
    ),
]


context_variable_creation_params_example = {
    "name": "UserBalance",
    "description": "Stores the account balances of users",
    "tool_id": {
        "service_name": "finance_service",
        "tool_name": "balance_checker",
    },
    "freshness_rules": "30 2 * * *",
}


ValueIdField: TypeAlias = Annotated[
    ContextVariableValueId,
    Field(
        description="Unique identifier for the variable value",
        examples=["val_789abc"],
    ),
]

LastModifiedField: TypeAlias = Annotated[
    datetime,
    Field(
        description="Timestamp of the last modification",
    ),
]


DataField: TypeAlias = Annotated[
    JSONSerializableDTO,
    Field(
        description="The actual data stored in the variable",
    ),
]

context_variable_value_example: ExampleJson = {
    "id": "val_789abc",
    "last_modified": "2024-03-24T12:00:00Z",
    "data": {
        "balance": 5000.50,
        "currency": "USD",
        "last_transaction": "2024-03-23T15:30:00Z",
        "status": "active",
    },
}


class ContextVariableValueDTO(
    DefaultBaseModel,
    json_schema_extra={"example": context_variable_value_example},
):
    """
    Represents the actual stored value for a specific customer's or tag's context.

    This could be their subscription details, feature usage history,
    preferences, or any other customer or tag information that helps
    personalize the agent's responses.
    """

    id: ValueIdField
    last_modified: LastModifiedField
    data: DataField


context_variable_value_update_params_example: ExampleJson = {
    "data": {
        "balance": 5000.50,
        "currency": "USD",
        "last_transaction": "2024-03-23T15:30:00Z",
        "status": "active",
    }
}


class ContextVariableValueUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": context_variable_value_update_params_example},
):
    """Parameters for updating a context variable value."""

    data: DataField


KeyValuePairsField: TypeAlias = Annotated[
    dict[str, ContextVariableValueDTO],
    Field(
        description="Collection of key-value pairs associated with the variable",
    ),
]


AgentIdPath: TypeAlias = Annotated[
    AgentId,
    Path(
        description="Unique identifier of the agent",
        examples=["a1g2e3n4t5"],
    ),
]

ContextVariableKeyPath: TypeAlias = Annotated[
    str,
    Path(
        description="Key for the variable value",
        examples=["user_1", "tag_vip"],
        min_length=1,
    ),
]


IncludeValuesQuery: TypeAlias = Annotated[
    bool,
    Query(
        description="Whether to include variable values in the response",
        examples=[True, False],
    ),
]


ContextVariableTagsField: TypeAlias = Annotated[
    list[TagId],
    Field(
        description="List of tags associated with the context variable",
    ),
]

context_variable_example: ExampleJson = {
    "id": "v9a8r7i6b5",
    "name": "UserBalance",
    "description": "Stores the account balances of users",
    "tool_id": {"service_name": "finance_service", "tool_name": "balance_checker"},
    "freshness_rules": "0 8,20 * * *",
    "tags": ["tag:123", "tag:456"],
}


class ContextVariableDTO(
    DefaultBaseModel,
    json_schema_extra={"example": context_variable_example},
):
    """
    Represents a context variable type.
    """

    id: ContextVariableIdPath
    name: ContextVariableNameField
    description: ContextVariableDescriptionField | None = None
    tool_id: ToolIdDTO | None = None
    freshness_rules: FreshnessRulesField | None = None
    tags: ContextVariableTagsField | None = None


context_variable_tags_update_params_example: ExampleJson = {
    "add": [
        "t9a8g703f4",
        "tag_456abc",
    ],
    "remove": [
        "tag_789def",
        "tag_012ghi",
    ],
}


ContextVariableTagsUpdateAddField: TypeAlias = Annotated[
    list[TagId],
    Field(
        description="List of tag IDs to add to the context variable",
        examples=[["tag1", "tag2"]],
    ),
]

ContextVariableTagsUpdateRemoveField: TypeAlias = Annotated[
    list[TagId],
    Field(
        description="List of tag IDs to remove from the context variable",
        examples=[["tag1", "tag2"]],
    ),
]


class ContextVariableTagsUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": context_variable_tags_update_params_example},
):
    """
    Parameters for updating the tags of an existing context variable.
    """

    add: ContextVariableTagsUpdateAddField | None = None
    remove: ContextVariableTagsUpdateRemoveField | None = None


context_variable_update_params_example: ExampleJson = {
    "name": "UserBalance",
    "description": "Stores the account balances of users",
    "tool_id": {"service_name": "finance_service", "tool_name": "balance_checker"},
    "freshness_rules": "0 8,20 * * *",
    "tags": {
        "add": ["tag:123", "tag:456"],
        "remove": ["tag:789", "tag:012"],
    },
}


class ContextVariableUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": context_variable_update_params_example},
):
    """Parameters for updating an existing context variable."""

    name: ContextVariableNameField | None = None
    description: ContextVariableDescriptionField | None = None
    tool_id: ToolIdDTO | None = None
    freshness_rules: FreshnessRulesField | None = None
    tags: ContextVariableTagsUpdateParamsDTO | None = None

    @field_validator("freshness_rules")
    @classmethod
    def validate_freshness_rules(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                croniter(value)
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="the provided freshness_rules. contain an invalid cron expression.",
                )
        return value


TagIdQuery: TypeAlias = Annotated[
    TagId | None,
    Query(
        description="The tag ID to filter context variables by",
        examples=["tag:123"],
    ),
]


class ContextVariableReadResult(
    DefaultBaseModel,
    json_schema_extra={"example": context_variable_example},
):
    """Complete context variable data including its values."""

    context_variable: ContextVariableDTO
    key_value_pairs: KeyValuePairsField | None = None


class ContextVariableCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": context_variable_creation_params_example},
):
    """Parameters for creating a new context variable."""

    name: ContextVariableNameField
    description: ContextVariableDescriptionField | None = None
    tool_id: ToolIdDTO | None = None
    freshness_rules: FreshnessRulesField | None = None
    tags: ContextVariableTagsField | None = None

    @field_validator("freshness_rules")
    @classmethod
    def validate_freshness_rules(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                croniter(value)
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail="the provided freshness_rules. contain an invalid cron expression.",
                )
        return value


def create_router(
    authorization_policy: AuthorizationPolicy,
    app: Application,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_variable",
        response_model=ContextVariableDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Context variable type successfully created",
                "content": common.example_json_content(context_variable_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Tool not found"},
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create"),
    )
    async def create_variable(
        request: Request,
        params: ContextVariableCreationParamsDTO,
    ) -> ContextVariableDTO:
        """
        Creates a new context variable

        Example uses:
        - Track subscription tiers to control feature access
        - Store usage patterns for personalized recommendations
        - Remember preferences for tailored responses
        """
        await authorization_policy.authorize(
            request=request,
            operation=Operation.CREATE_CONTEXT_VARIABLE,
        )

        variable = await app.variables.create(
            name=params.name,
            description=params.description,
            tool_id=ToolId(params.tool_id.service_name, params.tool_id.tool_name)
            if params.tool_id
            else None,
            freshness_rules=params.freshness_rules,
            tags=params.tags,
        )

        return ContextVariableDTO(
            id=variable.id,
            name=variable.name,
            description=variable.description,
            tool_id=ToolIdDTO(
                service_name=variable.tool_id.service_name, tool_name=variable.tool_id.tool_name
            )
            if variable.tool_id
            else None,
            freshness_rules=variable.freshness_rules,
            tags=variable.tags,
        )

    @router.patch(
        "/{variable_id}",
        operation_id="update_variable",
        response_model=ContextVariableDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Context variable type successfully updated",
                "content": common.example_json_content(context_variable_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Variable not found"},
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="update"),
    )
    async def update_variable(
        request: Request,
        variable_id: ContextVariableIdPath,
        params: ContextVariableUpdateParamsDTO,
    ) -> ContextVariableDTO:
        """
        Updates an existing context variable.

        Only provided fields will be updated; others remain unchanged.
        """
        await authorization_policy.authorize(
            request=request,
            operation=Operation.UPDATE_CONTEXT_VARIABLE,
        )

        updated_variable = await app.variables.update(
            variable_id=variable_id,
            name=params.name,
            description=params.description,
            tool_id=ToolId(params.tool_id.service_name, params.tool_id.tool_name)
            if params.tool_id
            else None,
            freshness_rules=params.freshness_rules,
            tags=ContextVariableTagsUpdateParams(
                add=params.tags.add,
                remove=params.tags.remove,
            )
            if params.tags
            else None,
        )

        return ContextVariableDTO(
            id=updated_variable.id,
            name=updated_variable.name,
            description=updated_variable.description,
            tool_id=ToolIdDTO(
                service_name=updated_variable.tool_id.service_name,
                tool_name=updated_variable.tool_id.tool_name,
            )
            if updated_variable.tool_id
            else None,
            freshness_rules=updated_variable.freshness_rules,
            tags=updated_variable.tags,
        )

    @router.get(
        "",
        operation_id="list_variables",
        response_model=Sequence[ContextVariableDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "List of all context variables",
                "content": common.example_json_content([context_variable_example]),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Agent not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="list"),
    )
    async def list_variables(
        request: Request,
        tag_id: TagIdQuery = None,
    ) -> Sequence[ContextVariableDTO]:
        """Lists all context variables set for the provided tag or all context variables if no tag is provided"""
        await authorization_policy.authorize(request, Operation.LIST_CONTEXT_VARIABLES)

        variables = await app.variables.find(tag_id=tag_id)

        return [
            ContextVariableDTO(
                id=v.id,
                name=v.name,
                description=v.description,
                tool_id=ToolIdDTO(
                    service_name=v.tool_id.service_name, tool_name=v.tool_id.tool_name
                )
                if v.tool_id
                else None,
                freshness_rules=v.freshness_rules,
                tags=v.tags,
            )
            for v in variables
        ]

    @router.get(
        "/{variable_id}",
        operation_id="read_variable",
        response_model=ContextVariableReadResult,
        responses={
            status.HTTP_200_OK: {
                "description": "Context variable details successfully retrieved",
                "content": common.example_json_content(context_variable_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Variable not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve"),
    )
    async def read_variable(
        request: Request,
        variable_id: ContextVariableIdPath,
        include_values: IncludeValuesQuery = True,
    ) -> ContextVariableReadResult:
        """
        Retrieves a context variable's details and optionally its values.

        Can return all customer or tag values for this variable type if include_values=True.
        """
        await authorization_policy.authorize(
            request=request,
            operation=Operation.READ_CONTEXT_VARIABLE,
        )

        variable = await app.variables.read(variable_id=variable_id)

        variable_dto = ContextVariableDTO(
            id=variable.id,
            name=variable.name,
            description=variable.description,
            tool_id=ToolIdDTO(
                service_name=variable.tool_id.service_name, tool_name=variable.tool_id.tool_name
            )
            if variable.tool_id
            else None,
            freshness_rules=variable.freshness_rules,
            tags=variable.tags,
        )

        if not include_values:
            return ContextVariableReadResult(
                context_variable=variable_dto,
                key_value_pairs=None,
            )

        key_value_pairs = await app.variables.find_values(variable_id=variable_id)

        return ContextVariableReadResult(
            context_variable=variable_dto,
            key_value_pairs={
                key: ContextVariableValueDTO(
                    id=value.id,
                    last_modified=value.last_modified,
                    data=cast(JSONSerializableDTO, value.data),
                )
                for key, value in key_value_pairs
            },
        )

    @router.delete(
        "",
        status_code=status.HTTP_204_NO_CONTENT,
        operation_id="delete_variables",
        responses={
            status.HTTP_204_NO_CONTENT: {"description": "All context variables deleted"},
            status.HTTP_404_NOT_FOUND: {"description": "Tag not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="delete_many"),
    )
    async def delete_variables(
        request: Request,
        tag_id: TagIdQuery = None,
    ) -> None:
        """Deletes all context variables for the provided tag"""
        await authorization_policy.authorize(
            request=request,
            operation=Operation.DELETE_CONTEXT_VARIABLES,
        )

        await app.variables.delete_many(tag_id)

    @router.delete(
        "/{variable_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        operation_id="delete_variable",
        responses={
            status.HTTP_204_NO_CONTENT: {"description": "Context variable deleted"},
            status.HTTP_404_NOT_FOUND: {"description": "Variable not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="delete"),
    )
    async def delete_variable(
        request: Request,
        variable_id: ContextVariableIdPath,
    ) -> None:
        """Deletes a context variable"""
        await authorization_policy.authorize(
            request=request,
            operation=Operation.DELETE_CONTEXT_VARIABLE,
        )

        await app.variables.delete(variable_id=variable_id)

    @router.get(
        "/{variable_id}/{key}",
        operation_id="read_variable_value",
        response_model=ContextVariableValueDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Retrieved context value for the customer or tag",
                "content": common.example_json_content(context_variable_value_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Variable, agent, or key not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="get_value"),
    )
    async def read_variable_value(
        request: Request,
        variable_id: ContextVariableIdPath,
        key: ContextVariableKeyPath,
    ) -> ContextVariableValueDTO:
        """Retrieves a customer or tag value for the provided context variable"""
        await authorization_policy.authorize(
            request=request,
            operation=Operation.READ_CONTEXT_VARIABLE_VALUE,
        )

        value = await app.variables.read_value(variable_id=variable_id, key=key)

        if value:
            return ContextVariableValueDTO(
                id=value.id,
                last_modified=value.last_modified,
                data=cast(JSONSerializableDTO, value.data),
            )

        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    @router.put(
        "/{variable_id}/{key}",
        operation_id="update_variable_value",
        response_model=ContextVariableValueDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Context value successfully updated for the customer or tag",
                "content": common.example_json_content(context_variable_value_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Variable, agent, or key not found"},
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="set_value"),
    )
    async def update_variable_value(
        request: Request,
        variable_id: ContextVariableIdPath,
        key: ContextVariableKeyPath,
        params: ContextVariableValueUpdateParamsDTO,
    ) -> ContextVariableValueDTO:
        """Updates a customer or tag value for the provided context variable"""
        await authorization_policy.authorize(
            request=request,
            operation=Operation.UPDATE_CONTEXT_VARIABLE_VALUE,
        )

        value = await app.variables.update_value(
            variable_id=variable_id,
            key=key,
            data=params.data,
        )

        return ContextVariableValueDTO(
            id=value.id,
            last_modified=value.last_modified,
            data=cast(JSONSerializableDTO, value.data),
        )

    @router.delete(
        "/{variable_id}/{key}",
        status_code=status.HTTP_204_NO_CONTENT,
        operation_id="delete_value",
        responses={
            status.HTTP_204_NO_CONTENT: {
                "description": "Context value deleted for the customer or tag"
            },
            status.HTTP_404_NOT_FOUND: {"description": "Variable, agent, or key not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="delete_value"),
    )
    async def delete_value(
        request: Request,
        variable_id: ContextVariableIdPath,
        key: ContextVariableKeyPath,
    ) -> None:
        """Deletes a customer or tag value for the provided context variable"""
        await authorization_policy.authorize(
            request=request,
            operation=Operation.DELETE_CONTEXT_VARIABLE_VALUE,
        )

        if not await app.variables.read_value(variable_id=variable_id, key=key):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Value not found for variable '{variable_id}' and key '{key}'",
            )

        await app.variables.delete_value(variable_id=variable_id, key=key)

    return router
