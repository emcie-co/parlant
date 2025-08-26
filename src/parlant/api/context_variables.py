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

from pydantic import Field, field_validator
from datetime import datetime
from croniter import croniter
from fastapi import HTTPException, Path, Query, Request, status
from typing import Annotated, Optional, Sequence, TypeAlias, cast

from fastapi import APIRouter
from parlant.api import common
from parlant.api.authorization import AuthorizationPolicy, Operation
from parlant.api.common import (
    ToolIdDTO,
    JSONSerializableDTO,
    apigen_config,
    ExampleJson,
    apigen_skip_config,
)
from parlant.core.agents import AgentId, AgentStore
from parlant.core.common import DefaultBaseModel
from parlant.core.context_variables import (
    ContextVariableId,
    ContextVariableStore,
    ContextVariableUpdateParams,
    ContextVariableValueId,
)
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.tags import TagId, TagStore, Tag
from parlant.core.tools import ToolId

API_GROUP = "context-variables"


FreshnessRulesField: TypeAlias = Annotated[
    str,
    Field(default=None, description="Cron expression defining the freshness rules"),
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

legacy_context_variable_example = {
    "id": "v9a8r7i6b5",
    "name": "UserBalance",
    "description": "Stores the account balances of users",
    "tool_id": {
        "service_name": "finance_service",
        "tool_name": "balance_checker",
    },
    "freshness_rules": "*/5 * * * *",
}


class LegacyContextVariableDTO(
    DefaultBaseModel,
    json_schema_extra={"example": legacy_context_variable_example},
):
    """
    Represents a type of customer or tag data that the agent tracks.

    Context variables store information that helps the agent provide
    personalized responses based on each customer's or group's specific situation,
    such as their subscription tier, usage patterns, or preferences.
    """

    id: ContextVariableIdPath
    name: ContextVariableNameField
    description: Optional[ContextVariableDescriptionField] = None
    tool_id: Optional[ToolIdDTO] = None
    freshness_rules: Optional[FreshnessRulesField] = None


context_variable_creation_params_example = {
    "name": "UserBalance",
    "description": "Stores the account balances of users",
    "tool_id": {
        "service_name": "finance_service",
        "tool_name": "balance_checker",
    },
    "freshness_rules": "30 2 * * *",
}


class LegacyContextVariableCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": context_variable_creation_params_example},
):
    """Parameters for creating a new context variable."""

    name: ContextVariableNameField
    description: Optional[ContextVariableDescriptionField] = None
    tool_id: Optional[ToolIdDTO] = None
    freshness_rules: Optional[FreshnessRulesField] = None

    @field_validator("freshness_rules")
    @classmethod
    def validate_freshness_rules(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            try:
                croniter(value)
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="the provided freshness_rules. contain an invalid cron expression.",
                )
        return value


legacy_context_variable_update_params_example = {
    "name": "CustomerBalance",
    "description": "Stores the account balances of users",
    "freshness_rules": "0 8,20 * * *",
    "tool_id": {
        "service_name": "finance_service",
        "tool_name": "balance_checker",
    },
}


class LegacyContextVariableUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": legacy_context_variable_update_params_example},
):
    """Parameters for updating an existing context variable."""

    name: Optional[ContextVariableNameField] = None
    description: Optional[ContextVariableDescriptionField] = None
    tool_id: Optional[ToolIdDTO] = None
    freshness_rules: Optional[FreshnessRulesField] = None

    @field_validator("freshness_rules")
    @classmethod
    def validate_freshness_rules(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            try:
                croniter(value)
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="the provided freshness_rules. contain an invalid cron expression.",
                )
        return value


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

legacy_context_variable_read_result_example: ExampleJson = {
    "context_variable": {
        "id": "v9a8r7i6b5",
        "name": "UserBalance",
        "description": "Stores the account balances of users",
        "tool_id": {"service_name": "finance_service", "tool_name": "balance_checker"},
        "freshness_rules": "0 8,20 * * *",
    },
    "key_value_pairs": {
        "user_123": {
            "id": "val_789abc",
            "last_modified": "2024-03-24T12:00:00Z",
            "data": {
                "balance": 5000.50,
                "currency": "USD",
                "last_transaction": "2024-03-23T15:30:00Z",
                "status": "active",
            },
        },
        "user_456": {
            "id": "val_def123",
            "last_modified": "2024-03-24T14:30:00Z",
            "data": {
                "balance": 7500.75,
                "currency": "EUR",
                "last_transaction": "2024-03-24T14:15:00Z",
                "status": "active",
            },
        },
    },
}


class LegacyContextVariableReadResult(
    DefaultBaseModel,
    json_schema_extra={"example": legacy_context_variable_read_result_example},
):
    """Complete context variable data including its values."""

    context_variable: LegacyContextVariableDTO
    key_value_pairs: Optional[KeyValuePairsField] = None


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


def create_legacy_router(
    context_variable_store: ContextVariableStore,
    service_registry: ServiceRegistry,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "/{agent_id}/context-variables",
        status_code=status.HTTP_201_CREATED,
        operation_id="legacy_create_variable",
        response_model=LegacyContextVariableDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Context variable type successfully created",
                "content": common.example_json_content(legacy_context_variable_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Agent or tool not found"},
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_skip_config(),
        deprecated=True,
    )
    async def create_variable(
        agent_id: AgentIdPath,
        params: LegacyContextVariableCreationParamsDTO,
    ) -> LegacyContextVariableDTO:
        """
        [DEPRECATED] Creates a new context variable for tracking customer-specific or tag-specific data.

        This endpoint is deprecated, and will be removed in a future release.
        Please use the tag-based context variables API instead.

        Example uses:
        - Track subscription tiers to control feature access
        - Store usage patterns for personalized recommendations
        - Remember customer preferences for tailored responses
        """
        if params.tool_id:
            service = await service_registry.read_tool_service(params.tool_id.service_name)
            _ = await service.read_tool(params.tool_id.tool_name)

        variable = await context_variable_store.create_variable(
            name=params.name,
            description=params.description,
            tool_id=ToolId(params.tool_id.service_name, params.tool_id.tool_name)
            if params.tool_id
            else None,
            freshness_rules=params.freshness_rules,
        )

        await context_variable_store.add_variable_tag(variable.id, Tag.for_agent_id(agent_id))

        return LegacyContextVariableDTO(
            id=variable.id,
            name=variable.name,
            description=variable.description,
            tool_id=ToolIdDTO(
                service_name=variable.tool_id.service_name, tool_name=variable.tool_id.tool_name
            )
            if variable.tool_id
            else None,
            freshness_rules=variable.freshness_rules,
        )

    @router.patch(
        "/{agent_id}/context-variables/{variable_id}",
        operation_id="legacy_update_variable",
        response_model=LegacyContextVariableDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Context variable type successfully updated",
                "content": common.example_json_content(legacy_context_variable_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Variable or agent not found"},
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_skip_config(),
        deprecated=True,
    )
    async def update_variable(
        agent_id: AgentIdPath,
        variable_id: ContextVariableIdPath,
        params: LegacyContextVariableUpdateParamsDTO,
    ) -> LegacyContextVariableDTO:
        """
        [DEPRECATED] Updates an existing context variable.

        This endpoint is deprecated, and will be removed in a future release.
        Please use the tag-based context variables API instead.

        Only provided fields will be updated; others remain unchanged.
        """

        def from_dto(dto: LegacyContextVariableUpdateParamsDTO) -> ContextVariableUpdateParams:
            params: ContextVariableUpdateParams = {}

            if dto.name:
                params["name"] = dto.name

            if dto.description:
                params["description"] = dto.description

            if dto.tool_id:
                params["tool_id"] = ToolId(
                    service_name=dto.tool_id.service_name, tool_name=dto.tool_id.tool_name
                )

            if dto.freshness_rules:
                params["freshness_rules"] = dto.freshness_rules

            return params

        variables = await context_variable_store.list_variables(
            tags=[Tag.for_agent_id(agent_id)],
        )

        if variable_id not in [v.id for v in variables]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Variable not found for the provided agent",
            )

        variable = await context_variable_store.update_variable(
            id=variable_id,
            params=from_dto(params),
        )

        return LegacyContextVariableDTO(
            id=variable.id,
            name=variable.name,
            description=variable.description,
            tool_id=ToolIdDTO(
                service_name=variable.tool_id.service_name,
                tool_name=variable.tool_id.tool_name,
            )
            if variable.tool_id
            else None,
            freshness_rules=variable.freshness_rules,
        )

    @router.delete(
        "/{agent_id}/context-variables",
        status_code=status.HTTP_204_NO_CONTENT,
        operation_id="legacy_delete_variables",
        responses={
            status.HTTP_204_NO_CONTENT: {"description": "All context variables deleted"},
            status.HTTP_404_NOT_FOUND: {"description": "Agent not found"},
        },
        **apigen_skip_config(),
        deprecated=True,
    )
    async def delete_all_variables(
        agent_id: AgentIdPath,
    ) -> None:
        """
        [DEPRECATED] Deletes all context variables and their values for the provided agent ID.

        This endpoint is deprecated, and will be removed in a future release.
        Please use the tag-based context variables API instead.
        """
        variables = await context_variable_store.list_variables(
            tags=[Tag.for_agent_id(agent_id)],
        )

        for v in variables:
            variable = await context_variable_store.remove_variable_tag(
                variable_id=v.id,
                tag_id=Tag.for_agent_id(agent_id),
            )

            if not variable.tags:
                await context_variable_store.delete_variable(id=v.id)

    @router.delete(
        "/{agent_id}/context-variables/{variable_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        operation_id="legacy_delete_variable",
        responses={
            status.HTTP_204_NO_CONTENT: {
                "description": "Context variable and all its values deleted"
            },
            status.HTTP_404_NOT_FOUND: {"description": "Variable or agent not found"},
        },
        **apigen_skip_config(),
        deprecated=True,
    )
    async def delete_variable(
        agent_id: AgentIdPath,
        variable_id: ContextVariableIdPath,
    ) -> None:
        """
        [DEPRECATED] Deletes a specific context variable and all its values.

        This endpoint is deprecated, and will be removed in a future release.
        Please use the tag-based context variables API instead.
        """
        variables = await context_variable_store.list_variables(
            tags=[Tag.for_agent_id(agent_id)],
        )
        if variable_id not in [v.id for v in variables]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Variable not found for the provided agent",
            )

        variable = await context_variable_store.remove_variable_tag(
            variable_id=variable_id,
            tag_id=Tag.for_agent_id(agent_id),
        )

        if not variable.tags:
            await context_variable_store.delete_variable(id=variable_id)

    @router.get(
        "/{agent_id}/context-variables",
        operation_id="legacy_list_variables",
        response_model=Sequence[LegacyContextVariableDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "List of all context variable for the provided agent",
                "content": common.example_json_content([legacy_context_variable_example]),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Agent not found"},
        },
        **apigen_skip_config(),
        deprecated=True,
    )
    async def list_variables(
        agent_id: AgentIdPath,
    ) -> Sequence[LegacyContextVariableDTO]:
        """
        [DEPRECATED] Lists all context variables set for the provided agent.

        This endpoint is deprecated, and will be removed in a future release.
        Please use the tag-based context variables API instead.
        """
        variables = await context_variable_store.list_variables(
            tags=[Tag.for_agent_id(agent_id)],
        )

        return [
            LegacyContextVariableDTO(
                id=variable.id,
                name=variable.name,
                description=variable.description,
                tool_id=ToolIdDTO(
                    service_name=variable.tool_id.service_name,
                    tool_name=variable.tool_id.tool_name,
                )
                if variable.tool_id
                else None,
                freshness_rules=variable.freshness_rules,
            )
            for variable in variables
        ]

    @router.put(
        "/{agent_id}/context-variables/{variable_id}/{key}",
        operation_id="legacy_update_variable_value",
        response_model=ContextVariableValueDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Context value successfully updated for the customer or tag",
                "content": common.example_json_content(context_variable_value_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Variable, agent, or key not found"},
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_skip_config(),
        deprecated=True,
    )
    async def update_variable_value(
        agent_id: AgentIdPath,
        variable_id: ContextVariableIdPath,
        key: ContextVariableKeyPath,
        params: ContextVariableValueUpdateParamsDTO,
    ) -> ContextVariableValueDTO:
        """
        [DEPRECATED] Updates the value of a context variable.

        This endpoint is deprecated, and will be removed in a future release.
        Please use the tag-based context variables API instead.

        The `key` represents a customer identifier or a customer tag in the format `tag:{tag_id}`.
        If `key="DEFAULT"`, the update applies to all customers.
        The `params` parameter contains the actual context information being stored.
        """
        variables = await context_variable_store.list_variables(
            tags=[Tag.for_agent_id(agent_id)],
        )
        if variable_id not in [v.id for v in variables]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Variable not found for the provided agent",
            )

        variable_value = await context_variable_store.update_value(
            variable_id=variable_id,
            key=key,
            data=params.data,
        )

        return ContextVariableValueDTO(
            id=variable_value.id,
            last_modified=variable_value.last_modified,
            data=cast(JSONSerializableDTO, variable_value.data),
        )

    @router.get(
        "/{agent_id}/context-variables/{variable_id}/{key}",
        operation_id="legacy_read_variable_value",
        response_model=ContextVariableValueDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Retrieved context value for the customer or tag",
                "content": common.example_json_content(context_variable_value_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Variable, agent, or key not found"},
        },
        **apigen_skip_config(),
        deprecated=True,
    )
    async def read_variable_value(
        agent_id: AgentIdPath,
        variable_id: ContextVariableIdPath,
        key: ContextVariableKeyPath,
    ) -> ContextVariableValueDTO:
        """
        [DEPRECATED] Retrieves the value of a context variable for a specific customer or tag.

        This endpoint is deprecated, and will be removed in a future release.
        Please use the tag-based context variables API instead.

        The key should be a customer identifier or a customer tag in the format `tag:{tag_id}`.
        """
        variables = await context_variable_store.list_variables(
            tags=[Tag.for_agent_id(agent_id)],
        )
        if variable_id not in [v.id for v in variables]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Variable not found for the provided agent",
            )

        variable_value = await context_variable_store.read_value(
            key=key,
            variable_id=variable_id,
        )

        if variable_value is not None:
            return ContextVariableValueDTO(
                id=variable_value.id,
                last_modified=variable_value.last_modified,
                data=cast(JSONSerializableDTO, variable_value.data),
            )

        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)

    @router.get(
        "/{agent_id}/context-variables/{variable_id}",
        operation_id="legacy_read_variable",
        response_model=LegacyContextVariableReadResult,
        responses={
            status.HTTP_200_OK: {
                "description": "Context variable details with optional values",
                "content": common.example_json_content(legacy_context_variable_read_result_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Variable or agent not found"},
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_skip_config(),
        deprecated=True,
    )
    async def read_variable(
        agent_id: AgentIdPath,
        variable_id: ContextVariableIdPath,
        include_values: IncludeValuesQuery = True,
    ) -> LegacyContextVariableReadResult:
        """
        [DEPRECATED] Retrieves a context variable's details and optionally its values.

        This endpoint is deprecated, and will be removed in a future release.
        Please use the tag-based context variables API instead.

        Can return all customer or tag values for this variable type if include_values=True.
        """
        variables = await context_variable_store.list_variables(
            tags=[Tag.for_agent_id(agent_id)],
        )
        variable = next((v for v in variables if v.id == variable_id), None)

        if variable is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Variable not found for the provided agent",
            )

        variable_dto = LegacyContextVariableDTO(
            id=variable.id,
            name=variable.name,
            description=variable.description,
            tool_id=ToolIdDTO(
                service_name=variable.tool_id.service_name,
                tool_name=variable.tool_id.tool_name,
            )
            if variable.tool_id
            else None,
            freshness_rules=variable.freshness_rules,
        )

        if not include_values:
            return LegacyContextVariableReadResult(
                context_variable=variable_dto,
                key_value_pairs=None,
            )

        key_value_pairs = await context_variable_store.list_values(
            variable_id=variable_id,
        )

        return LegacyContextVariableReadResult(
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
        "/{agent_id}/context-variables/{variable_id}/{key}",
        status_code=status.HTTP_204_NO_CONTENT,
        operation_id="legacy_delete_value",
        responses={
            status.HTTP_204_NO_CONTENT: {
                "description": "Context value deleted for the customer or tag"
            },
            status.HTTP_404_NOT_FOUND: {"description": "Variable, agent, or key not found"},
        },
        **apigen_skip_config(),
        deprecated=True,
    )
    async def delete_value(
        agent_id: AgentIdPath,
        variable_id: ContextVariableIdPath,
        key: ContextVariableKeyPath,
    ) -> None:
        """
        [DEPRECATED] Deletes a specific customer's or tag's value for this context variable.

        This endpoint is deprecated, and will be removed in a future release.
        Please use the tag-based context variables API instead.

        The key should be a customer identifier or a customer tag in the format `tag:{tag_id}`.
        Removes only the value for the specified key while keeping the variable's configuration.
        """
        variables = await context_variable_store.list_variables(
            tags=[Tag.for_agent_id(agent_id)],
        )
        if variable_id not in [v.id for v in variables]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Variable not found for the provided agent",
            )

        if not await context_variable_store.read_value(
            variable_id=variable_id,
            key=key,
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Value not found for variable '{variable_id}' and key '{key}'",
            )

        await context_variable_store.delete_value(
            variable_id=variable_id,
            key=key,
        )

    return router


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
    description: Optional[ContextVariableDescriptionField] = None
    tool_id: Optional[ToolIdDTO] = None
    freshness_rules: Optional[FreshnessRulesField] = None
    tags: Optional[ContextVariableTagsField] = None


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

    add: Optional[ContextVariableTagsUpdateAddField] = None
    remove: Optional[ContextVariableTagsUpdateRemoveField] = None


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

    name: Optional[ContextVariableNameField] = None
    description: Optional[ContextVariableDescriptionField] = None
    tool_id: Optional[ToolIdDTO] = None
    freshness_rules: Optional[FreshnessRulesField] = None
    tags: Optional[ContextVariableTagsUpdateParamsDTO] = None

    @field_validator("freshness_rules")
    @classmethod
    def validate_freshness_rules(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            try:
                croniter(value)
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="the provided freshness_rules. contain an invalid cron expression.",
                )
        return value


TagIdQuery: TypeAlias = Annotated[
    Optional[TagId],
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
    key_value_pairs: Optional[KeyValuePairsField] = None


class ContextVariableCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": context_variable_creation_params_example},
):
    """Parameters for creating a new context variable."""

    name: ContextVariableNameField
    description: Optional[ContextVariableDescriptionField] = None
    tool_id: Optional[ToolIdDTO] = None
    freshness_rules: Optional[FreshnessRulesField] = None
    tags: Optional[ContextVariableTagsField] = None

    @field_validator("freshness_rules")
    @classmethod
    def validate_freshness_rules(cls, value: Optional[str]) -> Optional[str]:
        if value is not None:
            try:
                croniter(value)
            except Exception:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="the provided freshness_rules. contain an invalid cron expression.",
                )
        return value


def create_router(
    authorization_policy: AuthorizationPolicy,
    context_variable_store: ContextVariableStore,
    service_registry: ServiceRegistry,
    agent_store: AgentStore,
    tag_store: TagStore,
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
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
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

        if params.tool_id:
            service = await service_registry.read_tool_service(params.tool_id.service_name)
            _ = await service.read_tool(params.tool_id.tool_name)

        tags = []

        if params.tags:
            for tag_id in params.tags:
                if agent_id := Tag.extract_agent_id(tag_id):
                    _ = await agent_store.read_agent(agent_id=AgentId(agent_id))
                else:
                    _ = await tag_store.read_tag(tag_id=tag_id)

            tags = list(set(params.tags))

        variable = await context_variable_store.create_variable(
            name=params.name,
            description=params.description,
            tool_id=ToolId(params.tool_id.service_name, params.tool_id.tool_name)
            if params.tool_id
            else None,
            freshness_rules=params.freshness_rules,
            tags=tags or None,
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
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
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

        def from_dto(dto: ContextVariableUpdateParamsDTO) -> ContextVariableUpdateParams:
            params: ContextVariableUpdateParams = {}

            if dto.name:
                params["name"] = dto.name

            if dto.description:
                params["description"] = dto.description

            if dto.tool_id:
                params["tool_id"] = ToolId(
                    service_name=dto.tool_id.service_name, tool_name=dto.tool_id.tool_name
                )

            if dto.freshness_rules:
                params["freshness_rules"] = dto.freshness_rules

            return params

        if params.tags:
            if params.tags.add:
                for tag_id in params.tags.add:
                    if agent_id := Tag.extract_agent_id(tag_id):
                        _ = await agent_store.read_agent(agent_id=AgentId(agent_id))
                    else:
                        _ = await tag_store.read_tag(tag_id=tag_id)
                    await context_variable_store.add_variable_tag(variable_id, tag_id)

            if params.tags.remove:
                for tag_id in params.tags.remove:
                    await context_variable_store.remove_variable_tag(variable_id, tag_id)

        updated_variable = await context_variable_store.update_variable(
            id=variable_id,
            params=from_dto(params),
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

        if tag_id:
            variables = await context_variable_store.list_variables(
                tags=[tag_id],
            )
        else:
            variables = await context_variable_store.list_variables()

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

        variable = await context_variable_store.read_variable(id=variable_id)

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

        key_value_pairs = await context_variable_store.list_values(variable_id=variable_id)

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

        if tag_id:
            variables = await context_variable_store.list_variables(
                tags=[tag_id],
            )
            for v in variables:
                updated_variable = await context_variable_store.remove_variable_tag(
                    variable_id=v.id,
                    tag_id=tag_id,
                )
                if not updated_variable.tags:
                    await context_variable_store.delete_variable(id=v.id)

        else:
            variables = await context_variable_store.list_variables()
            for v in variables:
                await context_variable_store.delete_variable(id=v.id)

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

        await context_variable_store.delete_variable(id=variable_id)

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

        _ = await context_variable_store.read_variable(id=variable_id)

        value = await context_variable_store.read_value(variable_id=variable_id, key=key)

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
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
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

        _ = await context_variable_store.read_variable(id=variable_id)

        value = await context_variable_store.update_value(
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
        _ = await context_variable_store.read_variable(id=variable_id)

        if not context_variable_store.read_value(variable_id=variable_id, key=key):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Value not found for variable '{variable_id}' and key '{key}'",
            )

        await context_variable_store.delete_value(variable_id=variable_id, key=key)

    return router
