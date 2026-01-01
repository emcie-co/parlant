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

from datetime import datetime
from enum import Enum
from pydantic import Field
from typing import Annotated, Any, Mapping, Sequence, TypeAlias

from parlant.core.agents import CompositionMode
from parlant.core.common import DefaultBaseModel
from parlant.core.evaluations import PayloadOperation
from parlant.core.persistence.common import SortDirection
from parlant.core.relationships import RelationshipId
from parlant.core.guidelines import GuidelineId
from parlant.core.tags import TagId
from parlant.core.tools import Tool, ToolParameterDescriptor


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


def composition_mode_dto_to_composition_mode(dto: CompositionModeDTO) -> CompositionMode:
    """Convert CompositionModeDTO to core CompositionMode."""
    match dto:
        case CompositionModeDTO.FLUID:
            return CompositionMode.FLUID
        case CompositionModeDTO.CANNED_STRICT:
            return CompositionMode.CANNED_STRICT
        case CompositionModeDTO.CANNED_COMPOSITED:
            return CompositionMode.CANNED_COMPOSITED
        case CompositionModeDTO.CANNED_FLUID:
            return CompositionMode.CANNED_FLUID


def composition_mode_to_composition_mode_dto(
    composition_mode: CompositionMode,
) -> CompositionModeDTO:
    """Convert core CompositionMode to CompositionModeDTO."""
    match composition_mode:
        case CompositionMode.FLUID:
            return CompositionModeDTO.FLUID
        case CompositionMode.CANNED_STRICT:
            return CompositionModeDTO.CANNED_STRICT
        case CompositionMode.CANNED_COMPOSITED:
            return CompositionModeDTO.CANNED_COMPOSITED
        case CompositionMode.CANNED_FLUID:
            return CompositionModeDTO.CANNED_FLUID


def apigen_config(group_name: str, method_name: str) -> Mapping[str, Any]:
    return {
        "openapi_extra": {
            "x-fern-sdk-group-name": group_name,
            "x-fern-sdk-method-name": method_name,
        }
    }


def apigen_skip_config() -> Mapping[str, Any]:
    return {
        "openapi_extra": {
            "x-fern-ignore": True,
        }
    }


ExampleJson: TypeAlias = dict[str, Any] | list[Any]
ExtraSchema: TypeAlias = dict[str, dict[str, Any]]


JSONSerializableDTO: TypeAlias = Annotated[
    Any,
    Field(
        description="Any valid json",
        examples=['"hello"', "[1, 2, 3]", '{"data"="something", "data2"="something2"}'],
    ),
]


class EvaluationStatusDTO(Enum):
    """
    Current state of an evaluation task
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


GuidelineConditionField: TypeAlias = Annotated[
    str,
    Field(
        description="If this condition is satisfied, the action will be performed",
        examples=["The user is angry."],
    ),
]

GuidelineActionField: TypeAlias = Annotated[
    str,
    Field(
        description="This action will be performed if the condition is satisfied",
        examples=["Sing the user a lullaby."],
    ),
]

GuidelineDescriptionField: TypeAlias = Annotated[
    str,
    Field(
        description="Optional description providing additional context for the guideline",
        examples=["This applies only to premium customers with active subscriptions."],
    ),
]


class CriticalityDTO(Enum):
    """
    The criticality level of a guideline.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


GuidelineCriticalityField: TypeAlias = Annotated[
    CriticalityDTO,
    Field(
        description="The criticality level of the guideline",
        examples=["high"],
    ),
]

guideline_content_example: ExampleJson = {
    "condition": "User asks about product pricing",
    "action": "Provide current price list and any active discounts",
}


class GuidelineContentDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_content_example},
):
    """
    Represention of a guideline with a condition-action pair.

    This model defines a structure for guidelines where specific actions should be taken
    when certain conditions are met. It follows a simple "if condition then action" pattern.
    """

    condition: GuidelineConditionField
    action: GuidelineActionField | None = None


class GuidelinePayloadOperationDTO(Enum):
    """
    The kind of operation that should be performed on the payload.
    """

    ADD = "add"
    UPDATE = "update"


class PayloadKindDTO(Enum):
    """
    The kind of payload.

    At this point only `"guideline"` is supported.
    """

    GUIDELINE = "guideline"


GuidelineIdField: TypeAlias = Annotated[
    GuidelineId,
    Field(
        description="Unique identifier for the guideline",
        examples=["IUCGT-l4pS"],
    ),
]


def operation_dto_to_operation(dto: GuidelinePayloadOperationDTO) -> PayloadOperation:
    if operation := {
        GuidelinePayloadOperationDTO.ADD: PayloadOperation.ADD,
        GuidelinePayloadOperationDTO.UPDATE: PayloadOperation.UPDATE,
    }.get(dto):
        return operation

    raise ValueError(f"Unsupported operation: {dto}")


ServiceNameField: TypeAlias = Annotated[
    str,
    Field(
        description="Name of the service",
        examples=["email_service", "payment_processor"],
    ),
]

ToolNameField: TypeAlias = Annotated[
    str,
    Field(
        description="Name of the tool",
        examples=["send_email", "process_payment"],
    ),
]


tool_id_example: ExampleJson = {"service_name": "email_service", "tool_name": "send_email"}


class ToolIdDTO(
    DefaultBaseModel,
    json_schema_extra={"example": tool_id_example},
):
    """Tool identifier associated with this variable"""

    service_name: ServiceNameField
    tool_name: ToolNameField


def example_json_content(json_example: ExampleJson) -> ExtraSchema:
    return {"application/json": {"example": json_example}}


GuidelineMetadataField: TypeAlias = Annotated[
    Mapping[str, JSONSerializableDTO],
    Field(description="Metadata for the guideline"),
]

GuidelineEnabledField: TypeAlias = Annotated[
    bool,
    Field(
        description="Whether the guideline is enabled",
        examples=[True, False],
    ),
]


guideline_dto_example = {
    "id": "guid_123xz",
    "condition": "when the customer asks about pricing",
    "action": "provide current pricing information and mention any ongoing promotions",
    "enabled": True,
    "tags": ["tag1", "tag2"],
    "metadata": {"key1": "value1", "key2": "value2"},
    "composition_mode": None,
}

GuidelineTagsField: TypeAlias = Annotated[
    Sequence[TagId],
    Field(
        description="The tags associated with the guideline",
        examples=[["tag1", "tag2"], []],
    ),
]


class GuidelineDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_dto_example},
):
    """Represents a guideline."""

    id: GuidelineIdField
    condition: GuidelineConditionField
    action: GuidelineActionField | None = None
    description: GuidelineDescriptionField | None = None
    criticality: GuidelineCriticalityField = CriticalityDTO.MEDIUM
    enabled: GuidelineEnabledField = True
    tags: GuidelineTagsField
    metadata: GuidelineMetadataField
    composition_mode: CompositionModeDTO | None = None


EnumValueTypeDTO: TypeAlias = str | int

ToolParameterDescriptionField: TypeAlias = Annotated[
    str,
    Field(
        description="Detailed description of what the parameter does and how it should be used",
        examples=["Email address of the recipient", "Maximum number of retries allowed"],
    ),
]

ToolParameterEnumField: TypeAlias = Annotated[
    Sequence[EnumValueTypeDTO],
    Field(
        description="List of allowed values for string or integer parameters. If provided, the parameter value must be one of these options.",
        examples=[["high", "medium", "low"], [1, 2, 3, 5, 8, 13]],
    ),
]


class ToolParameterTypeDTO(Enum):
    """
    The supported data types for tool parameters.

    Each type corresponds to a specific JSON Schema type and validation rules.
    """

    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    ARRAY = "array"


tool_parameter_example: ExampleJson = {
    "type": "string",
    "description": "Priority level for the email",
    "enum": ["high", "medium", "low"],
}


class ToolParameterDTO(
    DefaultBaseModel,
    json_schema_extra={"example": tool_parameter_example},
):
    """
    Defines a parameter that can be passed to a tool.

    Parameters can have different types with optional constraints like enums.
    Each parameter can include a description to help users understand its purpose.
    """

    type: ToolParameterTypeDTO
    description: ToolParameterDescriptionField | None = None
    enum: ToolParameterEnumField | None = None


ToolCreationUTCField: TypeAlias = Annotated[
    datetime,
    Field(
        description="UTC timestamp when the tool was first registered with the system",
        examples=["2024-03-24T12:00:00Z"],
    ),
]

ToolDescriptionField: TypeAlias = Annotated[
    str,
    Field(
        description="Detailed description of the tool's purpose and behavior",
        examples=[
            "Sends an email to specified recipients with optional attachments",
            "Processes a payment transaction and returns confirmation details",
        ],
    ),
]

ToolParametersField: TypeAlias = Annotated[
    dict[str, ToolParameterDTO],
    Field(
        description="Dictionary mapping parameter names to their definitions",
        examples=[
            {
                "recipient": {"type": "string", "description": "Email address to send to"},
                "amount": {"type": "number", "description": "Payment amount in dollars"},
            }
        ],
    ),
]

ToolRequiredField: TypeAlias = Annotated[
    Sequence[str],
    Field(
        description="List of parameter names that must be provided when calling the tool",
        examples=[["recipient", "subject"], ["payment_id", "amount"]],
    ),
]


tool_example: ExampleJson = {
    "creation_utc": "2024-03-24T12:00:00Z",
    "name": "send_email",
    "description": "Sends an email to specified recipients with configurable priority",
    "parameters": {
        "to": {"type": "string", "description": "Recipient email address"},
        "subject": {"type": "string", "description": "Email subject line"},
        "body": {"type": "string", "description": "Email body content"},
        "priority": {
            "type": "string",
            "description": "Priority level for the email",
            "enum": ["high", "medium", "low"],
        },
    },
    "required": ["to", "subject", "body"],
}


class ToolDTO(
    DefaultBaseModel,
    json_schema_extra={"example": tool_example},
):
    """
    Represents a single function provided by an integrated service.

    Tools are the primary way for agents to interact with external services.
    Each tool has defined parameters and can be invoked when those parameters
    are satisfied.
    """

    creation_utc: ToolCreationUTCField
    name: ToolNameField
    description: ToolDescriptionField
    parameters: ToolParametersField
    required: ToolRequiredField


def tool_parameters_to_dto(parameters: ToolParameterDescriptor) -> ToolParameterDTO:
    return ToolParameterDTO(
        type=ToolParameterTypeDTO(parameters["type"]),
        description=parameters["description"] if "description" in parameters else None,
        enum=parameters["enum"] if "enum" in parameters else None,
    )


def tool_to_dto(tool: Tool) -> ToolDTO:
    return ToolDTO(
        creation_utc=tool.creation_utc,
        name=tool.name,
        description=tool.description,
        parameters={
            name: tool_parameters_to_dto(descriptor)
            for name, (descriptor, _) in tool.parameters.items()
        },
        required=tool.required,
    )


TagIdField: TypeAlias = Annotated[
    TagId,
    Field(
        description="Unique identifier for the tag",
        examples=["tag_123xyz", "tag_premium42"],
    ),
]


TagNameField: TypeAlias = Annotated[
    str,
    Field(
        description="Human-readable name for the tag, used for display and organization",
        examples=["premium", "enterprise", "beta-tester"],
        min_length=1,
        max_length=50,
    ),
]

tag_example: ExampleJson = {
    "id": "tag_123xyz",
    "name": "premium",
    "creation_utc": "2024-03-24T12:00:00Z",
}


class TagDTO(
    DefaultBaseModel,
    json_schema_extra={"example": tag_example},
):
    """
    Represents a tag in the system.

    Tags can be used to categorize and label various resources like customers, sessions,
    or content. They provide a flexible way to organize and filter data.
    """

    id: TagIdField
    name: TagNameField


relationship_tag_dto_example: ExampleJson = {
    "id": "tid_123xz",
    "name": "tag1",
}


RelationshipIdField: TypeAlias = Annotated[
    RelationshipId,
    Field(
        description="Unique identifier for the relationship",
    ),
]


relationship_example: ExampleJson = {
    "id": "123",
    "source_guideline": {
        "id": "456",
        "condition": "when the customer asks about pricing",
        "action": "provide current pricing information",
        "enabled": True,
        "tags": ["tag1", "tag2"],
    },
    "target_tag": {
        "id": "789",
        "name": "tag1",
    },
    "indirect": False,
    "kind": "entailment",
}


class RelationshipKindDTO(Enum):
    """The kind of relationship."""

    ENTAILMENT = "entailment"
    PRIORITY = "priority"
    DEPENDENCY = "dependency"
    DISAMBIGUATION = "disambiguation"
    OVERLAP = "overlap"
    REEVALUATION = "reevaluation"


class SortDirectionDTO(Enum):
    """The direction to sort results."""

    ASC = "asc"
    DESC = "desc"


def sort_direction_dto_to_sort_direction(
    dto: SortDirectionDTO,
) -> SortDirection:
    match dto:
        case SortDirectionDTO.ASC:
            return SortDirection.ASC
        case SortDirectionDTO.DESC:
            return SortDirection.DESC
        case _:
            raise ValueError(f"Unsupported sort direction: {dto}")


class RelationshipDTO(
    DefaultBaseModel,
    json_schema_extra={"example": relationship_example},
):
    """Represents a relationship.

    Only one of `source_guideline` and `source_tag` can have a value.
    Only one of `target_guideline` and `target_tag` can have a value.
    Only one of `source_tool` and `target_tool` can have a value.
    """

    id: RelationshipIdField
    source_guideline: GuidelineDTO | None = None
    source_tag: TagDTO | None = None
    target_guideline: GuidelineDTO | None = None
    target_tag: TagDTO | None = None
    source_tool: ToolDTO | None = None
    target_tool: ToolDTO | None = None
    kind: RelationshipKindDTO
