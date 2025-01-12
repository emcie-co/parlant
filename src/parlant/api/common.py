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

from enum import Enum
from pydantic import Field
from typing import Annotated, Any, Mapping, Optional, Sequence, TypeAlias

from parlant.core.common import DefaultBaseModel, GuidelineId


def apigen_config(group_name: str, method_name: str) -> Mapping[str, Any]:
    return {
        "openapi_extra": {
            "x-fern-sdk-group-name": group_name,
            "x-fern-sdk-method-name": method_name,
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
    action: GuidelineActionField


class GuidelinePayloadOperationDTO(Enum):
    """
    The kind of operation that should be performed on the payload.
    """

    ADD = "add"
    UPDATE = "update"


class CoherenceCheckKindDTO(Enum):
    """
    The specific relationship between the contradicting guidelines.
    """

    CONTRADICTION_WITH_EXISTING_GUIDELINE = "contradiction_with_existing_guideline"
    CONTRADICTION_WITH_ANOTHER_EVALUATED_GUIDELINE = (
        "contradiction_with_another_evaluated_guideline"
    )


class ConnectionPropositionKindDTO(Enum):
    """
    The specific relationship between the connected guidelines.
    """

    CONNECTION_WITH_EXISTING_GUIDELINE = "connection_with_existing_guideline"
    CONNECTION_WITH_ANOTHER_EVALUATED_GUIDELINE = "connection_with_another_evaluated_guideline"


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


GuidelinePayloadCoherenceCheckField: TypeAlias = Annotated[
    bool,
    Field(
        description="Whether to check for contradictions with other Guidelines",
        examples=[True, False],
    ),
]

GuidelinePayloadConnectionPropositionField: TypeAlias = Annotated[
    bool,
    Field(
        description="Whether to propose logical connections with other Guidelines",
        examples=[True, False],
    ),
]

guideline_payload_example: ExampleJson = {
    "content": {
        "condition": "User asks about product pricing",
        "action": "Provide current price list and any active discounts",
    },
    "operation": "add",
    "updated_id": None,
    "coherence_check": True,
    "connection_proposition": True,
}


class GuidelinePayloadDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_payload_example},
):
    """Payload data for a Guideline operation"""

    content: GuidelineContentDTO
    operation: GuidelinePayloadOperationDTO
    updated_id: Optional[GuidelineIdField] = None
    coherence_check: GuidelinePayloadCoherenceCheckField
    connection_proposition: GuidelinePayloadConnectionPropositionField


payload_example: ExampleJson = {
    "kind": "guideline",
    "guideline": {
        "content": {
            "condition": "User asks about product pricing",
            "action": "Provide current price list and any active discounts",
        },
        "operation": "add",
        "updated_id": None,
        "coherence_check": True,
        "connection_proposition": True,
    },
}


class PayloadDTO(
    DefaultBaseModel,
    json_schema_extra={"example": payload_example},
):
    """
    A container for a guideline payload along with its kind

    Only `"guideline"` is available at this point.
    """

    kind: PayloadKindDTO
    guideline: Optional[GuidelinePayloadDTO] = None


CoherenceCheckIssueField: TypeAlias = Annotated[
    str,
    Field(
        description="Description of the contradiction or conflict between Guidelines",
        examples=[
            "The actions contradict each other: one suggests being formal while the other suggests being casual",
            "The conditions overlap but lead to opposing actions",
        ],
    ),
]

CoherenceCheckSeverityField: TypeAlias = Annotated[
    int,
    Field(
        description="Numerical rating of the contradiction's severity (1-10, where 10 is most severe)",
        examples=[5, 8],
        ge=1,
        le=10,
    ),
]


coherence_check_example: ExampleJson = {
    "kind": "contradiction_with_existing_guideline",
    "first": {"condition": "User is frustrated", "action": "Respond with technical details"},
    "second": {"condition": "User is frustrated", "action": "Focus on emotional support first"},
    "issue": "Conflicting approaches to handling user frustration",
    "severity": 7,
}


class CoherenceCheckDTO(
    DefaultBaseModel,
    json_schema_extra={"example": coherence_check_example},
):
    """Potential contradiction found between guidelines"""

    kind: CoherenceCheckKindDTO
    first: GuidelineContentDTO
    second: GuidelineContentDTO
    issue: CoherenceCheckIssueField
    severity: CoherenceCheckSeverityField


connection_proposition_example: ExampleJson = {
    "check_kind": "connection_with_existing_guideline",
    "source": {"condition": "User mentions technical problem", "action": "Request system logs"},
    "target": {
        "condition": "System logs are available",
        "action": "Analyze logs for error patterns",
    },
}


class ConnectionPropositionDTO(
    DefaultBaseModel,
    json_schema_extra={"example": connection_proposition_example},
):
    """Proposed logical connection between guidelines"""

    check_kind: ConnectionPropositionKindDTO
    source: GuidelineContentDTO
    target: GuidelineContentDTO


guideline_invoice_data_example: ExampleJson = {
    "coherence_checks": [coherence_check_example],
    "connection_propositions": [connection_proposition_example],
}


class GuidelineInvoiceDataDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_invoice_data_example},
):
    """Evaluation results for a Guideline, including contradiction checks and connection proposals"""

    coherence_checks: Sequence[CoherenceCheckDTO]
    connection_propositions: Optional[Sequence[ConnectionPropositionDTO]] = None


invoice_data_example: ExampleJson = {"guideline": guideline_invoice_data_example}


class InvoiceDataDTO(
    DefaultBaseModel,
    json_schema_extra={"example": invoice_data_example},
):
    """
    Contains the relevant invoice data.

    At this point only `guideline` is suppoerted.
    """

    guideline: Optional[GuidelineInvoiceDataDTO] = None


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
