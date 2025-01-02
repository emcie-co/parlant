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

from parlant.core.common import DefaultBaseModel
from parlant.core.guidelines import GuidelineId
from parlant.core.sessions import EventSource
from parlant.core.style_guides import (
    StyleGuideContent,
    StyleGuideEvent,
    StyleGuideExample,
    StyleGuideId,
)


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


class EventSourceDTO(Enum):
    """
    Source of an event in the session.

    Identifies who or what generated the event.
    """

    CUSTOMER = "customer"
    CUSTOMER_UI = "customer_ui"
    HUMAN_AGENT = "human_agent"
    HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT = "human_agent_on_behalf_of_ai_agent"
    AI_AGENT = "ai_agent"
    SYSTEM = "system"


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


class GuidelineCoherenceCheckKindDTO(Enum):
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


GuidelineCoherenceCheckIssueField: TypeAlias = Annotated[
    str,
    Field(
        description="Description of the contradiction or conflict between Guidelines",
        examples=[
            "The actions contradict each other: one suggests being formal while the other suggests being casual",
            "The conditions overlap but lead to opposing actions",
        ],
    ),
]

GuidelineCoherenceCheckSeverityField: TypeAlias = Annotated[
    int,
    Field(
        description="Numerical rating of the contradiction's severity (1-10, where 10 is most severe)",
        examples=[5, 8],
        ge=1,
        le=10,
    ),
]

guidelines_coherence_check_example: ExampleJson = {
    "kind": "contradiction_with_existing_guideline",
    "first": {"condition": "User is frustrated", "action": "Respond with technical details"},
    "second": {"condition": "User is frustrated", "action": "Focus on emotional support first"},
    "issue": "Conflicting approaches to handling user frustration",
    "severity": 7,
}


class GuidelinesCoherenceCheckDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guidelines_coherence_check_example},
):
    """Potential contradiction found between guidelines"""

    kind: GuidelineCoherenceCheckKindDTO
    first: GuidelineContentDTO
    second: GuidelineContentDTO
    issue: GuidelineCoherenceCheckIssueField
    severity: GuidelineCoherenceCheckSeverityField


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
    "coherence_checks": [guidelines_coherence_check_example],
    "connection_propositions": [connection_proposition_example],
}

StyleGuideCoherenceCheckIssueField: TypeAlias = Annotated[
    str,
    Field(
        description="Description of the contradiction or conflict between StyleGuides",
        examples=[
            "The actions contradict each other: one suggests being formal while the other suggests being casual",
            "The conditions overlap but lead to opposing actions",
        ],
    ),
]

StyleGuideCoherenceCheckSeverityField: TypeAlias = Annotated[
    int,
    Field(
        description="Numerical rating of the contradiction's severity (1-10, where 10 is most severe)",
        examples=[5, 8],
        ge=1,
        le=10,
    ),
]


StyleGuideEventMessageField: TypeAlias = Annotated[
    str,
    Field(
        description="The message shown to the customer as part of the event.",
        examples=["Thanks for being awesome and choosing us ;)"],
    ),
]

StyleGuidePrincipleField: TypeAlias = Annotated[
    str,
    Field(
        description="A statement explaining the overarching style principle.",
        examples=["Use a friendly tone with a hint of humor"],
    ),
]

StyleGuideViolationField: TypeAlias = Annotated[
    str,
    Field(
        description="Explains why the 'before' version violates or contradicts the style principle.",
        examples=["Too formal and lacks an engaging tone"],
    ),
]


class StyleGuideCoherenceCheckKindDTO(Enum):
    """
    The specific relationship between the contradicting style guides.
    """

    CONTRADICTION_WITH_EXISTING_STYLE_GUIDE = "contradiction_with_existing_style_guide"
    CONTRADICTION_WITH_ANOTHER_EVALUATED_STYLE_GUIDE = (
        "contradiction_with_another_evaluated_style_guide"
    )


class StyleGuideEventDTO(DefaultBaseModel):
    """
    Represents a single event within a style guide example,
    including its source and the message to the user.
    """

    source: EventSourceDTO
    message: StyleGuideEventMessageField


class StyleGuideExampleDTO(DefaultBaseModel):
    """
    Represents a style guide example consisting of 'before' and 'after' event sequences,
    along with a 'violation' description explaining the issue in the 'before' version.
    """

    before: Sequence[StyleGuideEventDTO]
    after: Sequence[StyleGuideEventDTO]
    violation: StyleGuideViolationField


style_guide_content_example: ExampleJson = {
    "principle": "Use inclusive language and a positive tone",
    "examples": [
        {
            "before": [
                {
                    "source": "ai_agent",
                    "message": "Your request is denied. Try again.",
                }
            ],
            "after": [
                {
                    "source": "ai_agent",
                    "message": "Unfortunately we can't fulfill that request right now. Let's see what else we can do to help!",
                }
            ],
            "violation": "The 'before' response is abrupt and lacks empathy.",
        }
    ],
}


class StyleGuideContentDTO(
    DefaultBaseModel,
    json_schema_extra={"example": style_guide_content_example},
):
    """
    Represents a style guide's content, including:
      - A 'principle' to highlight the main style guideline
      - One or more 'examples' illustrating correct and incorrect usage
    """

    principle: StyleGuidePrincipleField
    examples: Sequence[StyleGuideExampleDTO]


style_guides_coherence_check_example: ExampleJson = {
    "kind": "contradiction_with_existing_style_guide",
    "first": {
        "principle": "Use Imperial units",
        "examples": [
            {
                "before": [
                    {
                        "source": "customer",
                        "message": "How tall do I have to be to get on the roller coaster?",
                    },
                    {"source": "ai_agent", "message": "1.8 meters"},
                ],
                "after": [
                    {
                        "source": "customer",
                        "message": "How tall do I have to be to get on the roller coaster?",
                    },
                    {"source": "ai_agent", "message": "5 feet 11 inches"},
                ],
                "violation": "The 'before' message is in metric units, which is not imperial.",
            }
        ],
    },
    "second": {
        "principle": "Use Metric units",
        "examples": [
            {
                "before": [
                    {
                        "source": "customer",
                        "message": "How tall do I have to be to get on the roller coaster?",
                    },
                    {"source": "ai_agent", "message": "5 feet 11 inches"},
                ],
                "after": [
                    {
                        "source": "customer",
                        "message": "How tall do I have to be to get on the roller coaster?",
                    },
                    {"source": "ai_agent", "message": "1.8 meters"},
                ],
                "violation": "The 'before' message is in imperial units, which is not metric.",
            }
        ],
    },
    "issue": "Conflicting approaches to writing units",
    "severity": 8,
}


class StyleGuideCoherenceCheckDTO(
    DefaultBaseModel,
    json_schema_extra={"example": style_guides_coherence_check_example},
):
    """
    Indicates a potential contradiction between two different style guides.
    Helps to identify inconsistent or conflicting style rules.
    """

    kind: StyleGuideCoherenceCheckKindDTO
    first: StyleGuideContentDTO
    second: StyleGuideContentDTO
    issue: GuidelineCoherenceCheckIssueField
    severity: GuidelineCoherenceCheckSeverityField


style_guide_invoice_data_example: ExampleJson = {
    "coherence_checks": [style_guides_coherence_check_example],
}


class StyleGuidePayloadOperationDTO(Enum):
    """
    The kind of operation that should be performed on the payload.
    """

    ADD = "add"
    UPDATE = "update"


StyleGuideIdField: TypeAlias = Annotated[
    StyleGuideId,
    Field(
        description="Unique identifier for the style guide",
        examples=["sg_abc123"],
    ),
]

StyleGuidePayloadCoherenceCheckField: TypeAlias = Annotated[
    bool,
    Field(
        description="Whether to check for contradictions with other StyleGuides",
        examples=[True, False],
    ),
]

style_guide_payload_example: ExampleJson = {
    "content": {
        "principle": "Adopt a whimsical tone and include jokes or playful remarks.",
        "examples": [
            {
                "before": [
                    {
                        "source": "customer",
                        "message": "Can you tell me the weather for today?",
                    },
                    {
                        "source": "ai_agent",
                        "message": "It's sunny and warm, about 75°F.",
                    },
                ],
                "after": [
                    {
                        "source": "customer",
                        "message": "Can you tell me the weather for today?",
                    },
                    {
                        "source": "ai_agent",
                        "message": "It's sunny and warm, about 75°F. Perfect weather for pretending you're on a tropical island... even if you're just in your backyard with a kiddie pool!",
                    },
                ],
                "violation": "The 'before' message does not include a joke or playful remark.",
            },
        ],
    },
    "operation": "add",
    "updated_id": None,
    "coherence_check": True,
}


class StyleGuidePayloadDTO(
    DefaultBaseModel,
    json_schema_extra={"example": style_guide_payload_example},
):
    """Payload data for a StyleGuide operation"""

    content: StyleGuideContentDTO
    operation: StyleGuidePayloadOperationDTO
    updated_id: Optional[StyleGuideIdField] = None
    coherence_check: StyleGuidePayloadCoherenceCheckField


class PayloadKindDTO(Enum):
    """
    The kind of payload.
    choices are ['guideline', 'style_guide']
    """

    GUIDELINE = "guideline"
    STYLE_GUIDE = "style_guide"


class PayloadDTO(
    DefaultBaseModel,
    json_schema_extra={"example": payload_example},
):
    """
    A container for a guideline OR style guide payload along with its kind
    """

    kind: PayloadKindDTO
    guideline: Optional[GuidelinePayloadDTO] = None
    style_guide: Optional[StyleGuidePayloadDTO] = None


class GuidelineInvoiceDataDTO(
    DefaultBaseModel,
    json_schema_extra={"example": guideline_invoice_data_example},
):
    """Evaluation results for a Guideline, including contradiction checks and connection proposals"""

    coherence_checks: Sequence[GuidelinesCoherenceCheckDTO]
    connection_propositions: Optional[Sequence[ConnectionPropositionDTO]] = None


class StyleGuideInvoiceDataDTO(
    DefaultBaseModel,
    json_schema_extra={"example": style_guide_invoice_data_example},
):
    """Evaluation results for a StyleGuide, including contradiction checks"""

    coherence_checks: Sequence[StyleGuideCoherenceCheckDTO]


invoice_data_example: ExampleJson = {
    "guideline": guideline_invoice_data_example,
    "style_guide": style_guide_invoice_data_example,
}


class InvoiceDataDTO(
    DefaultBaseModel,
    json_schema_extra={"example": invoice_data_example},
):
    """
    Contains the relevant invoice data.

    At this point only `guideline` is suppoerted.
    """

    guideline: Optional[GuidelineInvoiceDataDTO] = None
    style_guide: Optional[StyleGuideInvoiceDataDTO] = None


ChecksumField: TypeAlias = Annotated[
    str,
    Field(
        description="Checksum of the invoice content.",
        examples=["abc123def456"],
    ),
]

ApprovedField: TypeAlias = Annotated[
    bool,
    Field(
        description="Indicates whether the evaluation task the invoice represents has been approved.",
        examples=[True],
    ),
]

ErrorField: TypeAlias = Annotated[
    str,
    Field(
        description="Describes any error that occurred during evaluation.",
        examples=["Failed to process evaluation due to invalid payload."],
    ),
]

invoice_example: ExampleJson = {
    "payload": {
        "kind": "guideline",
        "guideline": {
            "content": {
                "condition": "when customer asks about pricing",
                "action": "provide current pricing information",
            },
            "operation": "add",
            "updated_id": None,
            "coherence_check": True,
            "connection_proposition": True,
        },
    },
    "checksum": "abc123def456",
    "approved": True,
    "data": {
        "guideline": {
            "coherence_checks": [
                {
                    "kind": "semantic_overlap",
                    "first": {
                        "condition": "when customer asks about pricing",
                        "action": "provide current pricing information",
                    },
                    "second": {
                        "condition": "if customer inquires about cost",
                        "action": "share the latest pricing details",
                    },
                    "issue": "These guidelines handle similar scenarios",
                    "severity": "warning",
                }
            ],
            "connection_propositions": [
                {
                    "check_kind": "semantic_similarity",
                    "source": {
                        "condition": "when customer asks about pricing",
                        "action": "provide current pricing information",
                    },
                    "target": {
                        "condition": "if customer inquires about cost",
                        "action": "share the latest pricing details",
                    },
                }
            ],
        }
    },
    "error": None,
}


class InvoiceDTO(
    DefaultBaseModel,
    json_schema_extra={"example": invoice_example},
):
    """
    Represents the result of evaluating a single payload in an evaluation task.

    An Invoice is a comprehensive record of the evaluation results, including:
    - A `payload` describing what kind of data was evaluated (e.g., guideline or style_guide).
    - A `checksum` to verify the integrity of the content.
    - An `approved` flag indicating whether the results are finalized.
    - An optional `data` object containing detailed findings (coherence checks, connections, etc.),
    if the evaluation is approved.
    - An optional `error` message if the evaluation failed.
    """

    payload: PayloadDTO
    checksum: ChecksumField
    approved: ApprovedField
    data: Optional[InvoiceDataDTO] = None
    error: Optional[ErrorField] = None


def style_guide_content_to_dto(content: StyleGuideContent) -> StyleGuideContentDTO:
    def style_guide_event_to_dto(event: StyleGuideEvent) -> StyleGuideEventDTO:
        return StyleGuideEventDTO(
            source=EventSourceDTO(event.source),
            message=event.message,
        )

    def style_guide_example_to_dto(example: StyleGuideExample) -> StyleGuideExampleDTO:
        return StyleGuideExampleDTO(
            before=[style_guide_event_to_dto(e) for e in example.before],
            after=[style_guide_event_to_dto(e) for e in example.after],
            violation=example.violation,
        )

    return StyleGuideContentDTO(
        principle=content.principle,
        examples=[style_guide_example_to_dto(e) for e in content.examples],
    )


def style_guide_content_dto_to_content(dto: StyleGuideContentDTO) -> StyleGuideContent:
    def event_dto_to_event(dto: StyleGuideEventDTO) -> StyleGuideEvent:
        def source_dto_to_source(source: EventSourceDTO) -> EventSource:
            if source == EventSourceDTO.CUSTOMER:
                return "customer"
            elif source == EventSourceDTO.CUSTOMER_UI:
                return "customer_ui"
            elif source == EventSourceDTO.HUMAN_AGENT:
                return "human_agent"
            elif source == EventSourceDTO.HUMAN_AGENT_ON_BEHALF_OF_AI_AGENT:
                return "human_agent_on_behalf_of_ai_agent"
            elif source == EventSourceDTO.AI_AGENT:
                return "ai_agent"
            elif source == EventSourceDTO.SYSTEM:
                return "system"

        return StyleGuideEvent(source=source_dto_to_source(dto.source), message=dto.message)

    def example_dto_to_example(dto: StyleGuideExampleDTO) -> StyleGuideExample:
        return StyleGuideExample(
            violation=dto.violation,
            before=[event_dto_to_event(e) for e in dto.before],
            after=[event_dto_to_event(e) for e in dto.after],
        )

    return StyleGuideContent(
        principle=dto.principle, examples=[example_dto_to_example(e) for e in dto.examples]
    )


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
