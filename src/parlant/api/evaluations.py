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
from typing import Annotated, Sequence, TypeAlias, cast
from fastapi import APIRouter, HTTPException, Path, Query, Request, status
from pydantic import Field

from parlant.api import common
from parlant.api.authorization import AuthorizationPolicy, Operation
from parlant.api.common import (
    EvaluationStatusDTO,
    RuleContentDTO,
    RuleIdField,
    RulePayloadOperationDTO,
    JSONSerializableDTO,
    PayloadKindDTO,
    ExampleJson,
    ToolIdDTO,
    apigen_config,
    operation_dto_to_operation,
)
from parlant.core.agents import AgentId
from parlant.core.application import Application
from parlant.core.async_utils import Timeout
from parlant.core.common import DefaultBaseModel
from parlant.core.evaluations import (
    Evaluation,
    EvaluationId,
    EvaluationStatus,
    RulePayload,
    InvoiceRuleData,
    PayloadOperation,
    InvoiceData,
    Payload,
    PayloadDescriptor,
    PayloadKind,
)
from parlant.core.rules import RuleContent
from parlant.core.services.indexing.evaluation_service import (
    EvaluationValidationError,
)
from parlant.core.tools import ToolId

API_GROUP = "evaluations"


def _evaluation_status_to_dto(
    status: EvaluationStatus,
) -> EvaluationStatusDTO:
    return cast(
        EvaluationStatusDTO,
        {
            EvaluationStatus.PENDING: "pending",
            EvaluationStatus.RUNNING: "running",
            EvaluationStatus.COMPLETED: "completed",
            EvaluationStatus.FAILED: "failed",
        }[status],
    )


RulePayloadActionPropositionField: TypeAlias = Annotated[
    bool,
    Field(
        description="Whether the action proposition is enabled",
        examples=[True],
    ),
]

RulePayloadPropertiesPropositionField: TypeAlias = Annotated[
    bool,
    Field(
        description="Properties proposition",
        examples=[{"action_proposition": True}],
    ),
]

RulePayloadJourneyNodePropositionField: TypeAlias = Annotated[
    bool,
    Field(
        description="Journey step proposition",
        examples=[{"action_proposition": True}],
    ),
]

RulePayloadSignalPropositionField: TypeAlias = Annotated[
    bool,
    Field(
        description="Signals proposition",
        examples=[True],
    ),
]

RulePayloadTitlePropositionField: TypeAlias = Annotated[
    bool,
    Field(
        description="Title proposition",
        examples=[True],
    ),
]

EvaluationAgentIdField: TypeAlias = Annotated[
    AgentId,
    Field(
        description="Agent context to use when evaluating agent-dependent propositions such as rule signals.",
        examples=["agent_123xz"],
    ),
]

rule_payload_example: ExampleJson = {
    "content": {
        "condition": "User asks about product pricing",
        "action": "Provide current price list and any active discounts",
    },
    "tool_ids": ["google_calendar:get_events"],
    "operation": "add",
    "updated_id": None,
    "action_proposition": True,
    "properties_proposition": {"continuous": True},
}


class RulePayloadDTO(
    DefaultBaseModel,
    json_schema_extra={"example": rule_payload_example},
):
    """Payload data for a Rule operation"""

    content: RuleContentDTO
    tool_ids: Sequence[ToolIdDTO]
    operation: RulePayloadOperationDTO
    updated_id: RuleIdField | None = None
    title: common.RuleTitleField | None = None
    agent_id: EvaluationAgentIdField | None = None
    action_proposition: RulePayloadActionPropositionField = False
    properties_proposition: RulePayloadPropertiesPropositionField = False
    journey_node_proposition: RulePayloadJourneyNodePropositionField = False
    signal_proposition: RulePayloadSignalPropositionField = False
    title_proposition: RulePayloadTitlePropositionField = False


payload_example: ExampleJson = {
    "kind": "rule",
    "rule": {
        "content": {
            "condition": "User asks about product pricing",
            "action": None,
        },
        "operation": "add",
        "updated_id": None,
        "action_proposition": True,
        "properties_proposition": True,
    },
}


class PayloadDTO(
    DefaultBaseModel,
    json_schema_extra={"example": payload_example},
):
    kind: PayloadKindDTO
    rule: RulePayloadDTO | None = None


properties_proposition_example: ExampleJson = {
    "continuous": True,
    "internal_action": "Provide current price list and any active discounts",
}


ChecksumField: TypeAlias = Annotated[
    str,
    Field(
        description="Checksum of the invoice content",
        examples=["abc123def456"],
    ),
]

ApprovedField: TypeAlias = Annotated[
    bool,
    Field(
        description="Whether the evaluation task the invoice represents has been approved",
        examples=[True],
    ),
]


ErrorField: TypeAlias = Annotated[
    str,
    Field(
        description="Error message if the evaluation failed",
        examples=["Failed to process evaluation due to invalid payload"],
    ),
]


ActionPropositionField: TypeAlias = Annotated[
    str,
    Field(
        description="Proposed action proposition",
        examples=["provide current pricing information"],
    ),
]

PropertiesPropositionField: TypeAlias = Annotated[
    dict[str, JSONSerializableDTO] | None,
    Field(
        description="Properties proposition",
        examples=[{"continuous": True}],
    ),
]

invoice_example: ExampleJson = {
    "payload": {
        "kind": "rule",
        "rule": {
            "content": {
                "condition": "when customer asks about pricing",
                "action": "provide current pricing information",
            },
            "operation": "add",
            "updated_id": None,
            "action_proposition": True,
            "properties_proposition": True,
        },
    },
    "checksum": "abc123def456",
    "approved": True,
    "data": {
        "rule": {
            "action_proposition": {
                "content": {
                    "condition": "when customer asks about pricing",
                    "action": "provide current pricing information",
                },
                "properties_proposition": {
                    "continuous": True,
                },
            },
        }
    },
    "error": None,
}

rule_invoice_data_example: ExampleJson = {
    "properties_proposition": properties_proposition_example,
}


class RuleInvoiceDataDTO(
    DefaultBaseModel,
    json_schema_extra={"example": rule_invoice_data_example},
):
    """Evaluation results for a Rule, including action propositions"""

    action_proposition: ActionPropositionField | None = None
    properties_proposition: PropertiesPropositionField | None = None
    signals_proposition: Sequence[str] | None = None
    anti_signals_proposition: Sequence[str] | None = None
    title_proposition: str | None = None


invoice_data_example: ExampleJson = {"rule": rule_invoice_data_example}


class InvoiceDataDTO(
    DefaultBaseModel,
    json_schema_extra={"example": invoice_data_example},
):
    """
    Contains the relevant invoice data.

    At this point only `rule` is supported.
    """

    rule: RuleInvoiceDataDTO | None = None


class InvoiceDTO(
    DefaultBaseModel,
    json_schema_extra={"example": invoice_example},
):
    """Represents the result of evaluating a single payload in an evaluation task.

    An invoice is a comprehensive record of the evaluation results for a single payload.
    """

    payload: PayloadDTO
    checksum: ChecksumField
    approved: ApprovedField
    data: InvoiceDataDTO | None = None
    error: ErrorField | None = None


def _payload_from_dto(dto: PayloadDTO, agent_id: AgentId | None = None) -> Payload:
    if dto.kind == PayloadKindDTO.RULE:
        if not dto.rule:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Missing Rule payload",
            )

        if (
            not dto.rule.action_proposition
            and not dto.rule.properties_proposition
            and not dto.rule.journey_node_proposition
            and not dto.rule.signal_proposition
            and not dto.rule.title_proposition
        ):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="At least one of action_proposition, properties_proposition, journey_node_proposition, signal_proposition or title_proposition must be enabled",
            )

        return RulePayload(
            content=RuleContent(
                condition=dto.rule.content.condition,
                action=dto.rule.content.action,
                description=dto.rule.content.description,
            ),
            tool_ids=[
                ToolId(service_name=t.service_name, tool_name=t.tool_name)
                for t in dto.rule.tool_ids
            ],
            operation=operation_dto_to_operation(dto.rule.operation),
            updated_id=dto.rule.updated_id,
            title=dto.rule.title,
            agent_id=dto.rule.agent_id or agent_id,
            action_proposition=dto.rule.action_proposition,
            properties_proposition=dto.rule.properties_proposition,
            journey_node_proposition=dto.rule.journey_node_proposition,
            signal_proposition=dto.rule.signal_proposition,
            title_proposition=dto.rule.title_proposition,
        )

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail="Unsupported DTO kind",
    )


def _operation_to_operation_dto(
    operation: PayloadOperation,
) -> RulePayloadOperationDTO:
    if dto := {
        PayloadOperation.ADD: RulePayloadOperationDTO.ADD,
        PayloadOperation.UPDATE: RulePayloadOperationDTO.UPDATE,
    }.get(operation):
        return dto

    raise ValueError(f"Unsupported operation: {operation}")


def _payload_descriptor_to_dto(descriptor: PayloadDescriptor) -> PayloadDTO:
    if descriptor.kind == PayloadKind.RULE:
        return PayloadDTO(
            kind=PayloadKindDTO.RULE,
            rule=RulePayloadDTO(
                content=RuleContentDTO(
                    condition=cast(RulePayload, descriptor.payload).content.condition,
                    action=cast(RulePayload, descriptor.payload).content.action,
                    description=cast(RulePayload, descriptor.payload).content.description,
                ),
                tool_ids=[
                    ToolIdDTO(service_name=t.service_name, tool_name=t.tool_name)
                    for t in cast(RulePayload, descriptor.payload).tool_ids
                ],
                operation=_operation_to_operation_dto(descriptor.payload.operation),
                updated_id=cast(RulePayload, descriptor.payload).updated_id,
                title=cast(RulePayload, descriptor.payload).title,
                agent_id=cast(RulePayload, descriptor.payload).agent_id,
                action_proposition=cast(RulePayload, descriptor.payload).action_proposition,
                properties_proposition=cast(RulePayload, descriptor.payload).properties_proposition,
                journey_node_proposition=cast(
                    RulePayload, descriptor.payload
                ).journey_node_proposition,
                signal_proposition=cast(RulePayload, descriptor.payload).signal_proposition,
                title_proposition=cast(RulePayload, descriptor.payload).title_proposition,
            ),
        )

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail="Unsupported descriptor kind",
    )


def _invoice_data_to_dto(
    kind: PayloadKind,
    invoice_data: InvoiceData,
) -> InvoiceDataDTO:
    if kind == PayloadKind.RULE:
        rule_data = cast(InvoiceRuleData, invoice_data)
        properties = rule_data.properties_proposition or {}
        action_proposition = cast(str | None, properties.get("internal_action"))

        return InvoiceDataDTO(
            rule=RuleInvoiceDataDTO(
                action_proposition=action_proposition,
                properties_proposition=rule_data.properties_proposition,
                signals_proposition=rule_data.signals_proposition,
                anti_signals_proposition=rule_data.anti_signals_proposition,
                title_proposition=rule_data.title_proposition,
            ),
        )

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        detail="Unsupported descriptor kind",
    )


evaluation_creation_params_example: ExampleJson = {
    "agent_id": "a1g2e3n4t5",
    "payloads": [
        {
            "kind": "rule",
            "rule": {
                "content": {
                    "condition": "when customer asks about pricing",
                    "action": None,
                },
                "operation": "add",
                "action_proposition": True,
            },
        }
    ],
}


class EvaluationCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": evaluation_creation_params_example},
):
    """Parameters for creating a new evaluation task"""

    agent_id: EvaluationAgentIdField | None = None
    payloads: Sequence[PayloadDTO]


EvaluationIdPath: TypeAlias = Annotated[
    EvaluationId,
    Path(
        description="Unique identifier of the evaluation to retrieve",
        examples=["eval_123xz"],
    ),
]

EvaluationProgressField: TypeAlias = Annotated[
    float,
    Field(
        description="Progress of the evaluation from 0.0 to 100.0",
        ge=0.0,
        le=100.0,
        examples=[75.0],
    ),
]

CreationUtcField: TypeAlias = Annotated[
    datetime,
    Field(
        description="UTC timestamp when the evaluation was created",
    ),
]


evaluation_example: ExampleJson = {
    "id": "eval_123xz",
    "status": "completed",
    "progress": 100.0,
    "creation_utc": "2024-03-24T12:00:00Z",
    "error": None,
    "invoices": [
        {
            "payload": {
                "kind": "rule",
                "rule": {
                    "content": {
                        "condition": "when customer asks about pricing",
                        "action": "provide current pricing information",
                    },
                    "operation": "add",
                    "updated_id": None,
                    "action_proposition": True,
                    "properties_proposition": True,
                },
            },
            "checksum": "abc123def456",
            "approved": True,
            "data": {
                "rule": {
                    "properties_proposition": {
                        "continuous": True,
                        "internal_action": "Provide current price list and any active discounts",
                    },
                }
            },
            "error": None,
        }
    ],
}


class EvaluationDTO(
    DefaultBaseModel,
    json_schema_extra={"example": evaluation_example},
):
    """An evaluation task information tracking analysis of payloads."""

    id: EvaluationIdPath
    status: EvaluationStatusDTO
    progress: EvaluationProgressField
    creation_utc: CreationUtcField
    error: ErrorField | None = None
    invoices: Sequence[InvoiceDTO]


WaitForCompletionQuery: TypeAlias = Annotated[
    int,
    Query(
        description="Maximum time in seconds to wait for evaluation completion",
        ge=0,
    ),
]


def _evaluation_to_dto(evaluation: Evaluation) -> EvaluationDTO:
    return EvaluationDTO(
        id=evaluation.id,
        status=_evaluation_status_to_dto(evaluation.status),
        progress=evaluation.progress,
        creation_utc=evaluation.creation_utc,
        invoices=[
            InvoiceDTO(
                payload=_payload_descriptor_to_dto(
                    PayloadDescriptor(kind=invoice.kind, payload=invoice.payload)
                ),
                checksum=invoice.checksum,
                approved=invoice.approved,
                data=_invoice_data_to_dto(invoice.kind, invoice.data) if invoice.data else None,
                error=invoice.error,
            )
            for invoice in evaluation.invoices
        ],
        error=evaluation.error,
    )


def create_router(
    authorization_policy: AuthorizationPolicy,
    app: Application,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_evaluation",
        response_model=EvaluationDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Evaluation successfully created. Returns the initial evaluation state.",
                "content": common.example_json_content(evaluation_example),
            },
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Validation error in evaluation parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create"),
    )
    async def create_evaluation(
        request: Request,
        params: EvaluationCreationParamsDTO,
    ) -> EvaluationDTO:
        """
        Creates a new evaluation task for the specified payloads.

        Returns immediately with the created evaluation's initial state.
        """
        await authorization_policy.authorize(
            request=request,
            operation=Operation.CREATE_EVALUATION,
        )

        try:
            evaluation = await app.evaluations.create(
                payloads=[_payload_from_dto(p, params.agent_id) for p in params.payloads]
            )

        except EvaluationValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=str(exc),
            )

        return _evaluation_to_dto(evaluation)

    @router.get(
        "/{evaluation_id}",
        operation_id="read_evaluation",
        response_model=EvaluationDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Evaluation details successfully retrieved.",
                "content": common.example_json_content(evaluation_example),
            },
            status.HTTP_404_NOT_FOUND: {"description": "Evaluation not found"},
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Validation error in evaluation parameters"
            },
            status.HTTP_504_GATEWAY_TIMEOUT: {
                "description": "Timeout waiting for evaluation completion"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve"),
    )
    async def read_evaluation(
        request: Request,
        evaluation_id: EvaluationIdPath,
        wait_for_completion: WaitForCompletionQuery = 60,
    ) -> EvaluationDTO:
        """Retrieves the current state of an evaluation.

        * If wait_for_completion == 0, returns current state immediately.
        * If wait_for_completion > 0, waits for completion/failure or timeout. Defaults to 60.

        Notes:
        When wait_for_completion > 0:
        - Returns final state if evaluation completes within timeout
        - Raises 504 if timeout is reached before completion
        """
        await authorization_policy.authorize(
            request=request,
            operation=Operation.READ_EVALUATION,
        )

        if wait_for_completion > 0:
            if not await app.evaluations.wait_for_completion(
                evaluation_id=evaluation_id,
                timeout=Timeout(wait_for_completion),
            ):
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail="Request timed out",
                )

        evaluation = await app.evaluations.read(evaluation_id=evaluation_id)
        return _evaluation_to_dto(evaluation)

    return router
