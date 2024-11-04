from enum import Enum
from typing import Literal, Optional, TypeAlias, TypedDict, Union, cast

from parlant.core.common import DefaultBaseModel
from parlant.core.evaluations import (
    CoherenceCheckKind,
    ConnectionPropositionKind,
)
from parlant.core.guideline_connections import ConnectionKind
from parlant.core.guidelines import GuidelineContent, GuidelineId


class EvaluationStatusDTO(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ConnectionKindDTO(Enum):
    ENTAILS = "entails"
    SUGGESTS = "suggests"


class PayloadKindDTO(Enum):
    GUIDELINE = "guideline"


class GuidelineContentDTO(TypedDict):
    predicate: str
    action: str


class GuidelinePayloadDTO(DefaultBaseModel):
    kind: PayloadKindDTO
    content: GuidelineContentDTO
    operation: Literal["add", "update"]
    updated_id: Optional[GuidelineId] = None
    coherence_check: bool
    connection_proposition: bool


PayloadDTO: TypeAlias = Union[GuidelinePayloadDTO]


class CoherenceCheckDTO(DefaultBaseModel):
    kind: CoherenceCheckKind
    first: GuidelineContent
    second: GuidelineContent
    issue: str
    severity: int


class ConnectionPropositionDTO(DefaultBaseModel):
    check_kind: ConnectionPropositionKind
    source: GuidelineContent
    target: GuidelineContent
    connection_kind: ConnectionKindDTO


class GuidelineInvoiceDataDTO(DefaultBaseModel):
    coherence_checks: list[CoherenceCheckDTO]
    connection_propositions: Optional[list[ConnectionPropositionDTO]]


InvoiceDataDTO: TypeAlias = Union[GuidelineInvoiceDataDTO]


def connection_kind_to_dto(kind: ConnectionKind) -> ConnectionKindDTO:
    return cast(
        ConnectionKindDTO,
        {
            ConnectionKind.ENTAILS: "entails",
            ConnectionKind.SUGGESTS: "suggests",
        }[kind],
    )


def connection_kind_dto_to_connection_kind(dto: ConnectionKindDTO) -> ConnectionKind:
    return {
        ConnectionKindDTO.ENTAILS: ConnectionKind.ENTAILS,
        ConnectionKindDTO.SUGGESTS: ConnectionKind.SUGGESTS,
    }[dto]


class ToolIdDTO(DefaultBaseModel):
    service_name: str
    tool_name: str


JSONSerializableDTO: TypeAlias = Union[
    str,
    int,
    float,
    bool,
    list[Union[str, int, float, bool]],
    dict[str, Union[str, int, float, bool]],
]
