from typing import Literal, Optional, TypeAlias, TypedDict, Union, cast
from emcie.server.core.common import DefaultBaseModel
from emcie.server.core.evaluations import (
    CoherenceCheckKind,
    ConnectionPropositionKind,
)
from emcie.server.core.guideline_connections import ConnectionKind
from emcie.server.core.guidelines import GuidelineContent

EvaluationStatusDTO = Literal["pending", "running", "completed", "failed"]

ConnectionKindDTO = Literal["entails", "suggests"]

PayloadKindDTO = Literal["guideline"]


class GuidelinePayloadDTO(TypedDict):
    kind: PayloadKindDTO
    predicate: str
    action: str


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


class InvoiceGuidelineDataDTO(DefaultBaseModel):
    coherence_checks: list[CoherenceCheckDTO]
    connection_propositions: Optional[list[ConnectionPropositionDTO]]


InvoiceDataDTO: TypeAlias = Union[InvoiceGuidelineDataDTO]


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
        "entails": ConnectionKind.ENTAILS,
        "suggests": ConnectionKind.SUGGESTS,
    }[dto]
