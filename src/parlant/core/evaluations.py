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

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, auto
from itertools import chain
from typing import (
    Mapping,
    NamedTuple,
    NewType,
    Optional,
    Sequence,
    TypeAlias,
    Union,
    cast,
)
from typing_extensions import Literal, override, TypedDict, Self

from parlant.core.agents import AgentId
from parlant.core.async_utils import ReaderWriterLock, Timeout, safe_gather
from parlant.core.common import (
    ItemNotFoundError,
    try_or_none,
    JSONSerializable,
    UniqueId,
    Version,
    generate_id,
)
from parlant.core.rules import RuleContent, RuleId
from parlant.core.journeys import JourneyEdgeId, JourneyId, JourneyNodeId
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentDatabase,
    DocumentCollection,
)
from parlant.core.persistence.document_database_helper import (
    DocumentMigrationHelper,
    DocumentStoreMigrationHelper,
)
from parlant.core.groups import GroupId
from parlant.core.tools import ToolId

EvaluationId = NewType("EvaluationId", str)


class EvaluationStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


class PayloadKind(Enum):
    RULE = auto()
    JOURNEY = auto()


class PayloadOperation(Enum):
    ADD = "add"
    UPDATE = "update"


@dataclass(frozen=True)
class RulePayload:
    content: RuleContent
    tool_ids: Sequence[ToolId]
    operation: PayloadOperation
    action_proposition: bool
    properties_proposition: bool
    journey_node_proposition: bool
    signal_proposition: bool = False
    title_proposition: bool = False
    title: Optional[str] = None
    agent_id: Optional[AgentId] = None
    updated_id: Optional[RuleId] = None

    def __repr__(self) -> str:
        return f"condition: {self.content.condition}, action: {self.content.action}"


@dataclass(frozen=True)
class JourneyPayload:
    journey_id: JourneyId
    operation: PayloadOperation


Payload: TypeAlias = Union[RulePayload, JourneyPayload]


class PayloadDescriptor(NamedTuple):
    kind: PayloadKind
    payload: Payload


@dataclass(frozen=True)
class InvoiceRuleData:
    properties_proposition: Optional[dict[str, JSONSerializable]]
    signals_proposition: Optional[Sequence[str]] = None
    anti_signals_proposition: Optional[Sequence[str]] = None
    title_proposition: Optional[str] = None
    _type: Literal["rule"] = "rule"  # Union discriminator for Pydantic


@dataclass(frozen=True)
class InvoiceJourneyData:
    node_properties_proposition: dict[JourneyNodeId, dict[str, JSONSerializable]]
    edge_properties_proposition: dict[JourneyEdgeId, dict[str, JSONSerializable]]
    _type: Literal["journey"] = "journey"  # Union discriminator for Pydantic


InvoiceData: TypeAlias = Union[InvoiceRuleData, InvoiceJourneyData]


@dataclass(frozen=True)
class Invoice:
    kind: PayloadKind
    payload: Payload
    checksum: str
    state_version: str
    approved: bool
    data: Optional[InvoiceData]
    error: Optional[str]


@dataclass(frozen=True)
class Evaluation:
    id: EvaluationId
    creation_utc: datetime
    status: EvaluationStatus
    error: Optional[str]
    invoices: Sequence[Invoice]
    progress: float
    groups: Sequence[GroupId]


class EvaluationUpdateParams(TypedDict, total=False):
    status: EvaluationStatus
    error: Optional[str]
    invoices: Sequence[Invoice]
    progress: float


class EvaluationStore(ABC):
    @abstractmethod
    async def create_evaluation(
        self,
        payload_descriptors: Sequence[PayloadDescriptor],
        creation_utc: Optional[datetime] = None,
        extra: Optional[Mapping[str, JSONSerializable]] = None,
        groups: Optional[Sequence[GroupId]] = None,
    ) -> Evaluation: ...

    @abstractmethod
    async def update_evaluation(
        self,
        evaluation_id: EvaluationId,
        params: EvaluationUpdateParams,
    ) -> Evaluation: ...

    @abstractmethod
    async def read_evaluation(
        self,
        evaluation_id: EvaluationId,
    ) -> Evaluation: ...

    @abstractmethod
    async def list_evaluations(
        self,
    ) -> Sequence[Evaluation]: ...

    @abstractmethod
    async def upsert_group(
        self,
        evaluation_id: EvaluationId,
        group_id: GroupId,
        creation_utc: Optional[datetime] = None,
    ) -> bool: ...

    @abstractmethod
    async def remove_group(
        self,
        evaluation_id: EvaluationId,
        group_id: GroupId,
    ) -> None: ...


class RuleContentDocument(TypedDict, total=False):
    condition: str
    action: Optional[str]
    description: Optional[str]


class RulePayloadDocument_v0_1_0(TypedDict):
    content: RuleContentDocument
    action: Literal["add", "update"]
    updated_id: Optional[RuleId]
    coherence_check: bool
    connection_proposition: bool


class RulePayloadDocument_v0_2_0(TypedDict):
    content: RuleContentDocument
    tool_ids: Sequence[ToolId]
    action: Literal["add", "update"]
    updated_id: Optional[RuleId]
    coherence_check: bool
    connection_proposition: bool
    action_proposition: bool
    properties_proposition: bool


class RulePayloadDocument_v0_4_0(TypedDict):
    content: RuleContentDocument
    tool_ids: Sequence[ToolId]
    action: Literal["add", "update"]
    updated_id: Optional[RuleId]
    coherence_check: bool
    connection_proposition: bool
    action_proposition: bool
    properties_proposition: bool
    journey_node_proposition: bool


class RulePayloadDocument(TypedDict):
    content: RuleContentDocument
    tool_ids: Sequence[ToolId]
    action: Literal["add", "update"]
    updated_id: Optional[RuleId]
    action_proposition: bool
    properties_proposition: bool
    journey_node_proposition: bool
    signal_proposition: bool
    title_proposition: bool
    title: Optional[str]
    agent_id: Optional[AgentId]


class JourneyPayloadDocument(TypedDict):
    journey_id: JourneyId
    action: Literal["add", "update"]


_PayloadDocument = Union[RulePayloadDocument, JourneyPayloadDocument]


class _CoherenceCheckDocument(TypedDict):
    kind: str
    first: RuleContentDocument
    second: RuleContentDocument
    issue: str
    severity: int


class _ConnectionPropositionDocument(TypedDict):
    check_kind: str
    source: RuleContentDocument
    target: RuleContentDocument


class _InvoiceRuleDataDocument_v0_1_0(TypedDict):
    coherence_checks: Optional[Sequence[_CoherenceCheckDocument]]
    connection_propositions: Optional[Sequence[_ConnectionPropositionDocument]]


class InvoiceRuleDataDocument_v0_2_0(TypedDict):
    coherence_checks: Optional[Sequence[_CoherenceCheckDocument]]
    connection_propositions: Optional[Sequence[_ConnectionPropositionDocument]]
    action_proposition: Optional[str]
    properties_proposition: Optional[dict[str, JSONSerializable]]


_InvoiceDataDocument_v0_2_0 = Union[InvoiceRuleDataDocument_v0_2_0]


class InvoiceRuleDataDocument_v0_3_0(TypedDict):
    coherence_checks: Optional[Sequence[_CoherenceCheckDocument]]
    connection_propositions: Optional[Sequence[_ConnectionPropositionDocument]]
    action_proposition: Optional[str]
    properties_proposition: Optional[dict[str, JSONSerializable]]


_InvoiceDataDocument_v0_3_0 = Union[InvoiceRuleDataDocument_v0_3_0]


class InvoiceRuleDataDocument_v0_4_0(TypedDict):
    coherence_checks: Optional[Sequence[_CoherenceCheckDocument]]
    connection_propositions: Optional[Sequence[_ConnectionPropositionDocument]]
    properties_proposition: Optional[dict[str, JSONSerializable]]


class InvoiceJourneyDataDocument(TypedDict):
    node_properties_proposition: dict[JourneyNodeId, dict[str, JSONSerializable]]
    edge_properties_proposition: dict[JourneyEdgeId, dict[str, JSONSerializable]]


_InvoiceDataDocument_v0_4_0 = Union[InvoiceRuleDataDocument_v0_4_0, InvoiceJourneyDataDocument]


class InvoiceRuleDataDocument(TypedDict):
    properties_proposition: Optional[dict[str, JSONSerializable]]
    signals_proposition: Optional[Sequence[str]]
    anti_signals_proposition: Optional[Sequence[str]]
    title_proposition: Optional[str]


_InvoiceDataDocument = Union[InvoiceRuleDataDocument, InvoiceJourneyDataDocument]


class InvoiceDocument_v0_1_0(TypedDict, total=False):
    kind: str
    payload: RulePayloadDocument_v0_1_0
    checksum: str
    state_version: str
    approved: bool
    data: Optional[_InvoiceRuleDataDocument_v0_1_0]
    error: Optional[str]


class InvoiceDocument_v0_2_0(TypedDict, total=False):
    kind: str
    payload: RulePayloadDocument_v0_2_0
    checksum: str
    state_version: str
    approved: bool
    data: Optional[_InvoiceDataDocument_v0_2_0]
    error: Optional[str]


class InvoiceDocument_v0_3_0(TypedDict, total=False):
    kind: str
    payload: _PayloadDocument
    checksum: str
    state_version: str
    approved: bool
    data: Optional[_InvoiceDataDocument_v0_3_0]
    error: Optional[str]


class InvoiceDocument_v0_4_0(TypedDict, total=False):
    kind: str
    payload: _PayloadDocument
    checksum: str
    state_version: str
    approved: bool
    data: Optional[_InvoiceDataDocument_v0_4_0]
    error: Optional[str]


class InvoiceDocument(TypedDict, total=False):
    kind: str
    payload: _PayloadDocument
    checksum: str
    state_version: str
    approved: bool
    data: Optional[_InvoiceDataDocument]
    error: Optional[str]


class EvaluationDocument_v0_1_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    agent_id: AgentId
    creation_utc: str
    status: str
    error: Optional[str]
    invoices: Sequence[InvoiceDocument_v0_1_0]
    progress: float


class EvaluationDocument_v0_2_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    agent_id: AgentId
    creation_utc: str
    status: str
    error: Optional[str]
    invoices: Sequence[InvoiceDocument_v0_2_0]
    progress: float


class EvaluationDocument_v0_3_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    status: str
    error: Optional[str]
    invoices: Sequence[InvoiceDocument_v0_3_0]
    progress: float


class EvaluationDocument_v0_4_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    status: str
    error: Optional[str]
    invoices: Sequence[InvoiceDocument_v0_4_0]
    progress: float


class EvaluationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    status: str
    error: Optional[str]
    invoices: Sequence[InvoiceDocument]
    progress: float


class EvaluationTagAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    evaluation_id: EvaluationId
    group_id: GroupId


class EvaluationDocumentStore(EvaluationStore):
    VERSION = Version.from_string("0.5.0")

    def __init__(
        self,
        database: DocumentDatabase,
        allow_migration: bool = False,
        collections_prefix: str | None = None,
    ) -> None:
        self._database = database
        self._collection: DocumentCollection[EvaluationDocument]
        self._tag_association_collection: DocumentCollection[EvaluationTagAssociationDocument]

        self._allow_migration = allow_migration
        self._collections_prefix = collections_prefix
        self._lock = ReaderWriterLock()

    async def tag_association_document_loader(
        self, doc: BaseDocument
    ) -> Optional[EvaluationTagAssociationDocument]:
        async def v0_1_0_to_v0_2_0(doc: BaseDocument) -> Optional[BaseDocument]:
            raise Exception(
                "This code should not be reached! Please run the 'parlant-prepare-migration' script."
            )

        async def v0_2_0_to_v0_3_0(doc: BaseDocument) -> Optional[BaseDocument]:
            doc = cast(EvaluationTagAssociationDocument, doc)

            return EvaluationTagAssociationDocument(
                id=ObjectId(doc["id"]),
                version=Version.String("0.3.0"),
                creation_utc=doc["creation_utc"],
                evaluation_id=EvaluationId(doc["evaluation_id"]),
                group_id=GroupId(doc["group_id"]),
            )

        async def v0_3_0_to_v0_4_0(doc: BaseDocument) -> Optional[BaseDocument]:
            doc = cast(EvaluationTagAssociationDocument, doc)

            return EvaluationTagAssociationDocument(
                id=ObjectId(doc["id"]),
                version=Version.String("0.4.0"),
                creation_utc=doc["creation_utc"],
                evaluation_id=EvaluationId(doc["evaluation_id"]),
                group_id=GroupId(doc["group_id"]),
            )

        async def v0_4_0_to_v0_5_0(doc: BaseDocument) -> Optional[BaseDocument]:
            doc = cast(EvaluationTagAssociationDocument, doc)

            return EvaluationTagAssociationDocument(
                id=ObjectId(doc["id"]),
                version=Version.String("0.5.0"),
                creation_utc=doc["creation_utc"],
                evaluation_id=EvaluationId(doc["evaluation_id"]),
                group_id=GroupId(doc["group_id"]),
            )

        return await DocumentMigrationHelper[EvaluationTagAssociationDocument](
            self,
            {
                "0.1.0": v0_1_0_to_v0_2_0,
                "0.2.0": v0_2_0_to_v0_3_0,
                "0.3.0": v0_3_0_to_v0_4_0,
                "0.4.0": v0_4_0_to_v0_5_0,
            },
        ).migrate(doc)

    async def document_loader(self, doc: BaseDocument) -> Optional[EvaluationDocument]:
        async def v0_1_0_to_v0_2_0(doc: BaseDocument) -> Optional[BaseDocument]:
            raise Exception(
                "This code should not be reached! Please run the 'parlant-prepare-migration' script."
            )

        async def v0_2_0_to_v0_3_0(doc: BaseDocument) -> Optional[BaseDocument]:
            doc = cast(EvaluationDocument_v0_2_0, doc)

            return EvaluationDocument_v0_3_0(
                id=ObjectId(doc["id"]),
                version=Version.String("0.3.0"),
                creation_utc=doc["creation_utc"],
                status=doc["status"],
                error=doc.get("error"),
                invoices=[
                    InvoiceDocument_v0_3_0(
                        kind=inv["kind"],
                        payload=RulePayloadDocument_v0_4_0(
                            content=RuleContentDocument(
                                condition=inv["payload"]["content"]["condition"],
                                action=inv["payload"]["content"].get("action"),
                            ),
                            tool_ids=inv["payload"]["tool_ids"],
                            action=inv["payload"]["action"],
                            updated_id=inv["payload"].get("updated_id"),
                            coherence_check=inv["payload"]["coherence_check"],
                            connection_proposition=inv["payload"]["connection_proposition"],
                            action_proposition=inv["payload"]["action_proposition"],
                            properties_proposition=inv["payload"]["properties_proposition"],
                            journey_node_proposition=False,
                        ),
                        checksum=inv["checksum"],
                        state_version=inv["state_version"],
                        approved=inv["approved"],
                        data=inv["data"],
                    )
                    for inv in doc["invoices"]
                ],
                progress=doc["progress"],
            )

        async def v0_3_0_to_v0_4_0(doc: BaseDocument) -> Optional[BaseDocument]:
            doc = cast(EvaluationDocument_v0_3_0, doc)

            return EvaluationDocument(
                id=ObjectId(doc["id"]),
                version=Version.String("0.4.0"),
                creation_utc=doc["creation_utc"],
                status=doc["status"],
                error=doc.get("error"),
                invoices=[
                    InvoiceDocument(
                        kind=inv["kind"],
                        payload=inv["payload"],
                        checksum=inv["checksum"],
                        state_version=inv["state_version"],
                        approved=inv["approved"],
                        data=InvoiceRuleDataDocument_v0_4_0(
                            coherence_checks=inv["data"].get("coherence_checks"),
                            connection_propositions=inv["data"].get("connection_propositions"),
                            properties_proposition={
                                **cast(
                                    dict[str, JSONSerializable],
                                    inv["data"].get("properties_proposition", {}),
                                ),
                                **({"internal_action": inv["data"].get("action_proposition", {})}),
                            }
                            if inv["data"].get("properties_proposition")
                            or inv["data"].get("action_proposition")
                            else None,
                        )
                        if inv["data"]
                        else None,
                    )
                    for inv in doc["invoices"]
                ],
                progress=doc["progress"],
            )

        async def v0_4_0_to_v0_5_0(doc: BaseDocument) -> Optional[BaseDocument]:
            doc = cast(EvaluationDocument_v0_4_0, doc)

            return EvaluationDocument(
                id=ObjectId(doc["id"]),
                version=Version.String("0.5.0"),
                creation_utc=doc["creation_utc"],
                status=doc["status"],
                error=doc.get("error"),
                invoices=[
                    InvoiceDocument(
                        kind=inv["kind"],
                        payload=inv["payload"],
                        checksum=inv["checksum"],
                        state_version=inv["state_version"],
                        approved=inv["approved"],
                        data=InvoiceRuleDataDocument(
                            properties_proposition={
                                **cast(
                                    dict[str, JSONSerializable],
                                    inv["data"].get("properties_proposition", {}),
                                ),
                                **cast(
                                    dict[str, JSONSerializable],
                                    {"internal_action": inv["data"].get("action_proposition", {})},
                                ),
                            }
                            if inv["data"].get("properties_proposition")
                            or inv["data"].get("action_proposition")
                            else None,
                            signals_proposition=None,
                            anti_signals_proposition=None,
                            title_proposition=None,
                        )
                        if inv["data"]
                        else None,
                    )
                    for inv in doc["invoices"]
                ],
                progress=doc["progress"],
            )

        return await DocumentMigrationHelper[EvaluationDocument](
            self,
            {
                "0.1.0": v0_1_0_to_v0_2_0,
                "0.2.0": v0_2_0_to_v0_3_0,
                "0.3.0": v0_3_0_to_v0_4_0,
            },
        ).migrate(doc)

    async def __aenter__(self) -> Self:
        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
            collections_prefix=self._collections_prefix,
        ):
            self._collection = await self._database.get_or_create_collection(
                name=f"{self._collections_prefix}_evaluations"
                if self._collections_prefix
                else "evaluations",
                schema=EvaluationDocument,
                document_loader=self.document_loader,
            )

            self._tag_association_collection = await self._database.get_or_create_collection(
                name=f"{self._collections_prefix}_evaluation_tag_associations"
                if self._collections_prefix
                else "evaluation_tag_associations",
                schema=EvaluationTagAssociationDocument,
                document_loader=self.tag_association_document_loader,
            )

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        pass

    def _serialize_invoice(self, invoice: Invoice) -> InvoiceDocument:
        def serialize_invoice_rule_data(
            data: InvoiceRuleData,
        ) -> InvoiceRuleDataDocument:
            return InvoiceRuleDataDocument(
                properties_proposition=(
                    data.properties_proposition if data.properties_proposition is not None else None
                ),
                signals_proposition=(
                    data.signals_proposition if data.signals_proposition is not None else None
                ),
                anti_signals_proposition=(
                    data.anti_signals_proposition
                    if data.anti_signals_proposition is not None
                    else None
                ),
                title_proposition=(
                    data.title_proposition if data.title_proposition is not None else None
                ),
            )

        def serialize_invoice_journey_data(
            data: InvoiceJourneyData,
        ) -> InvoiceJourneyDataDocument:
            return InvoiceJourneyDataDocument(
                node_properties_proposition=data.node_properties_proposition or {},
                edge_properties_proposition=data.edge_properties_proposition or {},
            )

        def serialize_payload(payload: Payload) -> _PayloadDocument:
            if isinstance(payload, RulePayload):
                return RulePayloadDocument(
                    content=RuleContentDocument(
                        condition=payload.content.condition,
                        action=payload.content.action or None,
                        description=payload.content.description or None,
                    ),
                    tool_ids=payload.tool_ids,
                    action=payload.operation.value,
                    updated_id=payload.updated_id,
                    action_proposition=payload.action_proposition,
                    properties_proposition=payload.properties_proposition,
                    journey_node_proposition=payload.journey_node_proposition,
                    signal_proposition=payload.signal_proposition,
                    title_proposition=payload.title_proposition,
                    title=payload.title,
                    agent_id=payload.agent_id,
                )
            elif isinstance(payload, JourneyPayload):
                return JourneyPayloadDocument(
                    journey_id=payload.journey_id,
                    action=payload.operation.value,
                )
            elif isinstance(payload, JourneyPayload):
                return JourneyPayloadDocument(
                    journey_id=payload.journey_id,
                    action=payload.operation.value,
                )
            else:
                raise TypeError(f"Unknown payload type: {type(payload)}")

        kind = invoice.kind.name  # Convert Enum to string
        if kind == "RULE":
            return InvoiceDocument(
                kind=kind,
                payload=serialize_payload(invoice.payload),
                checksum=invoice.checksum,
                state_version=invoice.state_version,
                approved=invoice.approved,
                data=serialize_invoice_rule_data(cast(InvoiceRuleData, invoice.data))
                if invoice.data
                else None,
                error=invoice.error,
            )
        elif kind == "JOURNEY":
            return InvoiceDocument(
                kind=kind,
                payload=serialize_payload(invoice.payload),
                checksum=invoice.checksum,
                state_version=invoice.state_version,
                approved=invoice.approved,
                data=serialize_invoice_journey_data(cast(InvoiceJourneyData, invoice.data))
                if invoice.data
                else None,
                error=invoice.error,
            )
        else:
            raise ValueError(f"Unsupported invoice kind: {kind}")

    def _serialize_evaluation(self, evaluation: Evaluation) -> EvaluationDocument:
        return EvaluationDocument(
            id=ObjectId(evaluation.id),
            version=self.VERSION.to_string(),
            creation_utc=evaluation.creation_utc.isoformat(),
            status=evaluation.status.name,
            error=evaluation.error,
            invoices=[self._serialize_invoice(inv) for inv in evaluation.invoices],
            progress=evaluation.progress,
        )

    async def _deserialize_evaluation(self, evaluation_document: EvaluationDocument) -> Evaluation:
        def deserialize_rule_content_document(
            gc_doc: RuleContentDocument,
        ) -> RuleContent:
            return RuleContent(
                condition=gc_doc["condition"],
                action=gc_doc["action"],
                description=gc_doc.get("description"),
            )

        def deserialize_invoice_rule_data(
            data_doc: InvoiceRuleDataDocument,
        ) -> InvoiceRuleData:
            return InvoiceRuleData(
                properties_proposition=(
                    data_doc["properties_proposition"]
                    if data_doc["properties_proposition"] is not None
                    else None
                ),
                signals_proposition=data_doc.get("signals_proposition"),
                anti_signals_proposition=data_doc.get("anti_signals_proposition"),
                title_proposition=data_doc.get("title_proposition"),
            )

        def deserialize_payload_document(
            kind: PayloadKind,
            payload_doc: _PayloadDocument,
        ) -> Payload:
            if kind == PayloadKind.RULE:
                payload_doc = cast(RulePayloadDocument, payload_doc)

                return RulePayload(
                    content=RuleContent(
                        condition=payload_doc["content"]["condition"],
                        action=payload_doc["content"]["action"] or None,
                        description=payload_doc["content"].get("description"),
                    ),
                    tool_ids=payload_doc["tool_ids"],
                    operation=PayloadOperation(payload_doc["action"]),
                    updated_id=payload_doc["updated_id"],
                    action_proposition=payload_doc["action_proposition"],
                    properties_proposition=payload_doc["properties_proposition"],
                    journey_node_proposition=payload_doc["journey_node_proposition"],
                    signal_proposition=payload_doc.get("signal_proposition", False),
                    title_proposition=payload_doc.get("title_proposition", False),
                    title=payload_doc.get("title"),
                    agent_id=payload_doc.get("agent_id"),
                )
            elif kind == PayloadKind.JOURNEY:
                payload_doc = cast(JourneyPayloadDocument, payload_doc)

                return JourneyPayload(
                    journey_id=payload_doc["journey_id"],
                    operation=PayloadOperation(payload_doc["action"]),
                )
            elif kind == PayloadKind.JOURNEY:
                payload_doc = cast(JourneyPayloadDocument, payload_doc)

                return JourneyPayload(
                    journey_id=payload_doc["journey_id"],
                    operation=PayloadOperation(payload_doc["action"]),
                )
            else:
                raise ValueError(f"Unsupported payload kind: {kind}")

        def deserialize_invoice_document(invoice_doc: InvoiceDocument) -> Invoice:
            kind = PayloadKind[invoice_doc["kind"]]

            payload = deserialize_payload_document(kind, invoice_doc["payload"])

            data_doc = invoice_doc.get("data")
            if data_doc is not None:
                if kind == PayloadKind.RULE:
                    data: Optional[InvoiceData] = deserialize_invoice_rule_data(
                        cast(InvoiceRuleDataDocument, data_doc)
                    )
                elif kind == PayloadKind.JOURNEY:
                    data = InvoiceJourneyData(
                        node_properties_proposition=cast(InvoiceJourneyDataDocument, data_doc)[
                            "node_properties_proposition"
                        ],
                        edge_properties_proposition=cast(InvoiceJourneyDataDocument, data_doc)[
                            "edge_properties_proposition"
                        ],
                    )
            else:
                data = None

            return Invoice(
                kind=kind,
                payload=payload,
                checksum=invoice_doc["checksum"],
                state_version=invoice_doc["state_version"],
                approved=invoice_doc["approved"],
                data=data,
                error=invoice_doc.get("error"),
            )

        evaluation_id = EvaluationId(evaluation_document["id"])
        creation_utc = datetime.fromisoformat(evaluation_document["creation_utc"])

        status = EvaluationStatus[evaluation_document["status"]]

        invoices = [
            deserialize_invoice_document(inv_doc) for inv_doc in evaluation_document["invoices"]
        ]

        async with self._lock.reader_lock:
            tags_docs = await self._tag_association_collection.find(
                filters={"evaluation_id": {"$eq": evaluation_id}},
            )
            groups = [GroupId(tag_doc["group_id"]) for tag_doc in tags_docs]

        return Evaluation(
            id=evaluation_id,
            creation_utc=creation_utc,
            status=status,
            error=evaluation_document.get("error"),
            invoices=invoices,
            progress=evaluation_document["progress"],
            groups=groups,
        )

    @override
    async def create_evaluation(
        self,
        payload_descriptors: Sequence[PayloadDescriptor],
        creation_utc: Optional[datetime] = None,
        extra: Optional[Mapping[str, JSONSerializable]] = None,
        groups: Optional[Sequence[GroupId]] = None,
    ) -> Evaluation:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)

            evaluation_id = EvaluationId(generate_id())

            invoices = [
                Invoice(
                    kind=k,
                    payload=p,
                    state_version="",
                    checksum="",
                    approved=False,
                    data=None,
                    error=None,
                )
                for k, p in payload_descriptors
            ]

            evaluation = Evaluation(
                id=evaluation_id,
                status=EvaluationStatus.PENDING,
                creation_utc=creation_utc,
                error=None,
                invoices=invoices,
                progress=0.0,
                groups=groups or [],
            )

            await self._collection.insert_one(self._serialize_evaluation(evaluation=evaluation))

            for group in groups or []:
                await self._tag_association_collection.insert_one(
                    document={
                        "id": ObjectId(generate_id()),
                        "version": self.VERSION.to_string(),
                        "creation_utc": creation_utc.isoformat(),
                        "evaluation_id": evaluation_id,
                        "group_id": group,
                    }
                )

        return evaluation

    @override
    async def update_evaluation(
        self,
        evaluation_id: EvaluationId,
        params: EvaluationUpdateParams,
    ) -> Evaluation:
        async with self._lock.writer_lock:
            evaluation = await self.read_evaluation(evaluation_id)

            update_params: EvaluationDocument = {}
            if "invoices" in params:
                update_params["invoices"] = [self._serialize_invoice(i) for i in params["invoices"]]

            if "status" in params:
                update_params["status"] = params["status"].name
                update_params["error"] = params["error"] if "error" in params else None

            if "progress" in params:
                update_params["progress"] = params["progress"]

            result = await self._collection.update_one(
                filters={"id": {"$eq": evaluation.id}},
                params=update_params,
            )

        assert result.updated_document

        return await self._deserialize_evaluation(result.updated_document)

    @override
    async def read_evaluation(
        self,
        evaluation_id: EvaluationId,
    ) -> Evaluation:
        async with self._lock.reader_lock:
            evaluation_document = await self._collection.find_one(
                filters={"id": {"$eq": evaluation_id}},
            )

        if not evaluation_document:
            raise ItemNotFoundError(item_id=UniqueId(evaluation_id))

        return await self._deserialize_evaluation(evaluation_document=evaluation_document)

    @override
    async def list_evaluations(
        self,
    ) -> Sequence[Evaluation]:
        async with self._lock.reader_lock:
            return [
                await self._deserialize_evaluation(evaluation_document=e)
                for e in await self._collection.find(filters={})
            ]

    @override
    async def upsert_group(
        self,
        evaluation_id: EvaluationId,
        group_id: GroupId,
        creation_utc: Optional[datetime] = None,
    ) -> bool:
        async with self._lock.writer_lock:
            evaluation = await self.read_evaluation(evaluation_id)

            if group_id in evaluation.groups:
                return False

            creation_utc = creation_utc or datetime.now(timezone.utc)

            association_document: EvaluationTagAssociationDocument = {
                "id": ObjectId(generate_id()),
                "version": self.VERSION.to_string(),
                "creation_utc": creation_utc.isoformat(),
                "evaluation_id": evaluation_id,
                "group_id": group_id,
            }

            _ = await self._tag_association_collection.insert_one(document=association_document)

            evaluation_document = await self._collection.find_one({"id": {"$eq": evaluation_id}})

        if not evaluation_document:
            raise ItemNotFoundError(item_id=UniqueId(evaluation_id))

        return True

    @override
    async def remove_group(
        self,
        evaluation_id: EvaluationId,
        group_id: GroupId,
    ) -> None:
        async with self._lock.writer_lock:
            delete_result = await self._tag_association_collection.delete_one(
                {
                    "evaluation_id": {"$eq": evaluation_id},
                    "group_id": {"$eq": group_id},
                }
            )

            if delete_result.deleted_count == 0:
                raise ItemNotFoundError(item_id=UniqueId(group_id))

            evaluation_document = await self._collection.find_one({"id": {"$eq": evaluation_id}})

        if not evaluation_document:
            raise ItemNotFoundError(item_id=UniqueId(evaluation_id))


class EvaluationListener(ABC):
    @abstractmethod
    async def wait_for_completion(
        self,
        evaluation_id: EvaluationId,
        timeout: Timeout = Timeout.infinite(),
    ) -> bool: ...


class PollingEvaluationListener(EvaluationListener):
    def __init__(self, evaluation_store: EvaluationStore) -> None:
        self._evaluation_store = evaluation_store

    @override
    async def wait_for_completion(
        self,
        evaluation_id: EvaluationId,
        timeout: Timeout = Timeout.infinite(),
    ) -> bool:
        while True:
            evaluation = await self._evaluation_store.read_evaluation(
                evaluation_id,
            )

            if evaluation.status in [EvaluationStatus.COMPLETED, EvaluationStatus.FAILED]:
                return True
            elif timeout.expired():
                return False
            else:
                await timeout.wait_up_to(1)


class CompositeEvaluationStore(EvaluationStore):
    def __init__(
        self,
        writable_store: EvaluationStore,
        readable_stores: Sequence[EvaluationStore],
    ) -> None:
        self._writable_store = writable_store
        self._readable_stores = readable_stores
        self._all_stores: Sequence[EvaluationStore] = [writable_store, *readable_stores]

    @override
    async def create_evaluation(
        self,
        payload_descriptors: Sequence[PayloadDescriptor],
        creation_utc: Optional[datetime] = None,
        extra: Optional[Mapping[str, JSONSerializable]] = None,
        groups: Optional[Sequence[GroupId]] = None,
    ) -> Evaluation:
        return await self._writable_store.create_evaluation(
            payload_descriptors=payload_descriptors,
            creation_utc=creation_utc,
            extra=extra,
            groups=groups,
        )

    @override
    async def update_evaluation(
        self,
        evaluation_id: EvaluationId,
        params: EvaluationUpdateParams,
    ) -> Evaluation:
        return await self._writable_store.update_evaluation(evaluation_id, params)

    @override
    async def read_evaluation(
        self,
        evaluation_id: EvaluationId,
    ) -> Evaluation:
        results = await safe_gather(
            *[try_or_none(store.read_evaluation(evaluation_id)) for store in self._all_stores]
        )
        result = next((r for r in results if r is not None), None)
        if result is None:
            raise ItemNotFoundError(item_id=UniqueId(evaluation_id))
        return result

    @override
    async def list_evaluations(
        self,
    ) -> Sequence[Evaluation]:
        results = await safe_gather(*[store.list_evaluations() for store in self._all_stores])
        return list(chain.from_iterable(results))

    @override
    async def upsert_group(
        self,
        evaluation_id: EvaluationId,
        group_id: GroupId,
        creation_utc: Optional[datetime] = None,
    ) -> bool:
        return await self._writable_store.upsert_group(evaluation_id, group_id, creation_utc)

    @override
    async def remove_group(
        self,
        evaluation_id: EvaluationId,
        group_id: GroupId,
    ) -> None:
        return await self._writable_store.remove_group(evaluation_id, group_id)
