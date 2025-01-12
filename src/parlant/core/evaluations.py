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

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import (
    Mapping,
    NamedTuple,
    NewType,
    Optional,
    Sequence,
    TypeAlias,
    Union,
)
from typing_extensions import Literal, Self, TypedDict, override

from parlant.core.async_utils import ReaderWriterLock, Timeout
from parlant.core.common import (
    SCHEMA_VERSION_UNKNOWN,
    AgentId,
    CoherenceCheckKind,
    ConnectionPropositionKind,
    GuidelineContent,
    GuidelineId,
    ItemNotFoundError,
    JSONSerializable,
    SchemaVersion,
    UniqueId,
    Version,
    generate_id,
)
from parlant.core.documents.evaluations import (
    _CoherenceCheckDocument_v1,
    _ConnectionPropositionDocument_v1,
    _EvaluationDocument_v1,
    _GuidelineContentDocument_v1,
    _GuidelinePayloadDocument_v1,
    _InvoiceDocument_v1,
    _InvoiceGuidelineDataDocument_v1,
    _PayloadDocument_v1,
)
from parlant.core.persistence.common import VersionedStore, ObjectId
from parlant.core.persistence.document_database import DocumentCollection, DocumentDatabase

EvaluationId = NewType("EvaluationId", str)


class EvaluationStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


class PayloadKind(Enum):
    GUIDELINE = auto()


@dataclass(frozen=True)
class GuidelinePayload:
    content: GuidelineContent
    operation: Literal["add", "update"]
    coherence_check: bool
    connection_proposition: bool
    updated_id: Optional[GuidelineId] = None

    def __repr__(self) -> str:
        return f"condition: {self.content.condition}, action: {self.content.action}"


Payload: TypeAlias = Union[GuidelinePayload]


class PayloadDescriptor(NamedTuple):
    kind: PayloadKind
    payload: Payload


@dataclass(frozen=True)
class CoherenceCheck:
    kind: CoherenceCheckKind
    first: GuidelineContent
    second: GuidelineContent
    issue: str
    severity: int


@dataclass(frozen=True)
class ConnectionProposition:
    check_kind: ConnectionPropositionKind
    source: GuidelineContent
    target: GuidelineContent


@dataclass(frozen=True)
class InvoiceGuidelineData:
    coherence_checks: Sequence[CoherenceCheck]
    connection_propositions: Optional[Sequence[ConnectionProposition]]
    _type: Literal["guideline"] = "guideline"  # Union discrimator for Pydantic


InvoiceData: TypeAlias = Union[InvoiceGuidelineData]


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
    agent_id: AgentId
    creation_utc: datetime
    status: EvaluationStatus
    error: Optional[str]
    invoices: Sequence[Invoice]
    progress: float


class EvaluationUpdateParams(TypedDict, total=False):
    status: EvaluationStatus
    error: Optional[str]
    invoices: Sequence[Invoice]
    progress: float


class EvaluationStore(ABC):
    @abstractmethod
    async def create_evaluation(
        self,
        agent_id: AgentId,
        payload_descriptors: Sequence[PayloadDescriptor],
        creation_utc: Optional[datetime] = None,
        extra: Optional[Mapping[str, JSONSerializable]] = None,
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


class _GuidelineContentDocument(TypedDict):
    condition: str
    action: str


class _GuidelinePayloadDocument(TypedDict):
    content: _GuidelineContentDocument
    action: Literal["add", "update"]
    updated_id: Optional[GuidelineId]
    coherence_check: bool
    connection_proposition: bool


_PayloadDocument = Union[_GuidelinePayloadDocument]


class _CoherenceCheckDocument(TypedDict):
    kind: CoherenceCheckKind
    first: _GuidelineContentDocument
    second: _GuidelineContentDocument
    issue: str
    severity: int


class _ConnectionPropositionDocument(TypedDict):
    check_kind: ConnectionPropositionKind
    source: _GuidelineContentDocument
    target: _GuidelineContentDocument


class _InvoiceGuidelineDataDocument(TypedDict):
    coherence_checks: Sequence[_CoherenceCheckDocument]
    connection_propositions: Optional[Sequence[_ConnectionPropositionDocument]]


_InvoiceDataDocument = Union[_InvoiceGuidelineDataDocument]


class _InvoiceDocument(TypedDict, total=False):
    kind: str
    payload: _PayloadDocument
    checksum: str
    state_version: str
    approved: bool
    data: Optional[_InvoiceDataDocument]
    error: Optional[str]


class _EvaluationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    agent_id: AgentId
    creation_utc: str
    status: str
    error: Optional[str]
    invoices: Sequence[_InvoiceDocument]
    progress: float


class EvaluationDocumentStore(EvaluationStore, VersionedStore):
    VERSION = SchemaVersion(3)

    def __init__(self, database: DocumentDatabase):
        if database.version == SCHEMA_VERSION_UNKNOWN:
            database.version = self.VERSION
        self._database = database
        self._collection: DocumentCollection[_EvaluationDocument_v1]

        self._lock = ReaderWriterLock()

    async def __aenter__(self) -> Self:
        self._collection = await self._database.get_or_create_collection(
            name="evaluations",
            schema=_EvaluationDocument_v1,
        )
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        pass

    def _serialize_invoice(self, invoice: Invoice) -> _InvoiceDocument_v1:
        def serialize_coherence_check(check: CoherenceCheck) -> _CoherenceCheckDocument_v1:
            return _CoherenceCheckDocument_v1(
                kind=check.kind,
                first=_GuidelineContentDocument_v1(
                    condition=check.first.condition,
                    action=check.first.action,
                ),
                second=_GuidelineContentDocument_v1(
                    condition=check.second.condition,
                    action=check.second.action,
                ),
                issue=check.issue,
                severity=check.severity,
            )

        def serialize_connection_proposition(
            cp: ConnectionProposition,
        ) -> _ConnectionPropositionDocument_v1:
            return _ConnectionPropositionDocument_v1(
                check_kind=cp.check_kind,
                source=_GuidelineContentDocument_v1(
                    condition=cp.source.condition,
                    action=cp.source.action,
                ),
                target=_GuidelineContentDocument_v1(
                    condition=cp.target.condition,
                    action=cp.target.action,
                ),
            )

        def serialize_invoice_guideline_data(
            data: InvoiceGuidelineData,
        ) -> _InvoiceGuidelineDataDocument_v1:
            return _InvoiceGuidelineDataDocument_v1(
                coherence_checks=[serialize_coherence_check(cc) for cc in data.coherence_checks],
                connection_propositions=(
                    [serialize_connection_proposition(cp) for cp in data.connection_propositions]
                    if data.connection_propositions
                    else None
                ),
            )

        def serialize_payload(payload: Payload) -> _PayloadDocument_v1:
            if isinstance(payload, GuidelinePayload):
                return _GuidelinePayloadDocument_v1(
                    content=_GuidelineContentDocument_v1(
                        condition=payload.content.condition,
                        action=payload.content.action,
                    ),
                    action=payload.operation,
                    updated_id=payload.updated_id,
                    coherence_check=payload.coherence_check,
                    connection_proposition=payload.connection_proposition,
                )
            else:
                raise TypeError(f"Unknown payload type: {type(payload)}")

        kind = invoice.kind.name  # Convert Enum to string
        if kind == "GUIDELINE":
            return _InvoiceDocument_v1(
                kind=kind,
                payload=serialize_payload(invoice.payload),
                checksum=invoice.checksum,
                state_version=invoice.state_version,
                approved=invoice.approved,
                data=serialize_invoice_guideline_data(invoice.data) if invoice.data else None,
                error=invoice.error,
            )
        else:
            raise ValueError(f"Unsupported invoice kind: {kind}")

    def _serialize_evaluation(self, evaluation: Evaluation) -> _EvaluationDocument_v1:
        return _EvaluationDocument_v1(
            id=ObjectId(evaluation.id),
            agent_id=evaluation.agent_id,
            creation_utc=evaluation.creation_utc.isoformat(),
            status=evaluation.status.name,
            error=evaluation.error,
            invoices=[self._serialize_invoice(inv) for inv in evaluation.invoices],
            progress=evaluation.progress,
        )

    def _deserialize_evaluation(self, evaluation_document: _EvaluationDocument_v1) -> Evaluation:
        def deserialize_guideline_content_document(
            gc_doc: _GuidelineContentDocument_v1,
        ) -> GuidelineContent:
            return GuidelineContent(
                condition=gc_doc["condition"],
                action=gc_doc["action"],
            )

        def deserialize_coherence_check_document(
            cc_doc: _CoherenceCheckDocument_v1,
        ) -> CoherenceCheck:
            return CoherenceCheck(
                kind=cc_doc["kind"],
                first=deserialize_guideline_content_document(cc_doc["first"]),
                second=deserialize_guideline_content_document(cc_doc["second"]),
                issue=cc_doc["issue"],
                severity=cc_doc["severity"],
            )

        def deserialize_connection_proposition_document(
            cp_doc: _ConnectionPropositionDocument_v1,
        ) -> ConnectionProposition:
            return ConnectionProposition(
                check_kind=cp_doc["check_kind"],
                source=deserialize_guideline_content_document(cp_doc["source"]),
                target=deserialize_guideline_content_document(cp_doc["target"]),
            )

        def deserialize_invoice_guideline_data(
            data_doc: _InvoiceGuidelineDataDocument_v1,
        ) -> InvoiceGuidelineData:
            return InvoiceGuidelineData(
                coherence_checks=[
                    deserialize_coherence_check_document(cc_doc)
                    for cc_doc in data_doc["coherence_checks"]
                ],
                connection_propositions=(
                    [
                        deserialize_connection_proposition_document(cp_doc)
                        for cp_doc in data_doc["connection_propositions"]
                    ]
                    if data_doc["connection_propositions"] is not None
                    else None
                ),
            )

        def deserialize_payload_document(
            kind: PayloadKind, payload_doc: _PayloadDocument_v1
        ) -> Payload:
            if kind == PayloadKind.GUIDELINE:
                return GuidelinePayload(
                    content=GuidelineContent(
                        condition=payload_doc["content"]["condition"],
                        action=payload_doc["content"]["action"],
                    ),
                    operation=payload_doc["action"],
                    updated_id=payload_doc["updated_id"],
                    coherence_check=payload_doc["coherence_check"],
                    connection_proposition=payload_doc["connection_proposition"],
                )
            else:
                raise ValueError(f"Unsupported payload kind: {kind}")

        def deserialize_invoice_document(invoice_doc: _InvoiceDocument_v1) -> Invoice:
            kind = PayloadKind[invoice_doc["kind"]]

            payload = deserialize_payload_document(kind, invoice_doc["payload"])

            data_doc = invoice_doc.get("data")
            if data_doc is not None:
                data = deserialize_invoice_guideline_data(data_doc)
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

        return Evaluation(
            id=evaluation_id,
            agent_id=AgentId(evaluation_document["agent_id"]),
            creation_utc=creation_utc,
            status=status,
            error=evaluation_document.get("error"),
            invoices=invoices,
            progress=evaluation_document["progress"],
        )

    @property
    @override
    def database_version(self) -> SchemaVersion:
        return self._database.version

    @property
    @override
    def database_name(self) -> str:
        if "file_path" in self._database.__dict__ and isinstance(
            db_path := getattr(self._database, "file_path"), Path
        ):
            return db_path.name
        return "unknown"

    @override
    async def create_evaluation(
        self,
        agent_id: AgentId,
        payload_descriptors: Sequence[PayloadDescriptor],
        creation_utc: Optional[datetime] = None,
        extra: Optional[Mapping[str, JSONSerializable]] = None,
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
                agent_id=agent_id,
                status=EvaluationStatus.PENDING,
                creation_utc=creation_utc,
                error=None,
                invoices=invoices,
                progress=0.0,
            )

            await self._collection.insert_one(self._serialize_evaluation(evaluation=evaluation))

        return evaluation

    @override
    async def update_evaluation(
        self,
        evaluation_id: EvaluationId,
        params: EvaluationUpdateParams,
    ) -> Evaluation:
        async with self._lock.writer_lock:
            evaluation = await self.read_evaluation(evaluation_id)

            update_params: _EvaluationDocument_v1 = {}
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

        return self._deserialize_evaluation(result.updated_document)

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

        return self._deserialize_evaluation(evaluation_document=evaluation_document)

    @override
    async def list_evaluations(
        self,
    ) -> Sequence[Evaluation]:
        async with self._lock.reader_lock:
            return [
                self._deserialize_evaluation(evaluation_document=e)
                for e in await self._collection.find(filters={})
            ]


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
