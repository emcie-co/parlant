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

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, auto
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
from parlant.core.async_utils import ReaderWriterLock, Timeout
from parlant.core.common import (
    ItemNotFoundError,
    JSONSerializable,
    UniqueId,
    Version,
    generate_id,
)
from parlant.core.guidelines import GuidelineContent, GuidelineId
from parlant.core.persistence.common import ObjectId
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentDatabase,
    DocumentCollection,
)
from parlant.core.persistence.document_database_helper import DocumentStoreMigrationHelper
from parlant.core.tags import TagId
from parlant.core.tools import ToolId

EvaluationId = NewType("EvaluationId", str)


class EvaluationStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


class PayloadKind(Enum):
    GUIDELINE = auto()


class CoherenceCheckKind(Enum):
    # Legacy and will be removed in the future
    CONTRADICTION_WITH_EXISTING_GUIDELINE = "contradiction_with_existing_guideline"
    CONTRADICTION_WITH_ANOTHER_EVALUATED_GUIDELINE = (
        "contradiction_with_another_evaluated_guideline"
    )


class EntailmentRelationshipPropositionKind(Enum):
    # Legacy and will be removed in the future
    CONNECTION_WITH_EXISTING_GUIDELINE = "connection_with_existing_guideline"
    CONNECTION_WITH_ANOTHER_EVALUATED_GUIDELINE = "connection_with_another_evaluated_guideline"


class GuidelinePayloadOperation(Enum):
    ADD = "add"
    UPDATE = "update"


@dataclass(frozen=True)
class GuidelinePayload:
    content: GuidelineContent
    tool_ids: Sequence[ToolId]
    operation: GuidelinePayloadOperation
    coherence_check: bool  # Legacy and will be removed in the future
    connection_proposition: bool  # Legacy and will be removed in the future
    action_proposition: bool
    properties_proposition: bool
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
class EntailmentRelationshipProposition:
    check_kind: EntailmentRelationshipPropositionKind
    source: GuidelineContent
    target: GuidelineContent


@dataclass(frozen=True)
class InvoiceGuidelineData:
    coherence_checks: Optional[Sequence[CoherenceCheck]]
    entailment_propositions: Optional[Sequence[EntailmentRelationshipProposition]]
    action_proposition: Optional[str]
    properties_proposition: Optional[dict[str, JSONSerializable]]
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
    creation_utc: datetime
    status: EvaluationStatus
    error: Optional[str]
    invoices: Sequence[Invoice]
    progress: float
    tags: Sequence[TagId]


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
        tags: Optional[Sequence[TagId]] = None,
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
    async def upsert_tag(
        self,
        evaluation_id: EvaluationId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> bool: ...

    @abstractmethod
    async def remove_tag(
        self,
        evaluation_id: EvaluationId,
        tag_id: TagId,
    ) -> None: ...


class GuidelineContentDocument(TypedDict):
    condition: str
    action: Optional[str]


class GuidelinePayloadDocument_v0_1_0(TypedDict):
    content: GuidelineContentDocument
    action: Literal["add", "update"]
    updated_id: Optional[GuidelineId]
    coherence_check: bool
    connection_proposition: bool


class GuidelinePayloadDocument(TypedDict):
    content: GuidelineContentDocument
    tool_ids: Sequence[ToolId]
    action: Literal["add", "update"]
    updated_id: Optional[GuidelineId]
    coherence_check: bool
    connection_proposition: bool
    action_proposition: bool
    properties_proposition: bool


_PayloadDocument = Union[GuidelinePayloadDocument]


class _CoherenceCheckDocument(TypedDict):
    kind: str
    first: GuidelineContentDocument
    second: GuidelineContentDocument
    issue: str
    severity: int


class _ConnectionPropositionDocument(TypedDict):
    check_kind: str
    source: GuidelineContentDocument
    target: GuidelineContentDocument


class _InvoiceGuidelineDataDocument_v0_1_0(TypedDict):
    coherence_checks: Optional[Sequence[_CoherenceCheckDocument]]
    connection_propositions: Optional[Sequence[_ConnectionPropositionDocument]]


class InvoiceGuidelineDataDocument(TypedDict):
    coherence_checks: Optional[Sequence[_CoherenceCheckDocument]]
    connection_propositions: Optional[Sequence[_ConnectionPropositionDocument]]
    action_proposition: Optional[str]
    properties_proposition: Optional[dict[str, JSONSerializable]]


_InvoiceDataDocument = Union[InvoiceGuidelineDataDocument]


class InvoiceDocument_v0_1_0(TypedDict, total=False):
    kind: str
    payload: GuidelinePayloadDocument_v0_1_0
    checksum: str
    state_version: str
    approved: bool
    data: Optional[_InvoiceGuidelineDataDocument_v0_1_0]
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
    tag_id: TagId


class EvaluationDocumentStore(EvaluationStore):
    VERSION = Version.from_string("0.2.0")

    def __init__(self, database: DocumentDatabase, allow_migration: bool = False) -> None:
        self._database = database
        self._collection: DocumentCollection[EvaluationDocument]
        self._tag_association_collection: DocumentCollection[EvaluationTagAssociationDocument]

        self._allow_migration = allow_migration
        self._lock = ReaderWriterLock()

    async def tag_association_document_loader(
        self, doc: BaseDocument
    ) -> Optional[EvaluationTagAssociationDocument]:
        if doc["version"] == "0.1.0":
            raise Exception(
                "This code should not be reached! Please run the 'parlant-prepare-migration' script."
            )
        elif doc["version"] == "0.2.0":
            return cast(EvaluationTagAssociationDocument, doc)

        return None

    async def document_loader(self, doc: BaseDocument) -> Optional[EvaluationDocument]:
        if doc["version"] == "0.1.0":
            raise Exception(
                "This code should not be reached! Please run the 'parlant-prepare-migration' script."
            )

        if doc["version"] == "0.2.0":
            return cast(EvaluationDocument, doc)

        return None

    async def __aenter__(self) -> Self:
        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
        ):
            self._collection = await self._database.get_or_create_collection(
                name="evaluations",
                schema=EvaluationDocument,
                document_loader=self.document_loader,
            )

            self._tag_association_collection = await self._database.get_or_create_collection(
                name="evaluation_tag_associations",
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
        def serialize_coherence_check(check: CoherenceCheck) -> _CoherenceCheckDocument:
            return _CoherenceCheckDocument(
                kind=check.kind.value,
                first=GuidelineContentDocument(
                    condition=check.first.condition,
                    action=check.first.action,
                ),
                second=GuidelineContentDocument(
                    condition=check.second.condition,
                    action=check.second.action,
                ),
                issue=check.issue,
                severity=check.severity,
            )

        def serialize_connection_proposition(
            cp: EntailmentRelationshipProposition,
        ) -> _ConnectionPropositionDocument:
            return _ConnectionPropositionDocument(
                check_kind=cp.check_kind.value,
                source=GuidelineContentDocument(
                    condition=cp.source.condition,
                    action=cp.source.action,
                ),
                target=GuidelineContentDocument(
                    condition=cp.target.condition,
                    action=cp.target.action,
                ),
            )

        def serialize_invoice_guideline_data(
            data: InvoiceGuidelineData,
        ) -> InvoiceGuidelineDataDocument:
            return InvoiceGuidelineDataDocument(
                coherence_checks=(
                    [serialize_coherence_check(cc) for cc in data.coherence_checks]
                    if data.coherence_checks
                    else None
                ),
                connection_propositions=(
                    [serialize_connection_proposition(cp) for cp in data.entailment_propositions]
                    if data.entailment_propositions
                    else None
                ),
                action_proposition=(
                    data.action_proposition if data.action_proposition is not None else None
                ),
                properties_proposition=(
                    data.properties_proposition if data.properties_proposition is not None else None
                ),
            )

        def serialize_payload(payload: Payload) -> _PayloadDocument:
            if isinstance(payload, GuidelinePayload):
                return GuidelinePayloadDocument(
                    content=GuidelineContentDocument(
                        condition=payload.content.condition,
                        action=payload.content.action or None,
                    ),
                    tool_ids=payload.tool_ids,
                    action=payload.operation.value,
                    updated_id=payload.updated_id,
                    coherence_check=payload.coherence_check,
                    connection_proposition=payload.connection_proposition,
                    action_proposition=payload.action_proposition,
                    properties_proposition=payload.properties_proposition,
                )
            else:
                raise TypeError(f"Unknown payload type: {type(payload)}")

        kind = invoice.kind.name  # Convert Enum to string
        if kind == "GUIDELINE":
            return InvoiceDocument(
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
        def deserialize_guideline_content_document(
            gc_doc: GuidelineContentDocument,
        ) -> GuidelineContent:
            return GuidelineContent(
                condition=gc_doc["condition"],
                action=gc_doc["action"],
            )

        def deserialize_coherence_check_document(cc_doc: _CoherenceCheckDocument) -> CoherenceCheck:
            return CoherenceCheck(
                kind=CoherenceCheckKind(cc_doc["kind"]),
                first=deserialize_guideline_content_document(cc_doc["first"]),
                second=deserialize_guideline_content_document(cc_doc["second"]),
                issue=cc_doc["issue"],
                severity=cc_doc["severity"],
            )

        def deserialize_connection_proposition_document(
            cp_doc: _ConnectionPropositionDocument,
        ) -> EntailmentRelationshipProposition:
            return EntailmentRelationshipProposition(
                check_kind=EntailmentRelationshipPropositionKind(cp_doc["check_kind"]),
                source=deserialize_guideline_content_document(cp_doc["source"]),
                target=deserialize_guideline_content_document(cp_doc["target"]),
            )

        def deserialize_invoice_guideline_data(
            data_doc: InvoiceGuidelineDataDocument,
        ) -> InvoiceGuidelineData:
            return InvoiceGuidelineData(
                coherence_checks=(
                    [
                        deserialize_coherence_check_document(cc_doc)
                        for cc_doc in data_doc["coherence_checks"]
                    ]
                    if data_doc["coherence_checks"] is not None
                    else None
                ),
                entailment_propositions=(
                    [
                        deserialize_connection_proposition_document(cp_doc)
                        for cp_doc in data_doc["connection_propositions"]
                    ]
                    if data_doc["connection_propositions"] is not None
                    else None
                ),
                action_proposition=(
                    data_doc["action_proposition"]
                    if data_doc["action_proposition"] is not None
                    else None
                ),
                properties_proposition=(
                    data_doc["properties_proposition"]
                    if data_doc["properties_proposition"] is not None
                    else None
                ),
            )

        def deserialize_payload_document(
            kind: PayloadKind,
            payload_doc: _PayloadDocument,
        ) -> Payload:
            if kind == PayloadKind.GUIDELINE:
                return GuidelinePayload(
                    content=GuidelineContent(
                        condition=payload_doc["content"]["condition"],
                        action=payload_doc["content"]["action"] or None,
                    ),
                    tool_ids=payload_doc["tool_ids"],
                    operation=GuidelinePayloadOperation(payload_doc["action"]),
                    updated_id=payload_doc["updated_id"],
                    coherence_check=payload_doc["coherence_check"],
                    connection_proposition=payload_doc["connection_proposition"],
                    action_proposition=payload_doc["action_proposition"],
                    properties_proposition=payload_doc["properties_proposition"],
                )
            else:
                raise ValueError(f"Unsupported payload kind: {kind}")

        def deserialize_invoice_document(invoice_doc: InvoiceDocument) -> Invoice:
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

        async with self._lock.reader_lock:
            tags_docs = await self._tag_association_collection.find(
                filters={"evaluation_id": {"$eq": evaluation_id}},
            )
            tags = [TagId(tag_doc["tag_id"]) for tag_doc in tags_docs]

        return Evaluation(
            id=evaluation_id,
            creation_utc=creation_utc,
            status=status,
            error=evaluation_document.get("error"),
            invoices=invoices,
            progress=evaluation_document["progress"],
            tags=tags,
        )

    @override
    async def create_evaluation(
        self,
        payload_descriptors: Sequence[PayloadDescriptor],
        creation_utc: Optional[datetime] = None,
        extra: Optional[Mapping[str, JSONSerializable]] = None,
        tags: Optional[Sequence[TagId]] = None,
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
                tags=tags or [],
            )

            await self._collection.insert_one(self._serialize_evaluation(evaluation=evaluation))

            for tag in tags or []:
                await self._tag_association_collection.insert_one(
                    document={
                        "id": ObjectId(generate_id()),
                        "version": self.VERSION.to_string(),
                        "creation_utc": creation_utc.isoformat(),
                        "evaluation_id": evaluation_id,
                        "tag_id": tag,
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
    async def upsert_tag(
        self,
        evaluation_id: EvaluationId,
        tag_id: TagId,
        creation_utc: Optional[datetime] = None,
    ) -> bool:
        async with self._lock.writer_lock:
            evaluation = await self.read_evaluation(evaluation_id)

            if tag_id in evaluation.tags:
                return False

            creation_utc = creation_utc or datetime.now(timezone.utc)

            association_document: EvaluationTagAssociationDocument = {
                "id": ObjectId(generate_id()),
                "version": self.VERSION.to_string(),
                "creation_utc": creation_utc.isoformat(),
                "evaluation_id": evaluation_id,
                "tag_id": tag_id,
            }

            _ = await self._tag_association_collection.insert_one(document=association_document)

            evaluation_document = await self._collection.find_one({"id": {"$eq": evaluation_id}})

        if not evaluation_document:
            raise ItemNotFoundError(item_id=UniqueId(evaluation_id))

        return True

    @override
    async def remove_tag(
        self,
        evaluation_id: EvaluationId,
        tag_id: TagId,
    ) -> None:
        async with self._lock.writer_lock:
            delete_result = await self._tag_association_collection.delete_one(
                {
                    "evaluation_id": {"$eq": evaluation_id},
                    "tag_id": {"$eq": tag_id},
                }
            )

            if delete_result.deleted_count == 0:
                raise ItemNotFoundError(item_id=UniqueId(tag_id))

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
