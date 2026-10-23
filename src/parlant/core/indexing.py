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
from parlant.core.rules import RuleId
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

IndexingTaskId = NewType("IndexingTaskId", str)


class IndexingStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()


class PayloadKind(Enum):
    RULE = auto()


class PayloadOperation(Enum):
    ADD = "add"
    UPDATE = "update"


@dataclass(frozen=True)
class RulePayload:
    operation: PayloadOperation
    content: Optional[str] = None
    condition: Optional[str] = None
    relativity_proposition: bool = False
    signal_proposition: bool = False
    title_proposition: bool = False
    title: Optional[str] = None
    agent_id: Optional[AgentId] = None
    updated_id: Optional[RuleId] = None

    def __repr__(self) -> str:
        return f"content: {self.content}, condition: {self.condition}"


GuidelinePayload = RulePayload


Payload: TypeAlias = RulePayload


class PayloadDescriptor(NamedTuple):
    kind: PayloadKind
    payload: Payload


@dataclass(frozen=True)
class InvoiceRuleData:
    references_other_policies_proposition: Optional[bool] = None
    signals_proposition: Optional[Sequence[str]] = None
    anti_signals_proposition: Optional[Sequence[str]] = None
    title_proposition: Optional[str] = None
    _type: Literal["rule"] = "rule"  # Union discriminator for Pydantic


InvoiceData: TypeAlias = InvoiceRuleData


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
class IndexingTask:
    id: IndexingTaskId
    creation_utc: datetime
    status: IndexingStatus
    error: Optional[str]
    invoices: Sequence[Invoice]
    progress: float
    groups: Sequence[GroupId]


class IndexingTaskUpdateParams(TypedDict, total=False):
    status: IndexingStatus
    error: Optional[str]
    invoices: Sequence[Invoice]
    progress: float


class IndexingStore(ABC):
    @abstractmethod
    async def create_indexing_task(
        self,
        payload_descriptors: Sequence[PayloadDescriptor],
        creation_utc: Optional[datetime] = None,
        extra: Optional[Mapping[str, JSONSerializable]] = None,
        groups: Optional[Sequence[GroupId]] = None,
    ) -> IndexingTask: ...

    @abstractmethod
    async def update_indexing_task(
        self,
        indexing_task_id: IndexingTaskId,
        params: IndexingTaskUpdateParams,
    ) -> IndexingTask: ...

    @abstractmethod
    async def read_indexing_task(
        self,
        indexing_task_id: IndexingTaskId,
    ) -> IndexingTask: ...

    @abstractmethod
    async def list_indexing_tasks(
        self,
    ) -> Sequence[IndexingTask]: ...

    @abstractmethod
    async def upsert_group(
        self,
        indexing_task_id: IndexingTaskId,
        group_id: GroupId,
        creation_utc: Optional[datetime] = None,
    ) -> bool: ...

    @abstractmethod
    async def remove_group(
        self,
        indexing_task_id: IndexingTaskId,
        group_id: GroupId,
    ) -> None: ...


class RuleContentDocument(TypedDict, total=False):
    content: Optional[str]
    condition: Optional[str]


class RulePayloadDocument(TypedDict):
    content: RuleContentDocument
    action: Literal["add", "update"]
    updated_id: Optional[RuleId]
    relativity_proposition: bool
    signal_proposition: bool
    title_proposition: bool
    title: Optional[str]
    agent_id: Optional[AgentId]


_PayloadDocument = RulePayloadDocument


class InvoiceRuleDataDocument(TypedDict):
    references_other_policies_proposition: Optional[bool]
    signals_proposition: Optional[Sequence[str]]
    anti_signals_proposition: Optional[Sequence[str]]
    title_proposition: Optional[str]


_InvoiceDataDocument = InvoiceRuleDataDocument


class InvoiceDocument(TypedDict, total=False):
    kind: str
    payload: _PayloadDocument
    checksum: str
    state_version: str
    approved: bool
    data: Optional[_InvoiceDataDocument]
    error: Optional[str]


class IndexingTaskDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    status: str
    error: Optional[str]
    invoices: Sequence[InvoiceDocument]
    progress: float


class IndexingTaskTagAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    indexing_task_id: IndexingTaskId
    group_id: GroupId


class IndexingDocumentStore(IndexingStore):
    VERSION = Version.from_string("0.6.0")

    def __init__(
        self,
        database: DocumentDatabase,
        allow_migration: bool = False,
        collections_prefix: str | None = None,
    ) -> None:
        self._database = database
        self._collection: DocumentCollection[IndexingTaskDocument]
        self._tag_association_collection: DocumentCollection[IndexingTaskTagAssociationDocument]

        self._allow_migration = allow_migration
        self._collections_prefix = collections_prefix
        self._lock = ReaderWriterLock()

    async def tag_association_document_loader(
        self, doc: BaseDocument
    ) -> Optional[IndexingTaskTagAssociationDocument]:
        # Fresh data baseline: no historical versions. Register converters here
        # when the document schema next changes.
        return await DocumentMigrationHelper[IndexingTaskTagAssociationDocument](
            self,
            {},
        ).migrate(doc)

    async def document_loader(self, doc: BaseDocument) -> Optional[IndexingTaskDocument]:
        # Fresh data baseline: no historical versions. Register converters here
        # when the document schema next changes.
        return await DocumentMigrationHelper[IndexingTaskDocument](
            self,
            {},
        ).migrate(doc)

    async def __aenter__(self) -> Self:
        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
            collections_prefix=self._collections_prefix,
        ):
            self._collection = await self._database.get_or_create_collection(
                name=f"{self._collections_prefix}_indexing_tasks"
                if self._collections_prefix
                else "indexing_tasks",
                schema=IndexingTaskDocument,
                document_loader=self.document_loader,
            )

            self._tag_association_collection = await self._database.get_or_create_collection(
                name=f"{self._collections_prefix}_indexing_task_tag_associations"
                if self._collections_prefix
                else "indexing_task_tag_associations",
                schema=IndexingTaskTagAssociationDocument,
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
                references_other_policies_proposition=(
                    data.references_other_policies_proposition
                    if data.references_other_policies_proposition is not None
                    else None
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

        def serialize_payload(payload: Payload) -> _PayloadDocument:
            return RulePayloadDocument(
                content=RuleContentDocument(
                    content=payload.content or None,
                    condition=payload.condition or None,
                ),
                action=payload.operation.value,
                updated_id=payload.updated_id,
                relativity_proposition=payload.relativity_proposition,
                signal_proposition=payload.signal_proposition,
                title_proposition=payload.title_proposition,
                title=payload.title,
                agent_id=payload.agent_id,
            )

        kind = invoice.kind.name  # Convert Enum to string
        if kind != "RULE":
            raise ValueError(f"Unsupported invoice kind: {kind}")

        return InvoiceDocument(
            kind=kind,
            payload=serialize_payload(invoice.payload),
            checksum=invoice.checksum,
            state_version=invoice.state_version,
            approved=invoice.approved,
            data=serialize_invoice_rule_data(invoice.data) if invoice.data else None,
            error=invoice.error,
        )

    def _serialize_indexing_task(self, indexing_task: IndexingTask) -> IndexingTaskDocument:
        return IndexingTaskDocument(
            id=ObjectId(indexing_task.id),
            version=self.VERSION.to_string(),
            creation_utc=indexing_task.creation_utc.isoformat(),
            status=indexing_task.status.name,
            error=indexing_task.error,
            invoices=[self._serialize_invoice(inv) for inv in indexing_task.invoices],
            progress=indexing_task.progress,
        )

    async def _deserialize_indexing_task(
        self, indexing_task_document: IndexingTaskDocument
    ) -> IndexingTask:
        def deserialize_invoice_rule_data(
            data_doc: InvoiceRuleDataDocument,
        ) -> InvoiceRuleData:
            return InvoiceRuleData(
                references_other_policies_proposition=data_doc.get(
                    "references_other_policies_proposition"
                ),
                signals_proposition=data_doc.get("signals_proposition"),
                anti_signals_proposition=data_doc.get("anti_signals_proposition"),
                title_proposition=data_doc.get("title_proposition"),
            )

        def deserialize_payload_document(
            kind: PayloadKind,
            payload_doc: _PayloadDocument,
        ) -> Payload:
            if kind != PayloadKind.RULE:
                raise ValueError(f"Unsupported payload kind: {kind}")

            return RulePayload(
                content=payload_doc["content"].get("content"),
                condition=payload_doc["content"].get("condition"),
                operation=PayloadOperation(payload_doc["action"]),
                updated_id=payload_doc["updated_id"],
                relativity_proposition=payload_doc.get("relativity_proposition", False),
                signal_proposition=payload_doc.get("signal_proposition", False),
                title_proposition=payload_doc.get("title_proposition", False),
                title=payload_doc.get("title"),
                agent_id=payload_doc.get("agent_id"),
            )

        def deserialize_invoice_document(invoice_doc: InvoiceDocument) -> Invoice:
            kind = PayloadKind[invoice_doc["kind"]]

            payload = deserialize_payload_document(kind, invoice_doc["payload"])

            data_doc = invoice_doc.get("data")
            if data_doc is not None:
                data: Optional[InvoiceData] = deserialize_invoice_rule_data(data_doc)
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

        indexing_task_id = IndexingTaskId(indexing_task_document["id"])
        creation_utc = datetime.fromisoformat(indexing_task_document["creation_utc"])

        status = IndexingStatus[indexing_task_document["status"]]

        invoices = [
            deserialize_invoice_document(inv_doc) for inv_doc in indexing_task_document["invoices"]
        ]

        async with self._lock.reader_lock:
            tags_docs = await self._tag_association_collection.find(
                filters={"indexing_task_id": {"$eq": indexing_task_id}},
            )
            groups = [GroupId(tag_doc["group_id"]) for tag_doc in tags_docs]

        return IndexingTask(
            id=indexing_task_id,
            creation_utc=creation_utc,
            status=status,
            error=indexing_task_document.get("error"),
            invoices=invoices,
            progress=indexing_task_document["progress"],
            groups=groups,
        )

    @override
    async def create_indexing_task(
        self,
        payload_descriptors: Sequence[PayloadDescriptor],
        creation_utc: Optional[datetime] = None,
        extra: Optional[Mapping[str, JSONSerializable]] = None,
        groups: Optional[Sequence[GroupId]] = None,
    ) -> IndexingTask:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)

            indexing_task_id = IndexingTaskId(generate_id())

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

            indexing_task = IndexingTask(
                id=indexing_task_id,
                status=IndexingStatus.PENDING,
                creation_utc=creation_utc,
                error=None,
                invoices=invoices,
                progress=0.0,
                groups=groups or [],
            )

            await self._collection.insert_one(
                self._serialize_indexing_task(indexing_task=indexing_task)
            )

            for group in groups or []:
                await self._tag_association_collection.insert_one(
                    document={
                        "id": ObjectId(generate_id()),
                        "version": self.VERSION.to_string(),
                        "creation_utc": creation_utc.isoformat(),
                        "indexing_task_id": indexing_task_id,
                        "group_id": group,
                    }
                )

        return indexing_task

    @override
    async def update_indexing_task(
        self,
        indexing_task_id: IndexingTaskId,
        params: IndexingTaskUpdateParams,
    ) -> IndexingTask:
        async with self._lock.writer_lock:
            indexing_task = await self.read_indexing_task(indexing_task_id)

            update_params: IndexingTaskDocument = {}
            if "invoices" in params:
                update_params["invoices"] = [self._serialize_invoice(i) for i in params["invoices"]]

            if "status" in params:
                update_params["status"] = params["status"].name
                update_params["error"] = params["error"] if "error" in params else None

            if "progress" in params:
                update_params["progress"] = params["progress"]

            result = await self._collection.update_one(
                filters={"id": {"$eq": indexing_task.id}},
                params=update_params,
            )

        assert result.updated_document

        return await self._deserialize_indexing_task(result.updated_document)

    @override
    async def read_indexing_task(
        self,
        indexing_task_id: IndexingTaskId,
    ) -> IndexingTask:
        async with self._lock.reader_lock:
            indexing_task_document = await self._collection.find_one(
                filters={"id": {"$eq": indexing_task_id}},
            )

        if not indexing_task_document:
            raise ItemNotFoundError(item_id=UniqueId(indexing_task_id))

        return await self._deserialize_indexing_task(indexing_task_document=indexing_task_document)

    @override
    async def list_indexing_tasks(
        self,
    ) -> Sequence[IndexingTask]:
        async with self._lock.reader_lock:
            return [
                await self._deserialize_indexing_task(indexing_task_document=e)
                for e in await self._collection.find(filters={})
            ]

    @override
    async def upsert_group(
        self,
        indexing_task_id: IndexingTaskId,
        group_id: GroupId,
        creation_utc: Optional[datetime] = None,
    ) -> bool:
        async with self._lock.writer_lock:
            indexing_task = await self.read_indexing_task(indexing_task_id)

            if group_id in indexing_task.groups:
                return False

            creation_utc = creation_utc or datetime.now(timezone.utc)

            association_document: IndexingTaskTagAssociationDocument = {
                "id": ObjectId(generate_id()),
                "version": self.VERSION.to_string(),
                "creation_utc": creation_utc.isoformat(),
                "indexing_task_id": indexing_task_id,
                "group_id": group_id,
            }

            _ = await self._tag_association_collection.insert_one(document=association_document)

            indexing_task_document = await self._collection.find_one(
                {"id": {"$eq": indexing_task_id}}
            )

        if not indexing_task_document:
            raise ItemNotFoundError(item_id=UniqueId(indexing_task_id))

        return True

    @override
    async def remove_group(
        self,
        indexing_task_id: IndexingTaskId,
        group_id: GroupId,
    ) -> None:
        async with self._lock.writer_lock:
            delete_result = await self._tag_association_collection.delete_one(
                {
                    "indexing_task_id": {"$eq": indexing_task_id},
                    "group_id": {"$eq": group_id},
                }
            )

            if delete_result.deleted_count == 0:
                raise ItemNotFoundError(item_id=UniqueId(group_id))

            indexing_task_document = await self._collection.find_one(
                {"id": {"$eq": indexing_task_id}}
            )

        if not indexing_task_document:
            raise ItemNotFoundError(item_id=UniqueId(indexing_task_id))


class IndexingListener(ABC):
    @abstractmethod
    async def wait_for_completion(
        self,
        indexing_task_id: IndexingTaskId,
        timeout: Timeout = Timeout.infinite(),
    ) -> bool: ...


class PollingIndexingListener(IndexingListener):
    def __init__(self, indexing_store: IndexingStore) -> None:
        self._indexing_store = indexing_store

    @override
    async def wait_for_completion(
        self,
        indexing_task_id: IndexingTaskId,
        timeout: Timeout = Timeout.infinite(),
    ) -> bool:
        while True:
            indexing_task = await self._indexing_store.read_indexing_task(
                indexing_task_id,
            )

            if indexing_task.status in [IndexingStatus.COMPLETED, IndexingStatus.FAILED]:
                return True
            elif timeout.expired():
                return False
            else:
                await timeout.wait_up_to(1)


class CompositeIndexingStore(IndexingStore):
    def __init__(
        self,
        writable_store: IndexingStore,
        readable_stores: Sequence[IndexingStore],
    ) -> None:
        self._writable_store = writable_store
        self._readable_stores = readable_stores
        self._all_stores: Sequence[IndexingStore] = [writable_store, *readable_stores]

    @override
    async def create_indexing_task(
        self,
        payload_descriptors: Sequence[PayloadDescriptor],
        creation_utc: Optional[datetime] = None,
        extra: Optional[Mapping[str, JSONSerializable]] = None,
        groups: Optional[Sequence[GroupId]] = None,
    ) -> IndexingTask:
        return await self._writable_store.create_indexing_task(
            payload_descriptors=payload_descriptors,
            creation_utc=creation_utc,
            extra=extra,
            groups=groups,
        )

    @override
    async def update_indexing_task(
        self,
        indexing_task_id: IndexingTaskId,
        params: IndexingTaskUpdateParams,
    ) -> IndexingTask:
        return await self._writable_store.update_indexing_task(indexing_task_id, params)

    @override
    async def read_indexing_task(
        self,
        indexing_task_id: IndexingTaskId,
    ) -> IndexingTask:
        results = await safe_gather(
            *[try_or_none(store.read_indexing_task(indexing_task_id)) for store in self._all_stores]
        )
        result = next((r for r in results if r is not None), None)
        if result is None:
            raise ItemNotFoundError(item_id=UniqueId(indexing_task_id))
        return result

    @override
    async def list_indexing_tasks(
        self,
    ) -> Sequence[IndexingTask]:
        results = await safe_gather(*[store.list_indexing_tasks() for store in self._all_stores])
        return list(chain.from_iterable(results))

    @override
    async def upsert_group(
        self,
        indexing_task_id: IndexingTaskId,
        group_id: GroupId,
        creation_utc: Optional[datetime] = None,
    ) -> bool:
        return await self._writable_store.upsert_group(indexing_task_id, group_id, creation_utc)

    @override
    async def remove_group(
        self,
        indexing_task_id: IndexingTaskId,
        group_id: GroupId,
    ) -> None:
        return await self._writable_store.remove_group(indexing_task_id, group_id)
