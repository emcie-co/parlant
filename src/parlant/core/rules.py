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

from itertools import chain
from typing import Mapping, NewType, Optional, Sequence, Set, cast
from typing_extensions import override, TypedDict, Self, Required
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone

from parlant.core import async_utils
from parlant.core.agents import Effort
from parlant.core.async_utils import ReaderWriterLock, safe_gather
from parlant.core.common import (
    Weight,
    ItemNotFoundError,
    try_or_none,
    JSONSerializable,
    UniqueId,
    Version,
    IdGenerator,
    xxh3_checksum,
)
from parlant.core.nlp.embedding import Embedder, EmbedderProvider
from parlant.core.persistence.common import ObjectId, Where
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentDatabase,
    DocumentCollection,
    group_ids_by_entity,
)
from parlant.core.persistence.document_database_helper import (
    DocumentStoreMigrationHelper,
    DocumentMigrationHelper,
)
from parlant.core.persistence.vector_database import (
    SimilarDocumentResult,
    VectorCollection,
    VectorCollectionIndex,
    VectorDatabase,
    BaseDocument as VectorDocument,
)
from parlant.core.persistence.vector_database_helper import (
    VectorDocumentStoreMigrationHelper,
    calculate_min_vectors_for_max_item_count,
    query_chunks,
)
from parlant.core.groups import GroupId

RuleId = NewType("RuleId", str)


@dataclass(frozen=True)
class Rule:
    id: RuleId
    creation_utc: datetime
    modified_utc: datetime
    enabled: bool
    groups: Sequence[GroupId]
    metadata: Mapping[str, JSONSerializable]
    weight: Weight
    content: Optional[str] = None
    condition: Optional[str] = None
    title: Optional[str] = None
    labels: Set[str] = field(default_factory=set)
    effort: Optional[Effort] = None
    priority: int = 0
    signals: Sequence[str] = field(default_factory=list)
    anti_signals: Sequence[str] = field(default_factory=list)
    # Declarative review requests: when this rule matches a turn, ask the
    # engine to review the message / the tool calls before committing them.
    message_review: bool = False
    tool_review: bool = False

    def __post_init__(self) -> None:
        if not (self.content or self.condition):
            raise ValueError("A rule requires content or a condition")

    def __str__(self) -> str:
        parts: list[str] = []
        if self.condition:
            parts.append(f"Observation: {self.condition}")
        if self.content:
            parts.append(self.content)
        return "\n\n".join(parts)

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def query(self) -> str:
        """The canonical "what does this rule talk about" text — used to find
        the glossary terms the rule depends on (at matching time and during
        indexing)."""
        return compose_rule_query(
            title=self.title,
            condition=self.condition,
            content=self.content,
        )


def compose_rule_query(
    title: Optional[str],
    condition: Optional[str],
    content: Optional[str],
) -> str:
    """Concatenate a rule's definition into a relevance-search query,
    skipping absent parts. Shared by :attr:`Rule.query` and indexing
    flows that hold the parts separately."""
    title = (title or "").strip()
    condition = (condition or "").strip()
    content = (content or "").strip()

    lines: list[str] = []

    if title:
        lines.append(title)

    if condition:
        lines.append(condition)

    if content:
        lines.append(content)

    return "\n\n".join(lines)


class RuleUpdateParams(TypedDict, total=False):
    content: Optional[str]
    condition: Optional[str]
    title: Optional[str]
    weight: Weight
    enabled: bool
    metadata: Mapping[str, JSONSerializable]
    effort: Optional[Effort]
    priority: int
    signals: Sequence[str]
    anti_signals: Sequence[str]
    message_review: bool
    tool_review: bool


@dataclass(frozen=True)
class RuleRelevanceResult:
    rule: Rule
    score: float


class RuleStore(ABC):
    @abstractmethod
    async def create_rule(
        self,
        content: Optional[str] = None,
        condition: Optional[str] = None,
        title: Optional[str] = None,
        weight: Optional[Weight] = None,
        metadata: Mapping[str, JSONSerializable] = {},
        creation_utc: Optional[datetime] = None,
        enabled: bool = True,
        groups: Optional[Sequence[GroupId]] = None,
        id: Optional[RuleId] = None,
        effort: Optional[Effort] = None,
        labels: Optional[Set[str]] = None,
        priority: int = 0,
        signals: Sequence[str] = [],
        anti_signals: Sequence[str] = [],
        message_review: bool = False,
        tool_review: bool = False,
    ) -> Rule: ...

    @abstractmethod
    async def list_rules(
        self,
        groups: Optional[Sequence[GroupId]] = None,
        labels: Optional[Set[str]] = None,
    ) -> Sequence[Rule]: ...

    @abstractmethod
    async def find_relevant_rules(
        self,
        query: str,
        available_rules: Sequence[Rule],
        max_count: int,
    ) -> Sequence[RuleRelevanceResult]: ...

    @abstractmethod
    async def read_rule(
        self,
        rule_id: RuleId,
    ) -> Rule: ...

    @abstractmethod
    async def delete_rule(
        self,
        rule_id: RuleId,
    ) -> None: ...

    @abstractmethod
    async def update_rule(
        self,
        rule_id: RuleId,
        params: RuleUpdateParams,
    ) -> Rule: ...

    @abstractmethod
    async def find_rule(
        self,
        content: Optional[str] = None,
        condition: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Rule: ...

    @abstractmethod
    async def upsert_group(
        self,
        rule_id: RuleId,
        group_id: GroupId,
        creation_utc: Optional[datetime] = None,
    ) -> bool: ...

    @abstractmethod
    async def remove_group(
        self,
        rule_id: RuleId,
        group_id: GroupId,
    ) -> None: ...

    @abstractmethod
    async def set_metadata(
        self,
        rule_id: RuleId,
        key: str,
        value: JSONSerializable,
    ) -> Rule: ...

    @abstractmethod
    async def unset_metadata(
        self,
        rule_id: RuleId,
        key: str,
    ) -> Rule: ...

    @abstractmethod
    async def upsert_labels(
        self,
        rule_id: RuleId,
        labels: Set[str],
    ) -> Rule: ...

    @abstractmethod
    async def remove_labels(
        self,
        rule_id: RuleId,
        labels: Set[str],
    ) -> Rule: ...


class RuleDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    last_modified: str
    content: Optional[str]
    condition: Optional[str]
    title: Optional[str]
    weight: str
    enabled: bool
    metadata: Mapping[str, JSONSerializable]
    effort: Optional[str]
    labels: Sequence[str]
    priority: int
    signals: Sequence[str]
    anti_signals: Sequence[str]
    message_review: bool
    tool_review: bool


class RuleVectorDocument(TypedDict, total=False):
    id: ObjectId
    rule_id: ObjectId
    version: Version.String
    content: str
    checksum: Required[str]


class RuleTagAssociationDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    rule_id: RuleId
    group_id: GroupId


class RuleVectorStore(RuleStore):
    VERSION = Version.from_string("0.16.0")

    def __init__(
        self,
        id_generator: IdGenerator,
        vector_db: VectorDatabase,
        document_db: DocumentDatabase,
        embedder_provider: EmbedderProvider,
        allow_migration: bool = True,
        collections_prefix: str | None = None,
    ) -> None:
        self._id_generator = id_generator

        self._vector_db = vector_db
        self._database = document_db
        self._vector_collection: VectorCollection[RuleVectorDocument]
        self._collection: DocumentCollection[RuleDocument]
        self._tag_association_collection: DocumentCollection[RuleTagAssociationDocument]

        self._allow_migration = allow_migration
        self._collections_prefix = collections_prefix
        self._lock = ReaderWriterLock()
        self._embedder_provider = embedder_provider
        self._embedder: Embedder

    async def _vector_document_loader(self, doc: VectorDocument) -> Optional[RuleVectorDocument]:
        return cast(RuleVectorDocument, doc)

    async def _document_loader(self, doc: BaseDocument) -> Optional[RuleDocument]:
        # Fresh data baseline: no historical versions. Register converters here
        # when the document schema next changes.
        return await DocumentMigrationHelper[RuleDocument](
            self,
            {},
        ).migrate(doc)

    async def _association_document_loader(
        self, doc: BaseDocument
    ) -> Optional[RuleTagAssociationDocument]:
        # Fresh data baseline: no historical versions. Register converters here
        # when the document schema next changes.
        return await DocumentMigrationHelper[RuleTagAssociationDocument](
            self,
            {},
        ).migrate(doc)

    async def __aenter__(self) -> Self:
        self._embedder = await self._embedder_provider()

        async with VectorDocumentStoreMigrationHelper(
            store=self,
            database=self._vector_db,
            allow_migration=self._allow_migration,
        ):
            self._vector_collection = await self._vector_db.get_or_create_collection(
                name=f"{self._collections_prefix}_rules" if self._collections_prefix else "rules",
                schema=RuleVectorDocument,
                embedder=self._embedder,
                document_loader=self._vector_document_loader,
            )
            await self._vector_collection.ensure_indexes([VectorCollectionIndex(field="rule_id")])

        async with DocumentStoreMigrationHelper(
            store=self,
            database=self._database,
            allow_migration=self._allow_migration,
            collections_prefix=self._collections_prefix,
        ):
            self._collection = await self._database.get_or_create_collection(
                name=f"{self._collections_prefix}_rules" if self._collections_prefix else "rules",
                schema=RuleDocument,
                document_loader=self._document_loader,
            )

            self._tag_association_collection = await self._database.get_or_create_collection(
                name=f"{self._collections_prefix}_rule_group_associations"
                if self._collections_prefix
                else "rule_group_associations",
                schema=RuleTagAssociationDocument,
                document_loader=self._association_document_loader,
            )

        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None:
        pass

    def _serialize(
        self,
        rule: Rule,
    ) -> RuleDocument:
        return RuleDocument(
            id=ObjectId(rule.id),
            version=self.VERSION.to_string(),
            creation_utc=rule.creation_utc.isoformat(),
            last_modified=rule.modified_utc.isoformat(),
            content=rule.content,
            condition=rule.condition,
            title=rule.title,
            weight=rule.weight.value,
            enabled=rule.enabled,
            metadata=rule.metadata,
            effort=rule.effort.value if rule.effort else None,
            labels=list(rule.labels),
            priority=rule.priority,
            signals=list(rule.signals),
            anti_signals=list(rule.anti_signals),
            message_review=rule.message_review,
            tool_review=rule.tool_review,
        )

    async def _deserialize(
        self,
        rule_document: RuleDocument,
    ) -> Rule:
        group_ids = [
            d["group_id"]
            for d in await self._tag_association_collection.find(
                {"rule_id": {"$eq": rule_document["id"]}}
            )
        ]

        return self._to_rule(rule_document, group_ids)

    async def _deserialize_many(
        self,
        rule_documents: Sequence[RuleDocument],
    ) -> Sequence[Rule]:
        """Reconstruct a whole page of rules with one association query.

        Deserializing one at a time costs a query per rule, which the matcher
        pays on every turn.
        """
        group_ids = await group_ids_by_entity(
            self._tag_association_collection,
            entity_field="rule_id",
            value_field="group_id",
            entity_ids=[d["id"] for d in rule_documents],
        )

        return [self._to_rule(d, group_ids[d["id"]]) for d in rule_documents]

    def _to_rule(
        self,
        rule_document: RuleDocument,
        group_ids: Sequence[str],
    ) -> Rule:
        effort_str = rule_document.get("effort")
        effort = Effort(effort_str) if effort_str else None

        return Rule(
            id=RuleId(rule_document["id"]),
            creation_utc=datetime.fromisoformat(rule_document["creation_utc"]),
            modified_utc=datetime.fromisoformat(rule_document["last_modified"]),
            content=rule_document.get("content", None),
            condition=rule_document.get("condition", None),
            title=rule_document.get("title", None),
            weight=Weight(rule_document["weight"]),
            enabled=rule_document["enabled"],
            groups=[GroupId(group_id) for group_id in group_ids],
            metadata=rule_document["metadata"],
            labels=set(rule_document.get("labels", [])),
            effort=effort,
            priority=rule_document.get("priority", 0),
            signals=list(rule_document.get("signals", [])),
            anti_signals=list(rule_document.get("anti_signals", [])),
            message_review=rule_document.get("message_review", False),
            tool_review=rule_document.get("tool_review", False),
        )

    def _rule_embedding_content(self, rule: Rule) -> str:
        """Render a rule's authored text as the string to embed.

        Treats ``None`` / empty / whitespace-only fields as absent: the title
        becomes a heading, a condition gets a labeled form, and the
        content rides as its own block.
        """
        title = (rule.title or "").strip()
        condition = (rule.condition or "").strip()
        content = (rule.content or "").strip()

        parts: list[str] = []

        if title:
            parts.append(f"# {title}")

        if condition:
            parts.append(f"Condition: {condition}")

        if content:
            parts.append(content)

        if not parts:
            raise ValueError("Rule has no content")

        return "\n\n".join(parts)

    def _list_rule_contents(self, rule: Rule) -> list[str]:
        """The independent strings to embed for a rule: its rendered
        content followed by each signal as its own vector."""
        return [self._rule_embedding_content(rule), *rule.signals]

    def _build_vector_documents(self, rule: Rule) -> list[RuleVectorDocument]:
        documents = []

        for content in self._list_rule_contents(rule):
            doc_id = self._id_generator.generate(xxh3_checksum(f"{rule.id}{content}"))

            vec_doc = RuleVectorDocument(
                id=ObjectId(doc_id),
                rule_id=ObjectId(rule.id),
                version=self.VERSION.to_string(),
                content=content,
                checksum=xxh3_checksum(content),
            )

            documents.append(vec_doc)

        return documents

    async def _insert_vector_documents(self, rule: Rule) -> None:
        await async_utils.safe_gather(
            *[
                self._vector_collection.insert_one(document=document)
                for document in self._build_vector_documents(rule)
            ]
        )

    async def _delete_vector_documents(self, rule_id: RuleId) -> None:
        for v_doc in await self._vector_collection.find(filters={"rule_id": {"$eq": rule_id}}):
            await self._vector_collection.delete_one(filters={"id": {"$eq": v_doc["id"]}})

    @override
    async def create_rule(
        self,
        content: Optional[str] = None,
        condition: Optional[str] = None,
        title: Optional[str] = None,
        weight: Optional[Weight] = None,
        metadata: Mapping[str, JSONSerializable] = {},
        creation_utc: Optional[datetime] = None,
        enabled: bool = True,
        groups: Optional[Sequence[GroupId]] = None,
        id: Optional[RuleId] = None,
        effort: Optional[Effort] = None,
        labels: Optional[Set[str]] = None,
        priority: int = 0,
        signals: Sequence[str] = [],
        anti_signals: Sequence[str] = [],
        message_review: bool = False,
        tool_review: bool = False,
    ) -> Rule:
        async with self._lock.writer_lock:
            creation_utc = creation_utc or datetime.now(timezone.utc)
            weight = weight or Weight.MEDIUM

            # Use provided ID or generate one
            if id is not None:
                rule_id = id

                # Check if rule with this ID already exists
                existing = await self._collection.find_one(filters={"id": {"$eq": rule_id}})
                if existing:
                    raise ValueError(f"Rule with id '{rule_id}' already exists")
            else:
                rule_checksum = xxh3_checksum(
                    f"{content or ''}{condition or ''}{title or ''}{enabled}{metadata}"
                )
                rule_id = RuleId(self._id_generator.generate(rule_checksum))

            rule = Rule(
                id=rule_id,
                creation_utc=creation_utc,
                modified_utc=creation_utc,
                content=content,
                condition=condition,
                title=title,
                weight=weight,
                enabled=enabled,
                groups=groups or [],
                metadata=metadata,
                labels=labels or set(),
                effort=effort,
                priority=priority,
                signals=list(signals),
                anti_signals=list(anti_signals),
                message_review=message_review,
                tool_review=tool_review,
            )

            await self._insert_vector_documents(rule)

            await self._collection.insert_one(
                document=self._serialize(
                    rule=rule,
                )
            )

            for group_id in groups or []:
                tag_checksum = xxh3_checksum(f"{rule.id}{group_id}")

                await self._tag_association_collection.insert_one(
                    document={
                        "id": ObjectId(self._id_generator.generate(tag_checksum)),
                        "version": self.VERSION.to_string(),
                        "creation_utc": creation_utc.isoformat(),
                        "rule_id": rule.id,
                        "group_id": group_id,
                    }
                )

        return rule

    @override
    async def list_rules(
        self,
        groups: Optional[Sequence[GroupId]] = None,
        labels: Optional[Set[str]] = None,
    ) -> Sequence[Rule]:
        filters: Where = {}

        async with self._lock.reader_lock:
            if groups is not None:
                if len(groups) == 0:
                    rule_ids = {
                        doc["rule_id"]
                        for doc in await self._tag_association_collection.find(filters={})
                    }

                    filters = {"$and": [{"id": {"$ne": id}} for id in rule_ids]} if rule_ids else {}
                else:
                    tag_filters: Where = {"$or": [{"group_id": {"$eq": group}} for group in groups]}
                    tag_associations = await self._tag_association_collection.find(
                        filters=tag_filters
                    )
                    rule_ids = {assoc["rule_id"] for assoc in tag_associations}

                    if not rule_ids:
                        return []

                    filters = {"$or": [{"id": {"$eq": id}} for id in rule_ids]}

            rules = list(
                await self._deserialize_many(list(await self._collection.find(filters=filters)))
            )

            # Filter by labels if specified
            if labels is not None:
                rules = [g for g in rules if labels.issubset(g.labels)]

            return rules

    @override
    async def read_rule(
        self,
        rule_id: RuleId,
    ) -> Rule:
        async with self._lock.reader_lock:
            rule_document = await self._collection.find_one(
                filters={
                    "id": {"$eq": rule_id},
                }
            )

        if not rule_document:
            raise ItemNotFoundError(item_id=UniqueId(rule_id))

        return await self._deserialize(rule_document=rule_document)

    @override
    async def delete_rule(
        self,
        rule_id: RuleId,
    ) -> None:
        async with self._lock.writer_lock:
            await self._delete_vector_documents(rule_id)

            result = await self._collection.delete_one(
                filters={
                    "id": {"$eq": rule_id},
                }
            )

            for doc in await self._tag_association_collection.find(
                filters={
                    "rule_id": {"$eq": rule_id},
                }
            ):
                await self._tag_association_collection.delete_one(
                    filters={"id": {"$eq": doc["id"]}}
                )

        if not result.deleted_document:
            raise ItemNotFoundError(item_id=UniqueId(rule_id))

    @override
    async def update_rule(
        self,
        rule_id: RuleId,
        params: RuleUpdateParams,
    ) -> Rule:
        async with self._lock.writer_lock:
            rule_document = RuleDocument(
                {
                    **({"content": params["content"]} if "content" in params else {}),
                    **({"condition": params["condition"]} if "condition" in params else {}),
                    **({"title": params["title"]} if "title" in params else {}),
                    **({"weight": params["weight"].value} if "weight" in params else {}),
                    **({"enabled": params["enabled"]} if "enabled" in params else {}),
                    **(
                        {
                            "effort": (
                                # Note that updating to None is also valid
                                params["effort"].value if params["effort"] is not None else None
                            )
                        }
                        if "effort" in params
                        else {}
                    ),
                    **({"priority": params["priority"]} if "priority" in params else {}),
                    **({"signals": params["signals"]} if "signals" in params else {}),
                    **(
                        {"anti_signals": params["anti_signals"]} if "anti_signals" in params else {}
                    ),
                    **(
                        {"message_review": params["message_review"]}
                        if "message_review" in params
                        else {}
                    ),
                    **({"tool_review": params["tool_review"]} if "tool_review" in params else {}),
                    "last_modified": datetime.now(timezone.utc).isoformat(),
                }
            )

            result = await self._collection.update_one(
                filters={"id": {"$eq": rule_id}},
                params=rule_document,
            )

            assert result.updated_document

            updated = await self._deserialize(rule_document=result.updated_document)

            # Re-sync the embedded vectors if any embedded field changed
            # (content / condition / title / signals).
            if any(key in params for key in ("content", "condition", "title", "signals")):
                await self._vector_collection.replace_documents(
                    filters={"rule_id": {"$eq": rule_id}},
                    documents=self._build_vector_documents(updated),
                )

        return updated

    @override
    async def find_rule(
        self,
        content: Optional[str] = None,
        condition: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Rule:
        """Finds a rule by its authored content: content, condition, and
        title must all match (absent fields match rules without them)."""
        async with self._lock.reader_lock:
            filters = {
                "content": {"$eq": content},
                "condition": {"$eq": condition},
                "title": {"$eq": title},
            }

            rule_document = await self._collection.find_one(filters=cast(Where, filters))

        if not rule_document:
            raise ItemNotFoundError(
                item_id=UniqueId(f"{title or ''}{content or ''}{condition or ''}")
            )

        return await self._deserialize(rule_document=rule_document)

    @override
    async def find_relevant_rules(
        self,
        query: str,
        available_rules: Sequence[Rule],
        max_count: int,
    ) -> Sequence[RuleRelevanceResult]:
        if not available_rules:
            return []

        rules_by_id = {g.id: g for g in available_rules}

        async with self._lock.reader_lock:
            queries = await query_chunks(query, self._embedder)
            filters: Where = {"rule_id": {"$in": [str(g.id) for g in available_rules]}}

            tasks = [
                self._vector_collection.find_similar_documents(
                    filters=filters,
                    query=q,
                    k=calculate_min_vectors_for_max_item_count(
                        items=available_rules,
                        count_item_vectors=lambda g: len(self._list_rule_contents(g)),
                        max_items_to_return=max_count,
                    ),
                    hints={"group": "rules"},
                )
                for q in queries
            ]

        all_sdocs = chain.from_iterable(await async_utils.safe_gather(*tasks))

        # Dedupe by rule, keeping the closest matching vector per rule.
        unique_sdocs: dict[str, SimilarDocumentResult[RuleVectorDocument]] = {}

        for similar_doc in all_sdocs:
            rule_id = similar_doc.document["rule_id"]
            if rule_id not in unique_sdocs or unique_sdocs[rule_id].distance > similar_doc.distance:
                unique_sdocs[rule_id] = similar_doc

        top_results = sorted(unique_sdocs.values(), key=lambda r: r.distance)[:max_count]

        return [
            RuleRelevanceResult(
                rule=rules_by_id[RuleId(r.document["rule_id"])],
                score=1.0 - r.distance,
            )
            for r in top_results
            if RuleId(r.document["rule_id"]) in rules_by_id
        ]

    @override
    async def upsert_group(
        self,
        rule_id: RuleId,
        group_id: GroupId,
        creation_utc: Optional[datetime] = None,
    ) -> bool:
        async with self._lock.writer_lock:
            rule = await self.read_rule(rule_id)

            if group_id in rule.groups:
                return False

            creation_utc = creation_utc or datetime.now(timezone.utc)

            association_checksum = xxh3_checksum(f"{rule.id}{group_id}")

            association_document: RuleTagAssociationDocument = {
                "id": ObjectId(self._id_generator.generate(association_checksum)),
                "version": self.VERSION.to_string(),
                "creation_utc": creation_utc.isoformat(),
                "rule_id": RuleId(rule_id),
                "group_id": group_id,
            }

            _ = await self._tag_association_collection.insert_one(document=association_document)

            rule_document = await self._collection.find_one({"id": {"$eq": rule_id}})

        if not rule_document:
            raise ItemNotFoundError(item_id=UniqueId(rule_id))

        return True

    @override
    async def remove_group(
        self,
        rule_id: RuleId,
        group_id: GroupId,
    ) -> None:
        async with self._lock.writer_lock:
            delete_result = await self._tag_association_collection.delete_one(
                {
                    "rule_id": {"$eq": rule_id},
                    "group_id": {"$eq": group_id},
                }
            )

            if delete_result.deleted_count == 0:
                raise ItemNotFoundError(item_id=UniqueId(group_id))

            rule_document = await self._collection.find_one({"id": {"$eq": rule_id}})

        if not rule_document:
            raise ItemNotFoundError(item_id=UniqueId(rule_id))

    @override
    async def set_metadata(
        self,
        rule_id: RuleId,
        key: str,
        value: JSONSerializable,
    ) -> Rule:
        async with self._lock.writer_lock:
            rule_document = await self._collection.find_one({"id": {"$eq": rule_id}})

            if not rule_document:
                raise ItemNotFoundError(item_id=UniqueId(rule_id))

            updated_metadata = {**rule_document["metadata"], key: value}

            result = await self._collection.update_one(
                filters={"id": {"$eq": rule_id}},
                params={
                    "metadata": updated_metadata,
                    "last_modified": datetime.now(timezone.utc).isoformat(),
                },
            )

        assert result.updated_document

        return await self._deserialize(rule_document=result.updated_document)

    @override
    async def unset_metadata(
        self,
        rule_id: RuleId,
        key: str,
    ) -> Rule:
        async with self._lock.writer_lock:
            rule_document = await self._collection.find_one({"id": {"$eq": rule_id}})

            if not rule_document:
                raise ItemNotFoundError(item_id=UniqueId(rule_id))

            updated_metadata = {k: v for k, v in rule_document["metadata"].items() if k != key}

            result = await self._collection.update_one(
                filters={"id": {"$eq": rule_id}},
                params={
                    "metadata": updated_metadata,
                    "last_modified": datetime.now(timezone.utc).isoformat(),
                },
            )

        assert result.updated_document

        return await self._deserialize(rule_document=result.updated_document)

    @override
    async def upsert_labels(
        self,
        rule_id: RuleId,
        labels: Set[str],
    ) -> Rule:
        async with self._lock.writer_lock:
            rule_document = await self._collection.find_one({"id": {"$eq": rule_id}})

            if not rule_document:
                raise ItemNotFoundError(item_id=UniqueId(rule_id))

            current_labels = set(rule_document.get("labels", []))
            updated_labels = list(current_labels | labels)

            result = await self._collection.update_one(
                filters={"id": {"$eq": rule_id}},
                params={
                    "labels": updated_labels,
                },
            )

        assert result.updated_document

        return await self._deserialize(rule_document=result.updated_document)

    @override
    async def remove_labels(
        self,
        rule_id: RuleId,
        labels: Set[str],
    ) -> Rule:
        async with self._lock.writer_lock:
            rule_document = await self._collection.find_one({"id": {"$eq": rule_id}})

            if not rule_document:
                raise ItemNotFoundError(item_id=UniqueId(rule_id))

            current_labels = set(rule_document.get("labels", []))
            updated_labels = list(current_labels - labels)

            result = await self._collection.update_one(
                filters={"id": {"$eq": rule_id}},
                params={
                    "labels": updated_labels,
                },
            )

        assert result.updated_document

        return await self._deserialize(rule_document=result.updated_document)


class CompositeRuleStore(RuleStore):
    def __init__(
        self,
        writable_store: RuleStore,
        readable_stores: Sequence[RuleStore],
    ) -> None:
        self._writable_store = writable_store
        self._readable_stores = readable_stores
        self._all_stores: Sequence[RuleStore] = [writable_store, *readable_stores]

    @override
    async def create_rule(
        self,
        content: Optional[str] = None,
        condition: Optional[str] = None,
        title: Optional[str] = None,
        weight: Optional[Weight] = None,
        metadata: Mapping[str, JSONSerializable] = {},
        creation_utc: Optional[datetime] = None,
        enabled: bool = True,
        groups: Optional[Sequence[GroupId]] = None,
        id: Optional[RuleId] = None,
        effort: Optional[Effort] = None,
        labels: Optional[Set[str]] = None,
        priority: int = 0,
        signals: Sequence[str] = [],
        anti_signals: Sequence[str] = [],
        message_review: bool = False,
        tool_review: bool = False,
    ) -> Rule:
        return await self._writable_store.create_rule(
            content=content,
            condition=condition,
            title=title,
            weight=weight,
            metadata=metadata,
            creation_utc=creation_utc,
            enabled=enabled,
            groups=groups,
            id=id,
            effort=effort,
            labels=labels,
            priority=priority,
            signals=signals,
            anti_signals=anti_signals,
            message_review=message_review,
            tool_review=tool_review,
        )

    @override
    async def list_rules(
        self,
        groups: Optional[Sequence[GroupId]] = None,
        labels: Optional[Set[str]] = None,
    ) -> Sequence[Rule]:
        results = await safe_gather(
            *[store.list_rules(groups=groups, labels=labels) for store in self._all_stores]
        )
        return list(chain.from_iterable(results))

    @override
    async def find_relevant_rules(
        self,
        query: str,
        available_rules: Sequence[Rule],
        max_count: int,
    ) -> Sequence[RuleRelevanceResult]:
        results = await safe_gather(
            *[
                store.find_relevant_rules(query, available_rules, max_count)
                for store in self._all_stores
            ]
        )
        merged = list(chain.from_iterable(results))
        return sorted(merged, key=lambda r: r.score, reverse=True)[:max_count]

    @override
    async def read_rule(
        self,
        rule_id: RuleId,
    ) -> Rule:
        results = await safe_gather(
            *[try_or_none(store.read_rule(rule_id)) for store in self._all_stores]
        )
        result = next((r for r in results if r is not None), None)
        if result is None:
            raise ItemNotFoundError(item_id=UniqueId(rule_id))
        return result

    @override
    async def delete_rule(
        self,
        rule_id: RuleId,
    ) -> None:
        return await self._writable_store.delete_rule(rule_id)

    @override
    async def update_rule(
        self,
        rule_id: RuleId,
        params: RuleUpdateParams,
    ) -> Rule:
        return await self._writable_store.update_rule(rule_id, params)

    @override
    async def find_rule(
        self,
        content: Optional[str] = None,
        condition: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Rule:
        results = await safe_gather(
            *[try_or_none(store.find_rule(content, condition, title)) for store in self._all_stores]
        )
        result = next((r for r in results if r is not None), None)
        if result is None:
            raise ItemNotFoundError(
                item_id=UniqueId(f"{title or ''}{content or ''}{condition or ''}")
            )
        return result

    @override
    async def upsert_group(
        self,
        rule_id: RuleId,
        group_id: GroupId,
        creation_utc: Optional[datetime] = None,
    ) -> bool:
        return await self._writable_store.upsert_group(rule_id, group_id, creation_utc)

    @override
    async def remove_group(
        self,
        rule_id: RuleId,
        group_id: GroupId,
    ) -> None:
        return await self._writable_store.remove_group(rule_id, group_id)

    @override
    async def set_metadata(
        self,
        rule_id: RuleId,
        key: str,
        value: JSONSerializable,
    ) -> Rule:
        return await self._writable_store.set_metadata(rule_id, key, value)

    @override
    async def unset_metadata(
        self,
        rule_id: RuleId,
        key: str,
    ) -> Rule:
        return await self._writable_store.unset_metadata(rule_id, key)

    @override
    async def upsert_labels(
        self,
        rule_id: RuleId,
        labels: Set[str],
    ) -> Rule:
        return await self._writable_store.upsert_labels(rule_id, labels)

    @override
    async def remove_labels(
        self,
        rule_id: RuleId,
        labels: Set[str],
    ) -> Rule:
        return await self._writable_store.remove_labels(rule_id, labels)
