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
from typing import Awaitable, Callable, Mapping, NewType, Optional, Sequence, Set, cast
from typing_extensions import override, TypedDict, Self, Required
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone

from parlant.core import async_utils
from parlant.core.agents import CompositionMode, Effort
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
from parlant.core.nlp.embedding import Embedder, EmbedderFactory
from parlant.core.persistence.common import ObjectId, Where
from parlant.core.persistence.document_database import (
    BaseDocument,
    DocumentDatabase,
    DocumentCollection,
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
class RuleContent:
    condition: str
    action: Optional[str]
    description: Optional[str] = field(default=None)


@dataclass(frozen=True)
class Rule:
    id: RuleId
    creation_utc: datetime
    modified_utc: datetime
    content: RuleContent
    enabled: bool
    groups: Sequence[GroupId]
    metadata: Mapping[str, JSONSerializable]
    weight: Weight
    title: Optional[str] = None
    labels: Set[str] = field(default_factory=set)
    composition_mode: Optional[CompositionMode] = None
    effort_lift: Optional[Effort] = None
    track: bool = True
    priority: int = 0
    signals: Sequence[str] = field(default_factory=list)
    anti_signals: Sequence[str] = field(default_factory=list)

    def __str__(self) -> str:
        if self.content.condition and self.content.action:
            return f"When {self.content.condition}, then {self.content.action}"
        elif self.content.condition:
            return f"Observation: {self.content.condition}"
        elif self.content.action:
            return self.content.action
        else:
            raise Exception("Invalid rule content")

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def query(self) -> str:
        """The canonical "what does this rule talk about" text — used to find
        the glossary terms the rule depends on (at matching time and during
        evaluation)."""
        return compose_rule_query(
            title=self.title,
            condition=self.content.condition,
            action=self.content.action,
            description=self.content.description,
        )


def compose_rule_query(
    title: Optional[str],
    condition: Optional[str],
    action: Optional[str],
    description: Optional[str],
) -> str:
    """Concatenate a rule's definition into a relevance-search query,
    skipping absent parts. Shared by :attr:`Rule.query` and evaluation
    flows that hold the parts separately (payloads carry content + title)."""
    title = (title or "").strip()
    condition = (condition or "").strip()
    action = (action or "").strip()
    description = (description or "").strip()

    lines: list[str] = []

    if title:
        lines.append(title)

    if condition and action:
        lines.append(f"When {condition}, then {action}")
    elif condition:
        lines.append(condition)
    elif action:
        lines.append(action)

    if description:
        lines.append(description)

    return "\n\n".join(lines)


class RuleUpdateParams(TypedDict, total=False):
    condition: str
    action: Optional[str]
    description: Optional[str]
    title: Optional[str]
    weight: Weight
    enabled: bool
    metadata: Mapping[str, JSONSerializable]
    composition_mode: Optional[CompositionMode]
    effort_lift: Optional[Effort]
    track: bool
    priority: int
    signals: Sequence[str]
    anti_signals: Sequence[str]


@dataclass(frozen=True)
class RuleRelevanceResult:
    rule: Rule
    score: float


class RuleStore(ABC):
    @abstractmethod
    async def create_rule(
        self,
        condition: str,
        action: Optional[str] = None,
        description: Optional[str] = None,
        title: Optional[str] = None,
        weight: Optional[Weight] = None,
        metadata: Mapping[str, JSONSerializable] = {},
        creation_utc: Optional[datetime] = None,
        enabled: bool = True,
        groups: Optional[Sequence[GroupId]] = None,
        id: Optional[RuleId] = None,
        composition_mode: Optional[CompositionMode] = None,
        effort_lift: Optional[Effort] = None,
        track: bool = True,
        labels: Optional[Set[str]] = None,
        priority: int = 0,
        signals: Sequence[str] = [],
        anti_signals: Sequence[str] = [],
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
        rule_content: RuleContent,
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


class RuleDocument_v0_1_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    rule_set: str
    condition: str
    action: str


class RuleDocument_v0_2_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    rule_set: str
    condition: str
    action: str
    enabled: bool


class RuleDocument_v0_3_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    condition: str
    action: str
    enabled: bool


class RuleDocument_v0_4_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    condition: str
    action: Optional[str]
    description: Optional[str]
    enabled: bool
    metadata: Mapping[str, JSONSerializable]


class RuleDocument_v0_5_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    condition: str
    action: Optional[str]
    description: Optional[str]
    enabled: bool
    metadata: Mapping[str, JSONSerializable]


class RuleDocument_v0_6_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    condition: str
    action: Optional[str]
    description: Optional[str]
    criticality: str
    enabled: bool
    metadata: Mapping[str, JSONSerializable]


class RuleDocument_v0_7_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    condition: str
    action: Optional[str]
    description: Optional[str]
    criticality: str
    enabled: bool
    metadata: Mapping[str, JSONSerializable]
    composition_mode: Optional[str]


class RuleDocument_v0_8_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    condition: str
    action: Optional[str]
    description: Optional[str]
    criticality: str
    enabled: bool
    metadata: Mapping[str, JSONSerializable]
    composition_mode: Optional[str]
    track: bool


class RuleDocument_v0_9_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    condition: str
    action: Optional[str]
    description: Optional[str]
    criticality: str
    enabled: bool
    metadata: Mapping[str, JSONSerializable]
    composition_mode: Optional[str]
    track: bool
    labels: Sequence[str]


class RuleDocument_v0_10_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    condition: str
    action: Optional[str]
    description: Optional[str]
    criticality: str
    enabled: bool
    metadata: Mapping[str, JSONSerializable]
    composition_mode: Optional[str]
    track: bool
    labels: Sequence[str]
    priority: int


class RuleDocument_v0_11_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    last_modified: str
    condition: str
    action: Optional[str]
    description: Optional[str]
    title: Optional[str]
    criticality: str
    enabled: bool
    metadata: Mapping[str, JSONSerializable]
    composition_mode: Optional[str]
    track: bool
    labels: Sequence[str]
    priority: int


class RuleDocument_v0_12_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    last_modified: str
    condition: str
    action: Optional[str]
    description: Optional[str]
    title: Optional[str]
    criticality: str
    enabled: bool
    metadata: Mapping[str, JSONSerializable]
    composition_mode: Optional[str]
    track: bool
    labels: Sequence[str]
    priority: int
    signals: Sequence[str]


class RuleDocument_v0_13_0(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    last_modified: str
    condition: str
    action: Optional[str]
    description: Optional[str]
    title: Optional[str]
    criticality: str
    enabled: bool
    metadata: Mapping[str, JSONSerializable]
    composition_mode: Optional[str]
    effort: Optional[str]
    track: bool
    labels: Sequence[str]
    priority: int
    signals: Sequence[str]


class RuleDocument(TypedDict, total=False):
    id: ObjectId
    version: Version.String
    creation_utc: str
    last_modified: str
    condition: str
    action: Optional[str]
    description: Optional[str]
    title: Optional[str]
    weight: str
    enabled: bool
    metadata: Mapping[str, JSONSerializable]
    composition_mode: Optional[str]
    effort_lift: Optional[str]
    track: bool
    labels: Sequence[str]
    priority: int
    signals: Sequence[str]
    anti_signals: Sequence[str]


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


async def rule_document_converter_0_1_0_to_0_2_0(doc: BaseDocument) -> Optional[BaseDocument]:
    d = cast(RuleDocument_v0_1_0, doc)
    return RuleDocument_v0_2_0(
        id=d["id"],
        version=Version.String("0.2.0"),
        creation_utc=d["creation_utc"],
        rule_set=d["rule_set"],
        condition=d["condition"],
        action=d["action"],
        enabled=True,
    )


class RuleVectorStore(RuleStore):
    VERSION = Version.from_string("0.14.0")

    def __init__(
        self,
        id_generator: IdGenerator,
        vector_db: VectorDatabase,
        document_db: DocumentDatabase,
        embedder_type_provider: Callable[[], Awaitable[type[Embedder]]],
        embedder_factory: EmbedderFactory,
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
        self._embedder_factory = embedder_factory
        self._embedder_type_provider = embedder_type_provider
        self._embedder: Embedder

    async def _vector_document_loader(self, doc: VectorDocument) -> Optional[RuleVectorDocument]:
        return cast(RuleVectorDocument, doc)

    async def _document_loader(self, doc: BaseDocument) -> Optional[RuleDocument]:
        async def v0_13_0_to_v0_14_0(doc: BaseDocument) -> Optional[BaseDocument]:
            d = cast(RuleDocument_v0_13_0, doc)
            return RuleDocument(
                id=d["id"],
                version=Version.String("0.14.0"),
                creation_utc=d["creation_utc"],
                last_modified=d.get("last_modified", d["creation_utc"]),
                condition=d["condition"],
                action=d["action"],
                description=d.get("description", None),
                title=d.get("title", None),
                weight=d["criticality"],
                enabled=d["enabled"],
                metadata=d["metadata"],
                composition_mode=d.get("composition_mode"),
                effort_lift=d.get("effort"),
                track=d.get("track", True),
                labels=d.get("labels", []),
                priority=d.get("priority", 0),
                signals=d.get("signals", []),
                anti_signals=[],
            )

        async def v0_12_0_to_v0_13_0(doc: BaseDocument) -> Optional[BaseDocument]:
            d = cast(RuleDocument_v0_12_0, doc)
            return RuleDocument_v0_13_0(
                id=d["id"],
                version=Version.String("0.13.0"),
                creation_utc=d["creation_utc"],
                last_modified=d.get("last_modified", d["creation_utc"]),
                condition=d["condition"],
                action=d["action"],
                description=d.get("description", None),
                title=d.get("title", None),
                criticality=d["criticality"],
                enabled=d["enabled"],
                metadata=d["metadata"],
                composition_mode=d.get("composition_mode"),
                effort=None,
                track=d.get("track", True),
                labels=d.get("labels", []),
                priority=d.get("priority", 0),
                signals=d.get("signals", []),
            )

        async def v0_11_0_to_v0_12_0(doc: BaseDocument) -> Optional[BaseDocument]:
            d = cast(RuleDocument_v0_11_0, doc)
            return RuleDocument_v0_12_0(
                id=d["id"],
                version=Version.String("0.12.0"),
                creation_utc=d["creation_utc"],
                last_modified=d.get("last_modified", d["creation_utc"]),
                condition=d["condition"],
                action=d["action"],
                description=d.get("description", None),
                title=d.get("title", None),
                criticality=d["criticality"],
                enabled=d["enabled"],
                metadata=d["metadata"],
                composition_mode=d.get("composition_mode"),
                track=d.get("track", True),
                labels=d.get("labels", []),
                priority=d.get("priority", 0),
                signals=[],
            )

        async def v0_10_0_to_v0_11_0(doc: BaseDocument) -> Optional[BaseDocument]:
            d = cast(RuleDocument_v0_10_0, doc)
            return RuleDocument_v0_11_0(
                id=d["id"],
                version=Version.String("0.11.0"),
                creation_utc=d["creation_utc"],
                last_modified=d["creation_utc"],
                condition=d["condition"],
                action=d["action"],
                description=d.get("description", None),
                title=None,  # Default to None for existing rules
                criticality=d["criticality"],
                enabled=d["enabled"],
                metadata=d["metadata"],
                composition_mode=d.get("composition_mode"),
                track=d.get("track", True),
                labels=d.get("labels", []),
                priority=d.get("priority", 0),
            )

        async def v0_9_0_to_v0_10_0(doc: BaseDocument) -> Optional[BaseDocument]:
            d = cast(RuleDocument_v0_9_0, doc)
            return RuleDocument_v0_10_0(
                id=d["id"],
                version=Version.String("0.10.0"),
                creation_utc=d["creation_utc"],
                condition=d["condition"],
                action=d["action"],
                description=d.get("description", None),
                criticality=d["criticality"],
                enabled=d["enabled"],
                metadata=d["metadata"],
                composition_mode=d.get("composition_mode"),
                track=d.get("track", True),
                labels=d.get("labels", []),
                priority=0,  # Default to 0 for existing rules
            )

        async def v0_8_0_to_v0_9_0(doc: BaseDocument) -> Optional[BaseDocument]:
            d = cast(RuleDocument_v0_8_0, doc)
            return RuleDocument_v0_9_0(
                id=d["id"],
                version=Version.String("0.9.0"),
                creation_utc=d["creation_utc"],
                condition=d["condition"],
                action=d["action"],
                description=d.get("description", None),
                criticality=d["criticality"],
                enabled=d["enabled"],
                metadata=d["metadata"],
                composition_mode=d.get("composition_mode"),
                track=d.get("track", True),
                labels=[],  # Default to empty labels for existing rules
            )

        async def v0_7_0_to_v0_8_0(doc: BaseDocument) -> Optional[BaseDocument]:
            d = cast(RuleDocument_v0_7_0, doc)
            return RuleDocument_v0_8_0(
                id=d["id"],
                version=Version.String("0.8.0"),
                creation_utc=d["creation_utc"],
                condition=d["condition"],
                action=d["action"],
                description=d.get("description", None),
                criticality=d["criticality"],
                enabled=d["enabled"],
                metadata=d["metadata"],
                composition_mode=d.get("composition_mode"),
                track=True,  # Default to True for existing rules
            )

        async def v0_6_0_to_v0_7_0(doc: BaseDocument) -> Optional[BaseDocument]:
            d = cast(RuleDocument_v0_6_0, doc)
            return RuleDocument_v0_7_0(
                id=d["id"],
                version=Version.String("0.7.0"),
                creation_utc=d["creation_utc"],
                condition=d["condition"],
                action=d["action"],
                description=d.get("description", None),
                criticality=d["criticality"],
                enabled=d["enabled"],
                metadata=d["metadata"],
                composition_mode=None,  # Default to None for existing rules
            )

        async def v0_5_0_to_v0_6_0(doc: BaseDocument) -> Optional[BaseDocument]:
            d = cast(RuleDocument_v0_5_0, doc)
            return RuleDocument_v0_6_0(
                id=d["id"],
                version=Version.String("0.6.0"),
                creation_utc=d["creation_utc"],
                condition=d["condition"],
                action=d["action"],
                description=d.get("description", None),
                criticality="medium",  # Default to MEDIUM for existing rules
                enabled=d["enabled"],
                metadata=d["metadata"],
            )

        async def v0_4_0_to_v0_5_0(doc: BaseDocument) -> Optional[BaseDocument]:
            d = cast(RuleDocument_v0_4_0, doc)
            return RuleDocument_v0_5_0(
                id=d["id"],
                version=Version.String("0.5.0"),
                creation_utc=d["creation_utc"],
                condition=d["condition"],
                action=d["action"],
                description=d.get("description", None),
                enabled=d["enabled"],
                metadata=d["metadata"],
            )

        async def v0_3_0_to_v0_4_0(doc: BaseDocument) -> Optional[BaseDocument]:
            d = cast(RuleDocument_v0_3_0, doc)
            return RuleDocument_v0_4_0(
                id=d["id"],
                version=Version.String("0.4.0"),
                creation_utc=d["creation_utc"],
                condition=d["condition"],
                action=d["action"],
                enabled=d["enabled"],
                metadata={},
            )

        async def v0_2_0_to_v0_3_0(doc: BaseDocument) -> Optional[BaseDocument]:
            raise Exception(
                "This code should not be reached! Please run the 'parlant-prepare-migration' script."
            )

        return await DocumentMigrationHelper[RuleDocument](
            self,
            {
                "0.1.0": rule_document_converter_0_1_0_to_0_2_0,
                "0.2.0": v0_2_0_to_v0_3_0,
                "0.3.0": v0_3_0_to_v0_4_0,
                "0.4.0": v0_4_0_to_v0_5_0,
                "0.5.0": v0_5_0_to_v0_6_0,
                "0.6.0": v0_6_0_to_v0_7_0,
                "0.7.0": v0_7_0_to_v0_8_0,
                "0.8.0": v0_8_0_to_v0_9_0,
                "0.9.0": v0_9_0_to_v0_10_0,
                "0.10.0": v0_10_0_to_v0_11_0,
                "0.11.0": v0_11_0_to_v0_12_0,
                "0.12.0": v0_12_0_to_v0_13_0,
                "0.13.0": v0_13_0_to_v0_14_0,
            },
        ).migrate(doc)

    async def _association_document_loader(
        self, doc: BaseDocument
    ) -> Optional[RuleTagAssociationDocument]:
        if doc["version"] == "0.3.0":
            d = cast(RuleTagAssociationDocument, doc)
            return RuleTagAssociationDocument(
                id=d["id"],
                version=Version.String("0.5.0"),
                creation_utc=d["creation_utc"],
                rule_id=d["rule_id"],
                group_id=d["group_id"],
            )

        if doc["version"] == "0.4.0":
            d = cast(RuleTagAssociationDocument, doc)
            return RuleTagAssociationDocument(
                id=d["id"],
                version=Version.String("0.5.0"),
                creation_utc=d["creation_utc"],
                rule_id=d["rule_id"],
                group_id=d["group_id"],
            )

        if doc["version"] == "0.5.0":
            return cast(RuleTagAssociationDocument, doc)

        return None

    async def __aenter__(self) -> Self:
        embedder_type = await self._embedder_type_provider()

        self._embedder = self._embedder_factory.create_embedder(embedder_type)

        async with VectorDocumentStoreMigrationHelper(
            store=self,
            database=self._vector_db,
            allow_migration=self._allow_migration,
        ):
            self._vector_collection = await self._vector_db.get_or_create_collection(
                name=f"{self._collections_prefix}_rules" if self._collections_prefix else "rules",
                schema=RuleVectorDocument,
                embedder_type=embedder_type,
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
            condition=rule.content.condition,
            action=rule.content.action,
            description=rule.content.description,
            title=rule.title,
            weight=rule.weight.value,
            enabled=rule.enabled,
            metadata=rule.metadata,
            composition_mode=(rule.composition_mode.value if rule.composition_mode else None),
            effort_lift=rule.effort_lift.value if rule.effort_lift else None,
            track=rule.track,
            labels=list(rule.labels),
            priority=rule.priority,
            signals=list(rule.signals),
            anti_signals=list(rule.anti_signals),
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

        composition_mode_str = rule_document.get("composition_mode")
        composition_mode = CompositionMode(composition_mode_str) if composition_mode_str else None
        effort_str = rule_document.get("effort")
        effort = Effort(effort_str) if effort_str else None

        return Rule(
            id=RuleId(rule_document["id"]),
            creation_utc=datetime.fromisoformat(rule_document["creation_utc"]),
            modified_utc=datetime.fromisoformat(rule_document["last_modified"]),
            content=RuleContent(
                condition=rule_document["condition"],
                action=rule_document["action"],
                description=rule_document.get("description", None),
            ),
            title=rule_document.get("title", None),
            weight=Weight(rule_document["weight"]),
            enabled=rule_document["enabled"],
            groups=[GroupId(group_id) for group_id in group_ids],
            metadata=rule_document["metadata"],
            labels=set(rule_document.get("labels", [])),
            composition_mode=composition_mode,
            effort_lift=effort,
            track=rule_document.get("track", True),
            priority=rule_document.get("priority", 0),
            signals=list(rule_document.get("signals", [])),
            anti_signals=list(rule_document.get("anti_signals", [])),
        )

    def _rule_embedding_content(self, rule: Rule | RuleContent) -> str:
        """Render a rule's content as the string to embed.

        Treats ``None`` / empty / whitespace-only condition, action, and
        description as absent. The condition+action pair reads as
        ``When {condition}, then {action}``; a lone condition or action gets a
        labeled form; a non-empty description is appended as its own block.
        """
        content = rule.content if isinstance(rule, Rule) else rule
        title = rule.title if isinstance(rule, Rule) else None

        condition = (content.condition or "").strip()
        action = (content.action or "").strip()
        description = (content.description or "").strip()

        if title:
            head = f"# {title}\n\n"
        else:
            head = ""

        if condition and action:
            head += f"When {condition}, then {action}"
        elif condition:
            head += f"Condition: {condition}"
        elif action:
            head += f"Action: {action}"

        if description:
            text_content = f"{head}\n\nDescription: {description}"
        else:
            text_content = head

        if not text_content:
            raise ValueError("Rule has no content")

        return text_content

    def _list_rule_contents(self, rule: Rule) -> list[str]:
        """The independent strings to embed for a rule: its rendered
        content followed by each signal as its own vector."""
        return [self._rule_embedding_content(rule), *rule.signals]

    async def _insert_vector_documents(self, rule: Rule) -> None:
        insertion_tasks = []

        for content in self._list_rule_contents(rule):
            doc_id = self._id_generator.generate(xxh3_checksum(f"{rule.id}{content}"))

            vec_doc = RuleVectorDocument(
                id=ObjectId(doc_id),
                rule_id=ObjectId(rule.id),
                version=self.VERSION.to_string(),
                content=content,
                checksum=xxh3_checksum(content),
            )

            insertion_tasks.append(self._vector_collection.insert_one(document=vec_doc))

        await async_utils.safe_gather(*insertion_tasks)

    async def _delete_vector_documents(self, rule_id: RuleId) -> None:
        for v_doc in await self._vector_collection.find(filters={"rule_id": {"$eq": rule_id}}):
            await self._vector_collection.delete_one(filters={"id": {"$eq": v_doc["id"]}})

    @override
    async def create_rule(
        self,
        condition: str,
        action: Optional[str] = None,
        description: Optional[str] = None,
        title: Optional[str] = None,
        weight: Optional[Weight] = None,
        metadata: Mapping[str, JSONSerializable] = {},
        creation_utc: Optional[datetime] = None,
        enabled: bool = True,
        groups: Optional[Sequence[GroupId]] = None,
        id: Optional[RuleId] = None,
        composition_mode: Optional[CompositionMode] = None,
        effort_lift: Optional[Effort] = None,
        track: bool = True,
        labels: Optional[Set[str]] = None,
        priority: int = 0,
        signals: Sequence[str] = [],
        anti_signals: Sequence[str] = [],
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
                rule_checksum = xxh3_checksum(f"{condition}{action or ''}{enabled}{metadata}")
                rule_id = RuleId(self._id_generator.generate(rule_checksum))

            rule = Rule(
                id=rule_id,
                creation_utc=creation_utc,
                modified_utc=creation_utc,
                content=RuleContent(
                    condition=condition,
                    action=action,
                    description=description,
                ),
                title=title,
                weight=weight,
                enabled=enabled,
                groups=groups or [],
                metadata=metadata,
                labels=labels or set(),
                composition_mode=composition_mode,
                effort_lift=effort_lift,
                track=track,
                priority=priority,
                signals=list(signals),
                anti_signals=list(anti_signals),
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

            rules = [
                await self._deserialize(d) for d in await self._collection.find(filters=filters)
            ]

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
                    **({"condition": params["condition"]} if "condition" in params else {}),
                    **({"action": params["action"]} if "action" in params else {}),
                    **({"description": params["description"]} if "description" in params else {}),
                    **({"title": params["title"]} if "title" in params else {}),
                    **({"weight": params["weight"].value} if "weight" in params else {}),
                    **({"enabled": params["enabled"]} if "enabled" in params else {}),
                    **(
                        {
                            "composition_mode": (
                                # Note that updating to None is also valid
                                params["composition_mode"].value
                                if params["composition_mode"] is not None
                                else None
                            )
                        }
                        if "composition_mode" in params
                        else {}
                    ),
                    **(
                        {
                            "effort": (
                                # Note that updating to None is also valid
                                params["effort_lift"].value
                                if params["effort_lift"] is not None
                                else None
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
            # (condition / action / description / signals).
            if any(key in params for key in ("condition", "action", "description", "signals")):
                await self._delete_vector_documents(rule_id)
                await self._insert_vector_documents(updated)

        return updated

    @override
    async def find_rule(
        self,
        rule_content: RuleContent,
    ) -> Rule:
        async with self._lock.reader_lock:
            filters = {
                "condition": {"$eq": rule_content.condition},
                **({"action": {"$eq": rule_content.action}} if rule_content.action else {}),
            }

            rule_document = await self._collection.find_one(filters=cast(Where, filters))

        if not rule_document:
            raise ItemNotFoundError(
                item_id=UniqueId(f"{rule_content.condition}{rule_content.action}")
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
        condition: str,
        action: Optional[str] = None,
        description: Optional[str] = None,
        title: Optional[str] = None,
        weight: Optional[Weight] = None,
        metadata: Mapping[str, JSONSerializable] = {},
        creation_utc: Optional[datetime] = None,
        enabled: bool = True,
        groups: Optional[Sequence[GroupId]] = None,
        id: Optional[RuleId] = None,
        composition_mode: Optional[CompositionMode] = None,
        effort_lift: Optional[Effort] = None,
        track: bool = True,
        labels: Optional[Set[str]] = None,
        priority: int = 0,
        signals: Sequence[str] = [],
        anti_signals: Sequence[str] = [],
    ) -> Rule:
        return await self._writable_store.create_rule(
            condition=condition,
            action=action,
            description=description,
            title=title,
            weight=weight,
            metadata=metadata,
            creation_utc=creation_utc,
            enabled=enabled,
            groups=groups,
            id=id,
            composition_mode=composition_mode,
            effort_lift=effort_lift,
            track=track,
            labels=labels,
            priority=priority,
            signals=signals,
            anti_signals=anti_signals,
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
        rule_content: RuleContent,
    ) -> Rule:
        results = await safe_gather(
            *[try_or_none(store.find_rule(rule_content)) for store in self._all_stores]
        )
        result = next((r for r in results if r is not None), None)
        if result is None:
            raise ItemNotFoundError(
                item_id=UniqueId(f"{rule_content.condition}{rule_content.action}")
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
