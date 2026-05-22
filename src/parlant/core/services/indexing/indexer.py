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
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Mapping
from typing_extensions import override

from parlant.core.agents import AgentStore
from parlant.core.canned_responses import CannedResponseStore
from parlant.core.common import JSONSerializable, xxh3_checksum
from parlant.core.context_variables import ContextVariableStore
from parlant.core.glossary import GlossaryStore
from parlant.core.guidelines import GuidelineStore
from parlant.core.journeys import JourneyStore
from parlant.core.relationships import RelationshipStore
from parlant.core.services.indexing.common import ProgressReport
from parlant.core.services.tools.service_registry import ServiceRegistry


@dataclass(frozen=True)
class IndexRequest:
    type: str
    id: str
    last_modification_utc: datetime
    checksum: str
    data: JSONSerializable


class Indexer(ABC):
    def __init__(
        self,
        agent_store: AgentStore,
        guideline_store: GuidelineStore,
        journey_store: JourneyStore,
        relationship_store: RelationshipStore,
        glossary_store: GlossaryStore,
        context_variable_store: ContextVariableStore,
        canned_response_store: CannedResponseStore,
        service_registry: ServiceRegistry,
    ) -> None:
        self._agent_store = agent_store
        self._guideline_store = guideline_store
        self._journey_store = journey_store
        self._relationship_store = relationship_store
        self._glossary_store = glossary_store
        self._context_variable_store = context_variable_store
        self._canned_response_store = canned_response_store
        self._service_registry = service_registry

    @abstractmethod
    async def index(
        self,
        payload: Mapping[str, Mapping[str, IndexRequest]],
        progress_report: ProgressReport,
    ) -> None: ...

    async def run(
        self,
        progress_callback: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        payload: dict[str, dict[str, IndexRequest]] = {
            "agents": {},
            "guidelines": {},
            "journeys": {},
            "relationships": {},
            "glossary": {},
            "context_variables": {},
            "canned_responses": {},
            "tools": {},
        }

        for agent in await self._agent_store.list_agents():
            payload["agents"][str(agent.id)] = self._build_request(type="agent", entity=agent)

        for guideline in await self._guideline_store.list_guidelines():
            payload["guidelines"][str(guideline.id)] = self._build_request(
                type="guideline", entity=guideline
            )

        for journey in await self._journey_store.list_journeys():
            payload["journeys"][str(journey.id)] = self._build_request(
                type="journey", entity=journey
            )

        for relationship in await self._relationship_store.list_relationships():
            payload["relationships"][str(relationship.id)] = self._build_request(
                type="relationship", entity=relationship
            )

        for term in await self._glossary_store.list_terms():
            payload["glossary"][str(term.id)] = self._build_request(type="term", entity=term)

        for variable in await self._context_variable_store.list_variables():
            payload["context_variables"][str(variable.id)] = self._build_request(
                type="context_variable", entity=variable
            )

        for canned_response in await self._canned_response_store.list_canned_responses():
            payload["canned_responses"][str(canned_response.id)] = self._build_request(
                type="canned_response", entity=canned_response
            )

        for service_name, tool_service in await self._service_registry.list_tool_services():
            for tool in await tool_service.list_tools():
                tool_id = f"{service_name}:{tool.name}"
                payload["tools"][tool_id] = self._build_request(
                    type="tool",
                    entity=tool,
                    id_override=tool_id,
                )

        async def _noop(_: float) -> None:
            return None

        progress_report = ProgressReport(progress_callback or _noop)
        await progress_report.stretch(sum(len(bucket) for bucket in payload.values()))
        await self.index(payload, progress_report)

    def _build_request(
        self,
        type: str,
        entity: Any,
        id_override: str | None = None,
    ) -> IndexRequest:
        entity_id = id_override if id_override is not None else str(getattr(entity, "id"))
        data, checksum = self._serialize_with_checksum(entity)
        return IndexRequest(
            type=type,
            id=entity_id,
            last_modification_utc=self._extract_last_modified(entity),
            checksum=checksum,
            data=data,
        )

    def _extract_last_modified(self, entity: Any) -> datetime:
        modified = getattr(entity, "last_modification_utc", None)
        if isinstance(modified, datetime):
            return modified

        created = getattr(entity, "creation_utc", None)
        if isinstance(created, datetime):
            return created

        return datetime.now(timezone.utc)

    def _serialize_with_checksum(self, entity: Any) -> tuple[JSONSerializable, str]:
        if is_dataclass(entity) and not isinstance(entity, type):
            data = _to_jsonable(asdict(entity))
        else:
            data = _to_jsonable(entity)

        existing = getattr(entity, "checksum", None)
        checksum = existing if isinstance(existing, str) else xxh3_checksum(repr(data))
        return data, checksum


class NullIndexer(Indexer):
    @override
    async def index(
        self,
        payload: Mapping[str, Mapping[str, IndexRequest]],
        progress_report: ProgressReport,
    ) -> None:
        total = sum(len(bucket) for bucket in payload.values())
        if total > 0:
            await progress_report.increment(total)


def _to_jsonable(value: Any) -> JSONSerializable:
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, Enum):
        return _to_jsonable(value.value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
