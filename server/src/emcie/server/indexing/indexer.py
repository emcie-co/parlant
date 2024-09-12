import asyncio
from itertools import chain
import json
from pathlib import Path
from typing import Any, NamedTuple, Sequence

from emcie.server.core.agents import AgentStore
from emcie.server.core.common import JSONSerializable
from emcie.server.core.guideline_connections import GuidelineConnectionStore
from emcie.server.core.guidelines import Guideline, GuidelineContent, GuidelineId, GuidelineStore
from emcie.server.indexing.guideline_connection_proposer import GuidelineConnectionProposer
from emcie.server.logger import Logger
from emcie.server.utils import md5_checksum


class GuidelineIndexEntryItem(NamedTuple):
    guideline_id: GuidelineId
    checksum: str


class GuidelineIndexEntry(NamedTuple):
    guideline_set: str
    items: list[GuidelineIndexEntryItem]


class GuidelineIndexer:
    def __init__(
        self,
        logger: Logger,
        guideline_store: GuidelineStore,
        guideline_connection_store: GuidelineConnectionStore,
        agent_store: AgentStore,
        guideline_connection_proposer: GuidelineConnectionProposer,
    ) -> None:
        self.logger = logger

        self._guideline_store = guideline_store
        self._guideline_connection_store = guideline_connection_store
        self._agent_store = agent_store
        self._guideline_connection_proposer = guideline_connection_proposer

    @staticmethod
    def _guideline_checksum(guideline: Guideline) -> str:
        return md5_checksum(f"{guideline.content.predicate}_{guideline.content.action}")

    async def should_index(
        self,
        last_known_state: list[GuidelineIndexEntry],
    ) -> bool:
        for agent in await self._agent_store.list_agents():
            agent_guidelines = await self._guideline_store.list_guidelines(agent.id)

            introduced, _, deleted = self._assess_guideline_modifications(
                agent_guidelines,
                next(iter([e[1] for e in last_known_state if e[0] == agent.id]), []),
            )

            if introduced or deleted:
                return True

        return False

    async def index(
        self,
        last_known_state: list[GuidelineIndexEntry],
    ) -> list[GuidelineIndexEntry]:
        self.logger.info("Guideline indexing started")

        current_state: list[GuidelineIndexEntry] = []

        for agent in await self._agent_store.list_agents():
            agent_guidelines = await self._guideline_store.list_guidelines(agent.id)

            await self._index_guideline_connections(
                guidelines=agent_guidelines,
                last_know_state_of_set=next(
                    iter([e[1] for e in last_known_state if e[0] == agent.id]), []
                ),
            )

            current_state.append(
                GuidelineIndexEntry(
                    guideline_set=agent.id,
                    items=[
                        GuidelineIndexEntryItem(
                            guideline_id=g.id, checksum=self._guideline_checksum(g)
                        )
                        for g in agent_guidelines
                    ],
                )
            )

        self.logger.info("Guideline indexing finished")

        return current_state

    def _assess_guideline_modifications(
        self,
        guidelines: Sequence[Guideline],
        last_know_state_of_set: list[GuidelineIndexEntryItem],
    ) -> tuple[Sequence[Guideline], Sequence[Guideline], list[GuidelineIndexEntryItem]]:
        introduced_guidelines, existing_guidelines = [], []
        deleted_guidelines = {i[1]: i for i in last_know_state_of_set}

        for guideline in guidelines:
            guideline_checksum = self._guideline_checksum(guideline)

            if guideline_checksum in deleted_guidelines:
                existing_guidelines.append(guideline)
                del deleted_guidelines[guideline_checksum]
            else:
                introduced_guidelines.append(guideline)

        return introduced_guidelines, existing_guidelines, list(deleted_guidelines.values())

    async def _remove_deleted_guidelines_connections(
        self,
        deleted_guidelines: list[GuidelineIndexEntryItem],
    ) -> None:
        for item in deleted_guidelines:
            try:
                connections = list(
                    await self._guideline_connection_store.list_connections(
                        indirect=False, source=item[0]
                    )
                )
                connections.extend(
                    await self._guideline_connection_store.list_connections(
                        indirect=False, target=item[0]
                    )
                )
                await asyncio.gather(
                    *(self._guideline_connection_store.delete_connection(c.id) for c in connections)
                )
            except ValueError:  # in case connections with the guideline id do not exist
                pass

    async def _index_guideline_connections(
        self,
        guidelines: Sequence[Guideline],
        last_know_state_of_set: list[GuidelineIndexEntryItem],
    ) -> None:
        introduced, existing, deleted = self._assess_guideline_modifications(
            guidelines, last_know_state_of_set
        )

        await self._remove_deleted_guidelines_connections(deleted)

        proposed_connections = (
            p
            for p in await self._guideline_connection_proposer.propose_connections(
                introduced_guidelines=[
                    GuidelineContent(predicate=s.content.predicate, action=s.content.action)
                    for s in introduced
                ],
                existing_guidelines=[
                    GuidelineContent(predicate=s.content.predicate, action=s.content.action)
                    for s in existing
                ],
            )
            if p.score >= 6
        )

        data_to_guideline = {
            f"{s.content.predicate}_{s.content.action}": s for s in chain(introduced, existing)
        }

        for p in proposed_connections:
            self.logger.debug(f"Add guideline connection between source: {p.source} and {p.target}")

            await self._guideline_connection_store.update_connection(
                source=data_to_guideline[f"{p.source.predicate}_{p.source.action}"].id,
                target=data_to_guideline[f"{p.target.predicate}_{p.target.action}"].id,
                kind=p.kind,
            )


class Indexer:
    def __init__(
        self,
        index_file: Path,
        logger: Logger,
        guideline_store: GuidelineStore,
        guideline_connection_store: GuidelineConnectionStore,
        agent_store: AgentStore,
        guideline_connection_proposer: GuidelineConnectionProposer,
    ) -> None:
        self._index_file = index_file
        self._guideline_indexer = GuidelineIndexer(
            logger,
            guideline_store,
            guideline_connection_store,
            agent_store,
            guideline_connection_proposer,
        )

    def _read_index_file(self) -> dict[str, Any]:
        if self._index_file.exists() and self._index_file.stat().st_size > 0:
            with self._index_file.open("r") as f:
                data: dict[str, Any] = json.load(f)
                return data
        return {}

    def _write_index_file(self, indices: JSONSerializable) -> None:
        with self._index_file.open("w") as f:
            json.dump(indices, f, indent=2)

    async def should_index(self) -> bool:
        indexed_data = self._read_index_file()

        return await self._guideline_indexer.should_index(indexed_data.get("guidelines", {}))

    async def index(self) -> None:
        indexed_data = self._read_index_file()

        data = {}
        data["guidelines"] = await self._guideline_indexer.index(
            indexed_data.get("guidelines", []),
        )

        self._write_index_file(data)
