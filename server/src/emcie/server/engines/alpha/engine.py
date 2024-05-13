from typing import Iterable

from emcie.server.engines.alpha.event_producer import EventProducer
from emcie.server.engines.alpha.guideline_filter import GuidelineFilter
from emcie.server.engines.common import Context, Engine, ProducedEvent
from emcie.server.guidelines import GuidelineStore
from emcie.server.sessions import SessionStore


class AlphaEngine(Engine):
    def __init__(
        self,
        session_store: SessionStore,
        guideline_store: GuidelineStore,
    ) -> None:
        self.session_store = session_store
        self.guideline_store = guideline_store

        self.event_producer = EventProducer()
        self.guide_filter = GuidelineFilter()

    async def process(self, context: Context) -> Iterable[ProducedEvent]:
        events = await self.session_store.list_events(
            session_id=context.session_id,
        )

        all_possible_guides = await self.guideline_store.list_guidelines(
            guideline_set=context.agent_id,
        )

        relevant_guides = await self.guide_filter.find_relevant_guides(
            guides=all_possible_guides,
            interaction_history=events,
        )

        return await self.event_producer.produce_events(
            interaction_history=events,
            guides=relevant_guides,
        )
