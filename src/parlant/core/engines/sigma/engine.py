from collections.abc import Sequence
from typing_extensions import override

from parlant.core.emissions import EventEmitter
from parlant.core.engines.types import Context, Engine, UtteranceRequest


class SigmaEngine(Engine):
    @override
    async def process(
        self,
        context: Context,
        event_emitter: EventEmitter,
    ) -> bool:
        return False

    @override
    async def utter(
        self,
        context: Context,
        event_emitter: EventEmitter,
        requests: Sequence[UtteranceRequest],
    ) -> bool:
        return False
