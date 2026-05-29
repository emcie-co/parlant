from collections.abc import Sequence

from parlant.core.agents import CompositionMode, MessageOutputMode
from parlant.core.engines.alpha.message_event_composer import MessageEventComposition
from parlant.core.engines.alpha.optimization_policy import OptimizationPolicy
from parlant.core.engines.engine_context import EngineContext
from parlant.core.engines.sigma.responder.streaming_responder import StreamingResponder
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.tracer import Tracer


class Responder:
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        optimization_policy: OptimizationPolicy,
        streaming_responder: StreamingResponder,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._meter = meter
        self._optimization_policy = optimization_policy
        self._streaming_responder = streaming_responder

    async def respond(self, context: EngineContext) -> Sequence[MessageEventComposition]:
        composition_mode = await self._resolve_composition_mode(context)
        output_mode = context.agent.message_output_mode

        if (
            output_mode == MessageOutputMode.STREAM
            and composition_mode == CompositionMode.CANNED_FLUID
        ):
            return await self._streaming_responder.do_respond(context)
        else:
            raise Exception(f"Unsupported message output mode: {output_mode}")

    async def _resolve_composition_mode(self, context: EngineContext) -> CompositionMode:
        """Resolve effective composition mode from matched guidelines.

        Most restrictive rule: CANNED_STRICT > CANNED_COMPOSITED > CANNED_FLUID
        """
        if context.agent.composition_mode == CompositionMode.CANNED_STRICT:
            return CompositionMode.CANNED_STRICT

        restrictiveness_priority = {
            CompositionMode.CANNED_STRICT: 3,
            CompositionMode.CANNED_COMPOSITED: 2,
            CompositionMode.CANNED_FLUID: 1,
        }

        most_restrictive_mode: CompositionMode | None = None
        max_restrictiveness = 0

        # Check all matched guidelines for composition mode
        for guideline in context.state.guidelines:
            if guideline.composition_mode is not None:
                mode = guideline.composition_mode

                # Track most restrictive (only CANNED_* modes)
                if mode in restrictiveness_priority:
                    restrictiveness = restrictiveness_priority[mode]
                    if restrictiveness > max_restrictiveness:
                        most_restrictive_mode = mode
                        max_restrictiveness = restrictiveness

        # Default to agent's composition mode
        if most_restrictive_mode is None:
            most_restrictive_mode = context.agent.composition_mode

        return most_restrictive_mode
