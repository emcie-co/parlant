from typing import Sequence
from lagom import Container

from parlant.core.async_utils import Timeout
from parlant.core.loggers import Logger
from parlant.core.evaluations import (
    EvaluationId,
    EvaluationListener,
    EvaluationStore,
    Evaluation,
    EvaluationUpdateParams,
    Payload,
    PayloadDescriptor,
    PayloadKind,
)
from parlant.core.services.indexing.behavioral_change_evaluation import BehavioralChangeEvaluator


class EvaluationModule:
    def __init__(
        self,
        container: Container,
    ):
        self._logger = container[Logger]
        self._evaluation_store = container[EvaluationStore]
        self._evaluation_service = container[BehavioralChangeEvaluator]
        self._evaluation_listener = container[EvaluationListener]

    async def create(self, payloads: Sequence[Payload]) -> Evaluation:
        evaluation_id = await self._evaluation_service.create_evaluation_task(
            payload_descriptors=[
                PayloadDescriptor(PayloadKind.GUIDELINE, p) for p in [p for p in payloads]
            ],
        )

        evaluation = await self._evaluation_store.read_evaluation(evaluation_id)

        return evaluation

    async def read(self, evaluation_id: EvaluationId) -> Evaluation:
        evaluation = await self._evaluation_store.read_evaluation(evaluation_id=evaluation_id)
        return evaluation

    async def find(self) -> Sequence[Evaluation]:
        evaluations = await self._evaluation_store.list_evaluations()
        return evaluations

    async def update(
        self, evaluation_id: EvaluationId, params: EvaluationUpdateParams
    ) -> Evaluation:
        evaluation = await self._evaluation_store.update_evaluation(
            evaluation_id=evaluation_id, params=params
        )
        return evaluation

    async def wait_for_completion(
        self,
        evaluation_id: EvaluationId,
        timeout: Timeout,
    ) -> bool:
        return await self._evaluation_listener.wait_for_completion(
            evaluation_id=evaluation_id,
            timeout=timeout,
        )
