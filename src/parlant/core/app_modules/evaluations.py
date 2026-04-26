from typing import Sequence

from parlant.core.app_modules.application_context import ApplicationContext
from parlant.core.async_utils import Timeout
from parlant.core.loggers import Logger
from parlant.core.evaluations import (
    EvaluationId,
    EvaluationListener,
    Evaluation,
    EvaluationUpdateParams,
    Payload,
    PayloadDescriptor,
    PayloadKind,
)
from parlant.core.services.indexing.evaluation_service import EvaluationService
from parlant.core.store_provider import StoreProviderHints


class EvaluationModule:
    def __init__(
        self,
        application_context: ApplicationContext,
        logger: Logger,
        evaluation_service: EvaluationService,
        evaluation_listener: EvaluationListener,
    ):
        self._application_context = application_context
        self._logger = logger
        self._evaluation_service = evaluation_service
        self._evaluation_listener = evaluation_listener

    def _hints(self) -> StoreProviderHints:
        return StoreProviderHints(
            call_site="app",
            origin=self._application_context.get_origin(),
        )

    async def create(self, payloads: Sequence[Payload]) -> Evaluation:
        evaluation_id = await self._evaluation_service.create_evaluation_task(
            payload_descriptors=[
                PayloadDescriptor(PayloadKind.GUIDELINE, p) for p in [p for p in payloads]
            ],
            hints=self._hints(),
        )

        evaluation = await self._evaluation_service.read_evaluation(
            evaluation_id=evaluation_id,
            hints=self._hints(),
        )

        return evaluation

    async def read(self, evaluation_id: EvaluationId) -> Evaluation:
        return await self._evaluation_service.read_evaluation(
            evaluation_id=evaluation_id,
            hints=self._hints(),
        )

    async def find(self) -> Sequence[Evaluation]:
        return await self._evaluation_service.list_evaluations(hints=self._hints())

    async def update(
        self, evaluation_id: EvaluationId, params: EvaluationUpdateParams
    ) -> Evaluation:
        return await self._evaluation_service.update_evaluation(
            evaluation_id=evaluation_id,
            params=params,
            hints=self._hints(),
        )

    async def wait_for_completion(
        self,
        evaluation_id: EvaluationId,
        timeout: Timeout,
    ) -> bool:
        return await self._evaluation_listener.wait_for_completion(
            evaluation_id=evaluation_id,
            timeout=timeout,
        )
