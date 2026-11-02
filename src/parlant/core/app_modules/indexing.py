from typing import Sequence

from parlant.core.app_modules.request_context import RequestContext
from parlant.core.async_utils import Timeout
from parlant.core.loggers import Logger
from parlant.core.indexing import (
    IndexingListener,
    IndexingTask,
    IndexingTaskId,
    IndexingTaskUpdateParams,
    Payload,
    PayloadDescriptor,
    PayloadKind,
)
from parlant.core.services.indexing.indexing_service import IndexingService
from parlant.core.store_provider import StoreProviderHints


class IndexingModule:
    def __init__(
        self,
        request_context: RequestContext,
        logger: Logger,
        indexing_service: IndexingService,
        indexing_listener: IndexingListener,
    ):
        self._request_context = request_context
        self._logger = logger
        self._indexing_service = indexing_service
        self._indexing_listener = indexing_listener

    def _hints(self) -> StoreProviderHints:
        return StoreProviderHints(
            call_site="app",
            origin=self._request_context.get_origin(),
        )

    async def create(self, payloads: Sequence[Payload]) -> IndexingTask:
        indexing_task_id = await self._indexing_service.create_indexing_task(
            payload_descriptors=[
                PayloadDescriptor(PayloadKind.RULE, p) for p in [p for p in payloads]
            ],
            hints=self._hints(),
        )

        indexing_task = await self._indexing_service.read_indexing_task(
            indexing_task_id=indexing_task_id,
            hints=self._hints(),
        )

        return indexing_task

    async def read(self, indexing_task_id: IndexingTaskId) -> IndexingTask:
        return await self._indexing_service.read_indexing_task(
            indexing_task_id=indexing_task_id,
            hints=self._hints(),
        )

    async def find(self) -> Sequence[IndexingTask]:
        return await self._indexing_service.list_indexing_tasks(hints=self._hints())

    async def update(
        self, indexing_task_id: IndexingTaskId, params: IndexingTaskUpdateParams
    ) -> IndexingTask:
        return await self._indexing_service.update_indexing_task(
            indexing_task_id=indexing_task_id,
            params=params,
            hints=self._hints(),
        )

    async def wait_for_completion(
        self,
        indexing_task_id: IndexingTaskId,
        timeout: Timeout,
    ) -> bool:
        return await self._indexing_listener.wait_for_completion(
            indexing_task_id=indexing_task_id,
            timeout=timeout,
        )
