import asyncio

from parlant.core.background_tasks import BackgroundTaskService
from parlant.core.loggers import Logger


async def test_that_cancel_returns_true_for_running_task(logger: Logger) -> None:
    async with BackgroundTaskService(logger) as background_tasks:
        task_started = asyncio.Event()

        async def run_forever() -> None:
            task_started.set()
            await asyncio.Future[None]()

        await background_tasks.start(run_forever(), tag="test-task")
        await asyncio.wait_for(task_started.wait(), timeout=1)

        assert await background_tasks.cancel(tag="test-task") is True


async def test_that_cancel_returns_false_for_missing_task(logger: Logger) -> None:
    async with BackgroundTaskService(logger) as background_tasks:
        assert await background_tasks.cancel(tag="missing-task") is False
