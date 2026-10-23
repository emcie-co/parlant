import asyncio
import time

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


async def _shutdown_within(
    service: BackgroundTaskService,
    seconds: float,
    *,
    exception: BaseException | None = None,
) -> bool:
    """Whether ``__aexit__`` actually returns, without using it as its own oracle.

    `asyncio.wait_for` is no use here: `collect` swallows CancelledError per task,
    so a timeout's cancellation makes a wedged teardown complete and look clean.
    Running it as a task and watching `done()` observes the hang instead of
    curing it."""
    shutdown = asyncio.ensure_future(
        service.__aexit__(
            type(exception) if exception else None,
            exception,
            None,
        )
    )

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if shutdown.done():
            await shutdown
            return True
        await asyncio.sleep(0.05)

    shutdown.cancel()
    return False


async def test_that_a_clean_shutdown_cancels_a_task_that_never_ends_on_its_own(
    logger: Logger,
) -> None:
    # A long-running task — a log streamer, a poller — exits only when cancelled.
    # Shutting down without an exception has to cancel it, or `collect(force=True)`
    # waits on it forever and the process never gets past teardown.
    service = BackgroundTaskService(logger)
    task_started = asyncio.Event()

    async def run_forever() -> None:
        task_started.set()
        await asyncio.Future[None]()

    await service.__aenter__()
    task = await service.start(run_forever(), tag="runs-forever")
    await asyncio.wait_for(task_started.wait(), timeout=1)

    try:
        assert await _shutdown_within(service, 3.0)
    finally:
        task.cancel()


async def test_that_a_clean_shutdown_lets_a_cancelled_task_finish_unwinding(
    logger: Logger,
) -> None:
    # Cancelling is not enough on its own: the point of awaiting each task is that
    # its cleanup runs before teardown continues. A log streamer releases its
    # subscribers there, so cancelling without awaiting would strand them.
    service = BackgroundTaskService(logger)
    task_started = asyncio.Event()
    unwound = asyncio.Event()

    async def run_forever() -> None:
        try:
            task_started.set()
            await asyncio.Future[None]()
        finally:
            unwound.set()

    await service.__aenter__()
    task = await service.start(run_forever(), tag="runs-forever")
    await asyncio.wait_for(task_started.wait(), timeout=1)

    try:
        assert await _shutdown_within(service, 3.0)
        assert unwound.is_set()
    finally:
        task.cancel()


async def test_that_a_failed_shutdown_still_cancels_a_task_that_never_ends(
    logger: Logger,
) -> None:
    # The error path cancelled before this change and must keep doing so.
    service = BackgroundTaskService(logger)
    task_started = asyncio.Event()

    async def run_forever() -> None:
        task_started.set()
        await asyncio.Future[None]()

    await service.__aenter__()
    task = await service.start(run_forever(), tag="runs-forever")
    await asyncio.wait_for(task_started.wait(), timeout=1)

    try:
        assert await _shutdown_within(service, 3.0, exception=RuntimeError("shutting down"))
    finally:
        task.cancel()
