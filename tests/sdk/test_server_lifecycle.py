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

import asyncio
from contextlib import contextmanager
from typing import Iterator

import pytest

from parlant import sdk as p
from parlant.core.tracer import Tracer


class StartupFailure(Exception):
    pass


class SetupFailure(Exception):
    pass


class FakeProgress:
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        pass


class FakeTracer:
    @contextmanager
    def span(self, span_id: str, attributes: dict[str, str]) -> Iterator[None]:
        yield


class FakeContainer:
    def __getitem__(self, key: object) -> object:
        if key is Tracer:
            return FakeTracer()

        raise KeyError(key)


class FailingStartupContextManager:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        await asyncio.sleep(0)
        raise self.error


class RecordingStartupContextManager:
    def __init__(self) -> None:
        self.exc_type: type[BaseException] | None = None
        self.exc_value: BaseException | None = None
        self.closed = False

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object | None,
    ) -> None:
        self.exc_type = exc_type
        self.exc_value = exc_value
        self.closed = True


class FakeExitStack:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


async def test_that_server_startup_errors_are_not_masked_by_health_check_cancellation() -> None:
    server = p.Server()
    startup_error = StartupFailure("backend rejected tunnel auth")
    exit_stack = FakeExitStack()
    polling_started = asyncio.Event()

    async def process_evaluations() -> None:
        pass

    async def setup_retrievers() -> None:
        pass

    async def poll_health_endpoint() -> None:
        polling_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise

    server._creation_progress = FakeProgress()  # type: ignore[assignment]
    server._container = FakeContainer()  # type: ignore[assignment]
    server._startup_context_manager = FailingStartupContextManager(startup_error)  # type: ignore[assignment]
    server._exit_stack = exit_stack  # type: ignore[assignment]
    server._process_evaluations = process_evaluations  # type: ignore[method-assign]
    server._setup_retrievers = setup_retrievers  # type: ignore[method-assign]
    server._poll_health_endpoint = poll_health_endpoint  # type: ignore[method-assign]

    with pytest.raises(StartupFailure) as exc_info:
        await server.__aexit__(None, None, None)

    assert exc_info.value is startup_error
    assert polling_started.is_set()
    assert exit_stack.closed


async def test_that_pre_serving_setup_errors_close_the_startup_context() -> None:
    server = p.Server()
    setup_error = SetupFailure("indexing failed")
    startup_context = RecordingStartupContextManager()
    exit_stack = FakeExitStack()

    async def process_evaluations() -> None:
        raise setup_error

    async def setup_retrievers() -> None:
        pass

    server._creation_progress = FakeProgress()  # type: ignore[assignment]
    server._container = FakeContainer()  # type: ignore[assignment]
    server._startup_context_manager = startup_context  # type: ignore[assignment]
    server._exit_stack = exit_stack  # type: ignore[assignment]
    server._process_evaluations = process_evaluations  # type: ignore[method-assign]
    server._setup_retrievers = setup_retrievers  # type: ignore[method-assign]

    with pytest.raises(SetupFailure) as exc_info:
        await server.__aexit__(None, None, None)

    assert exc_info.value is setup_error
    assert startup_context.exc_type is SetupFailure
    assert startup_context.exc_value is setup_error
    assert startup_context.closed
    assert exit_stack.closed
