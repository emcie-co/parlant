# Copyright 2025 Emcie Co Ltd.
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

from exceptiongroup import ExceptionGroup
import pytest
from contextlib import asynccontextmanager
from fastapi import FastAPI
from typing import AsyncIterator, List

from parlant.api.app import APIConfigurationSteps, AppWrapper
from lagom import Container


class MockLogger:
    def __init__(self) -> None:
        self.logs: List[str] = []

    def log(self, message: str) -> None:
        self.logs.append(message)


@pytest.fixture
def container() -> Container:
    c = Container()
    c[MockLogger] = MockLogger()
    return c


@pytest.fixture
def app() -> FastAPI:
    return FastAPI()


@asynccontextmanager
async def first_step(app: FastAPI, container: Container) -> AsyncIterator[FastAPI]:
    container[MockLogger].log("First step setup")
    yield app
    container[MockLogger].log("First step teardown")


@asynccontextmanager
async def second_step(app: FastAPI, container: Container) -> AsyncIterator[FastAPI]:
    container[MockLogger].log("Second step setup")
    yield app
    container[MockLogger].log("Second step teardown")


@asynccontextmanager
async def error_teardown_step(app: FastAPI, container: Container) -> AsyncIterator[FastAPI]:
    container[MockLogger].log("Error step setup")
    yield app
    raise RuntimeError("Error during teardown")


@asynccontextmanager
async def error_setup_step(app: FastAPI, container: Container) -> AsyncIterator[FastAPI]:
    raise RuntimeError("Error during setup")
    yield app
    container[MockLogger].log("Error step teardown")


@pytest.mark.asyncio
async def test_configuration_step_execution_order(app: FastAPI, container: Container) -> None:
    """Test that configuration steps are executed in the correct order."""
    container[APIConfigurationSteps] = [first_step, second_step]
    app_wrapper = AppWrapper(app, container)

    async with app_wrapper:
        assert container[MockLogger].logs == ["First step setup", "Second step setup"]

    assert container[MockLogger].logs == [
        "First step setup",
        "Second step setup",
        "Second step teardown",
        "First step teardown",
    ]


@pytest.mark.asyncio
async def test_error_handling_during_setup(app: FastAPI, container: Container) -> None:
    container[APIConfigurationSteps] = [first_step, error_setup_step, second_step]
    app_wrapper = AppWrapper(app, container)

    with pytest.raises(RuntimeError):
        async with app_wrapper:
            pass

    assert "First step setup" in container[MockLogger].logs
    assert "First step teardown" in container[MockLogger].logs


@pytest.mark.asyncio
async def test_error_handling_during_teardown(app: FastAPI, container: Container) -> None:
    """Test that errors during teardown don't prevent other steps from tearing down."""
    container[APIConfigurationSteps] = [first_step, error_teardown_step, second_step]
    app_wrapper = AppWrapper(app, container)

    with pytest.raises(ExceptionGroup):
        async with app_wrapper:
            pass

    assert "Second step teardown" in container[MockLogger].logs
    assert "First step teardown" in container[MockLogger].logs
