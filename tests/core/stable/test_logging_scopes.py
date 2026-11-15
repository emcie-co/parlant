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

import pytest

from parlant.core.loggers import TracingLogger
from parlant.core.tracer import LocalTracer


def test_that_logging_scopes_restore_the_previous_scope_after_normal_exit() -> None:
    logger = TracingLogger(LocalTracer())

    with logger.scope("outer"):
        assert logger.current_scope == "[outer]"
        with logger.scope("inner"):
            assert logger.current_scope == "[outer][inner]"
        assert logger.current_scope == "[outer]"

    assert logger.current_scope == ""


def test_that_a_failed_inner_scope_restores_the_outer_logging_scope() -> None:
    logger = TracingLogger(LocalTracer())
    failure = RuntimeError("inner operation failed")

    with logger.scope("outer"):
        with pytest.raises(RuntimeError) as caught:
            with logger.scope("inner"):
                raise failure

        assert caught.value is failure
        assert logger.current_scope == "[outer]"

    assert logger.current_scope == ""


def test_that_failed_nested_logging_scopes_restore_the_original_scope() -> None:
    logger = TracingLogger(LocalTracer())
    failure = RuntimeError("nested operation failed")

    with pytest.raises(RuntimeError) as caught:
        with logger.scope("outer"):
            with logger.scope("inner"):
                raise failure

    assert caught.value is failure
    assert logger.current_scope == ""


async def test_that_cancelled_logging_scopes_restore_the_original_scope() -> None:
    logger = TracingLogger(LocalTracer())
    entered = asyncio.Event()
    never_released = asyncio.Event()
    after_cancellation: list[str] = []

    async def operation() -> None:
        try:
            with logger.scope("cancelled"):
                entered.set()
                await never_released.wait()
        finally:
            # Observe cleanup in the task that owned the scope, before its
            # ContextVar context is discarded when the cancelled task exits.
            after_cancellation.append(logger.current_scope)

    task = asyncio.create_task(operation())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert after_cancellation == [""]
    assert logger.current_scope == ""


async def test_that_concurrent_tasks_keep_their_logging_scopes_isolated() -> None:
    logger = TracingLogger(LocalTracer())
    first_entered = asyncio.Event()
    second_entered = asyncio.Event()

    async def first() -> None:
        with logger.scope("first"):
            first_entered.set()
            await second_entered.wait()
            assert logger.current_scope == "[first]"
        assert logger.current_scope == ""

    async def second() -> None:
        await first_entered.wait()
        with logger.scope("second"):
            second_entered.set()
            assert logger.current_scope == "[second]"
        assert logger.current_scope == ""

    await asyncio.gather(first(), second())
    assert logger.current_scope == ""
