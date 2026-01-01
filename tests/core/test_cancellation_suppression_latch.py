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

from __future__ import annotations
import asyncio
import pytest

from parlant.core.async_utils import (
    CancellationSuppressionLatch,
    latched_shield,
)


async def test_latch_behavior_with_no_cancellation() -> None:
    """Test latch behavior when task is cancelled but latch suppresses it."""
    execution_log = []

    async def shielded_task(suppression_latch: CancellationSuppressionLatch[str]) -> str:
        execution_log.append("started")
        suppression_latch.enable()
        await asyncio.sleep(0.1)  # Simulate some work
        execution_log.append("finished")
        return "done"

    async def test_task() -> str:
        return await latched_shield(shielded_task)

    t = asyncio.create_task(test_task())
    assert (await t) == "done"

    # When latch is enabled and cancellation is suppressed, ALL code should execute
    assert execution_log == ["started", "finished"]


async def test_latch_behavior_with_cancellation_after_suppression() -> None:
    """Test latch behavior when task is cancelled but latch suppresses it."""
    ready_to_cancel = asyncio.Event()
    cancelled = asyncio.Event()

    execution_log = []

    async def shielded_task(suppression_latch: CancellationSuppressionLatch[None]) -> None:
        execution_log.append("started")
        suppression_latch.enable()
        ready_to_cancel.set()  # Trigger cancellation suppression
        await cancelled.wait()
        await asyncio.sleep(0.1)  # Simulate some work
        execution_log.append("finished")

    async def test_task() -> None:
        await latched_shield(shielded_task)

    t = asyncio.create_task(test_task())

    # Wait for shielded task to start
    await ready_to_cancel.wait()
    # Cancel it
    t.cancel()
    cancelled.set()
    # Wait for task to complete
    await t

    # When latch is enabled and cancellation is suppressed, ALL code should execute
    assert execution_log == ["started", "finished"]


async def test_latch_behavior_with_cancellation_before_suppression() -> None:
    """Test latch behavior when task is cancelled but latch suppresses it."""
    ready_to_cancel = asyncio.Event()
    cancelled = asyncio.Event()
    cancellation_raised_at_expected_point = False

    execution_log = []

    async def shielded_task(suppression_latch: CancellationSuppressionLatch[None]) -> None:
        nonlocal cancellation_raised_at_expected_point

        execution_log.append("started")

        try:
            ready_to_cancel.set()  # Trigger cancellation suppression
            await cancelled.wait()
        except asyncio.CancelledError:
            cancellation_raised_at_expected_point = True
            raise

        suppression_latch.enable()
        await asyncio.sleep(0.1)  # Simulate some work
        execution_log.append("finished")

    async def test_task() -> None:
        await latched_shield(shielded_task)

    t = asyncio.create_task(test_task())

    # Wait for shielded task to start
    await ready_to_cancel.wait()
    # Cancel it
    t.cancel()
    cancelled.set()
    # Wait for task to complete
    with pytest.raises(asyncio.CancelledError):
        await t

    # When latch is enabled and cancellation is suppressed, ALL code should execute
    assert execution_log == ["started"]
    assert cancellation_raised_at_expected_point
