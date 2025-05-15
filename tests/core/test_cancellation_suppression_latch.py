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

import asyncio

from pytest import raises

from parlant.core.common import CancellationSuppressionLatch


async def test_that_function_returns_value_when_cancelled_error_is_raised_latch_enabled() -> None:
    async def get_value(latch: CancellationSuppressionLatch) -> str:
        latch.enable()
        await asyncio.sleep(0.2)
        return "Task completed"

    with CancellationSuppressionLatch() as latch:
        task = asyncio.create_task(get_value(latch))
        await asyncio.sleep(0.1)
        task.cancel()

        result = await task
        await asyncio.sleep(0.3)
        assert result == "Task completed"


async def test_that_function_raises_cancelled_error_when_cancelled_before_latch_is_enabled() -> (
    None
):
    async def get_value() -> str:
        await asyncio.sleep(0.3)
        return "Task completed"

    with CancellationSuppressionLatch():
        task = asyncio.create_task(get_value())
        await asyncio.sleep(0.1)
        task.cancel()

        with raises(asyncio.CancelledError):
            await task
