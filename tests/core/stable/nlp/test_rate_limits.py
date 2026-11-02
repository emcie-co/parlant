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

from parlant.core.nlp.rate_limits import (
    RateLimit,
    RateLimitConflictError,
    RateLimitError,
    TransientRateLimiter,
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"max_concurrency": 0},
        {"max_concurrency": 1.5},
        {"requests_per_minute": 0},
        {"requests_per_minute": float("inf")},
        {"bucket": "shared"},
        {"max_concurrency": 1, "bucket": "  "},
    ],
)
def test_that_invalid_rate_limits_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(RateLimitError):
        RateLimit(**kwargs)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_that_concurrency_admissions_are_fifo() -> None:
    limiter = TransientRateLimiter()
    limit = RateLimit(max_concurrency=1)
    first = await limiter.acquire(bucket="test", limit=limit)

    second_task = asyncio.create_task(limiter.acquire(bucket="test", limit=limit))
    third_task = asyncio.create_task(limiter.acquire(bucket="test", limit=limit))
    await asyncio.sleep(0)
    assert not second_task.done()
    assert not third_task.done()

    await first.release()
    second = await asyncio.wait_for(second_task, timeout=1)
    assert not third_task.done()

    await second.release()
    third = await asyncio.wait_for(third_task, timeout=1)
    await third.release()


@pytest.mark.asyncio
async def test_that_cancelling_a_waiter_does_not_block_the_queue() -> None:
    limiter = TransientRateLimiter()
    limit = RateLimit(max_concurrency=1)
    first = await limiter.acquire(bucket="test", limit=limit)

    cancelled_task = asyncio.create_task(limiter.acquire(bucket="test", limit=limit))
    next_task = asyncio.create_task(limiter.acquire(bucket="test", limit=limit))
    await asyncio.sleep(0)
    cancelled_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_task

    await first.release()
    next_permit = await asyncio.wait_for(next_task, timeout=1)
    await next_permit.release()


@pytest.mark.asyncio
async def test_that_requests_per_minute_paces_request_starts() -> None:
    limiter = TransientRateLimiter()
    limit = RateLimit(requests_per_minute=6000)
    loop = asyncio.get_running_loop()

    first = await limiter.acquire(bucket="test", limit=limit)
    first_started = loop.time()
    await first.release()
    second = await limiter.acquire(bucket="test", limit=limit)
    second_started = loop.time()
    await second.release()

    assert second_started - first_started >= 0.007


@pytest.mark.asyncio
async def test_that_one_named_bucket_rejects_conflicting_limits() -> None:
    limiter = TransientRateLimiter()
    permit = await limiter.acquire(
        bucket="shared",
        limit=RateLimit(max_concurrency=1, bucket="shared"),
    )
    await permit.release()

    with pytest.raises(RateLimitConflictError, match="already configured"):
        await limiter.acquire(
            bucket="shared",
            limit=RateLimit(max_concurrency=2, bucket="shared"),
        )
