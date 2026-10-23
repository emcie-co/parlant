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

from abc import ABC, abstractmethod
import asyncio
from collections import deque
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Generic, TypeVar

from parlant.core.common import DefaultBaseModel
from parlant.core.engines.compass.prompt_builder import PromptBuilder
from parlant.core.nlp.generation import (
    SchematicGenerationResult,
    SchematicGenerator,
)
from parlant.core.nlp.react import (
    Message,
    ReactGenerator,
    ReactGeneratorHints,
    ReasoningConfig,
    StreamEvent,
    ToolChoice,
    ToolMessageDeserializer,
    ToolMessageSerializer,
    ToolSpec,
    TurnBuilder,
    Usage,
)
from parlant.core.nlp.tokenization import EstimatingTokenizer


class RateLimitError(ValueError):
    """Base class for invalid or incompatible model rate-limit configuration."""


class RateLimitConflictError(RateLimitError):
    """Raised when one named bucket is configured with incompatible limits."""


@dataclass(frozen=True)
class RateLimit:
    """Admission limits for model requests selected through one model route.

    ``max_concurrency`` limits requests that are currently in flight.
    ``requests_per_minute`` paces request starts evenly across a minute. When
    both are set, a request must satisfy both constraints before it is admitted.
    A named ``bucket`` deliberately shares capacity across route bindings.
    """

    max_concurrency: int | None = None
    requests_per_minute: float | None = None
    bucket: str | None = None

    def __post_init__(self) -> None:
        if self.max_concurrency is None and self.requests_per_minute is None:
            raise RateLimitError("RateLimit requires max_concurrency, requests_per_minute, or both")
        if self.max_concurrency is not None and (
            isinstance(self.max_concurrency, bool)
            or not isinstance(self.max_concurrency, int)
            or self.max_concurrency <= 0
        ):
            raise RateLimitError("max_concurrency must be a positive integer")
        if self.requests_per_minute is not None and (
            isinstance(self.requests_per_minute, bool)
            or not isinstance(self.requests_per_minute, (int, float))
            or not math.isfinite(self.requests_per_minute)
            or self.requests_per_minute <= 0
        ):
            raise RateLimitError("requests_per_minute must be a positive finite number")
        if self.bucket is not None and (
            not isinstance(self.bucket, str) or not self.bucket.strip()
        ):
            raise RateLimitError("bucket must be a non-empty string")


class RateLimitPermit(ABC):
    """One admitted request. Releasing it frees concurrency capacity."""

    @abstractmethod
    async def release(self) -> None: ...


class RateLimiter(ABC):
    """DI port for acquiring capacity from a named model-request bucket."""

    @abstractmethod
    async def acquire(self, *, bucket: str, limit: RateLimit) -> RateLimitPermit: ...


@dataclass
class _TransientBucket:
    limit: RateLimit
    condition: asyncio.Condition
    waiters: deque[object]
    active_requests: int = 0
    next_start_at: float = 0.0


class _TransientPermit(RateLimitPermit):
    def __init__(self, owner: TransientRateLimiter, bucket: str) -> None:
        self._owner = owner
        self._bucket = bucket
        self._released = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._owner._release(self._bucket)


class TransientRateLimiter(RateLimiter):
    """Process-local FIFO rate limiter.

    State intentionally disappears when the process exits. Applications that
    need quotas shared across workers or preserved across restarts can replace
    this implementation through DI without changing model route configuration.
    """

    def __init__(self) -> None:
        self._buckets: dict[str, _TransientBucket] = {}
        self._buckets_lock = asyncio.Lock()

    async def acquire(self, *, bucket: str, limit: RateLimit) -> RateLimitPermit:
        state = await self._get_bucket(bucket, limit)
        waiter = object()

        async with state.condition:
            state.waiters.append(waiter)
            try:
                while True:
                    now = asyncio.get_running_loop().time()
                    concurrency_available = (
                        limit.max_concurrency is None
                        or state.active_requests < limit.max_concurrency
                    )
                    delay = max(0.0, state.next_start_at - now)
                    if state.waiters[0] is waiter and concurrency_available and delay == 0.0:
                        state.waiters.popleft()
                        state.active_requests += 1
                        if limit.requests_per_minute is not None:
                            state.next_start_at = now + 60.0 / limit.requests_per_minute
                        state.condition.notify_all()
                        return _TransientPermit(self, bucket)

                    try:
                        if delay > 0.0 and state.waiters[0] is waiter:
                            await asyncio.wait_for(state.condition.wait(), timeout=delay)
                        else:
                            await state.condition.wait()
                    except asyncio.TimeoutError:
                        pass
            except BaseException:
                try:
                    state.waiters.remove(waiter)
                except ValueError:
                    pass
                state.condition.notify_all()
                raise

    async def _get_bucket(self, bucket: str, limit: RateLimit) -> _TransientBucket:
        async with self._buckets_lock:
            if existing := self._buckets.get(bucket):
                if existing.limit != limit:
                    raise RateLimitConflictError(
                        f"Rate-limit bucket '{bucket}' is already configured with different limits"
                    )
                return existing

            state = _TransientBucket(
                limit=limit,
                condition=asyncio.Condition(),
                waiters=deque(),
            )
            self._buckets[bucket] = state
            return state

    async def _release(self, bucket: str) -> None:
        state = self._buckets[bucket]
        async with state.condition:
            state.active_requests -= 1
            state.condition.notify_all()


SchemaT = TypeVar("SchemaT", bound=DefaultBaseModel)


class _RateLimitedSchematicGenerator(SchematicGenerator[SchemaT], Generic[SchemaT]):
    def __init__(
        self,
        wrapped: SchematicGenerator[SchemaT],
        limiter: RateLimiter,
        limit: RateLimit,
        bucket: str,
    ) -> None:
        self._wrapped = wrapped
        self._limiter = limiter
        self._limit = limit
        self._bucket = bucket

    @property
    def schema(self) -> type[SchemaT]:
        return self._wrapped.schema

    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[SchemaT]:
        permit = await self._limiter.acquire(bucket=self._bucket, limit=self._limit)
        try:
            return await self._wrapped.generate(prompt, hints)
        finally:
            await permit.release()

    @property
    def id(self) -> str:
        return self._wrapped.id

    @property
    def max_tokens(self) -> int:
        return self._wrapped.max_tokens

    @property
    def tokenizer(self) -> EstimatingTokenizer:
        return self._wrapped.tokenizer


class _RateLimitedReactGenerator(ReactGenerator):
    """Wraps the provider seam so hedged ReAct attempts consume separate permits."""

    def __init__(
        self,
        wrapped: ReactGenerator,
        limiter: RateLimiter,
        limit: RateLimit,
        bucket: str,
    ) -> None:
        super().__init__(
            model=wrapped.model,
            cache=wrapped.cache,
            usage_reporter=wrapped.usage_reporter,
        )
        self._wrapped = wrapped
        self._limiter = limiter
        self._limit = limit
        self._bucket = bucket

    @property
    def provider_name(self) -> str:
        return self._wrapped.provider_name

    def _resolve_model(self, hints: ReactGeneratorHints) -> str:
        return self._wrapped.resolve_model(hints)

    def serialize_tool_messages(
        self,
        messages: Sequence[Message],
        serializer: ToolMessageSerializer,
    ) -> None:
        self._wrapped.serialize_tool_messages(messages, serializer)

    def deserialize_tool_messages(
        self,
        deserializer: ToolMessageDeserializer,
        *,
        model: str | None = None,
    ) -> Sequence[Message] | None:
        return self._wrapped.deserialize_tool_messages(deserializer, model=model)

    def _encode(
        self,
        history: Sequence[Message],
        tools: Sequence[ToolSpec],
        tool_choice: ToolChoice,
        *,
        reasoning: ReasoningConfig,
        hints: ReactGeneratorHints = {},
    ) -> Any:
        return self._wrapped._encode(
            history,
            tools,
            tool_choice,
            reasoning=reasoning,
            hints=hints,
        )

    async def _raw_stream(self, request: Any) -> AsyncIterator[Any]:
        permit = await self._limiter.acquire(bucket=self._bucket, limit=self._limit)
        try:
            async for event in self._wrapped._raw_stream(request):
                yield event
        finally:
            await permit.release()

    def _decode(self, raw_event: Any, builder: TurnBuilder) -> Sequence[StreamEvent]:
        return self._wrapped._decode(raw_event, builder)

    async def _prefill(self, request: Any) -> Usage:
        permit = await self._limiter.acquire(bucket=self._bucket, limit=self._limit)
        try:
            return await self._wrapped._prefill(request)
        finally:
            await permit.release()

    async def _should_prefill(
        self,
        history: Sequence[Message],
        tools: Sequence[ToolSpec],
        hints: ReactGeneratorHints,
        reasoning: ReasoningConfig | None = None,
    ) -> bool:
        return await self._wrapped._should_prefill(history, tools, hints, reasoning)


def rate_limit_schematic_generator(
    generator: SchematicGenerator[SchemaT],
    *,
    limiter: RateLimiter,
    limit: RateLimit,
    bucket: str,
) -> SchematicGenerator[SchemaT]:
    return _RateLimitedSchematicGenerator(generator, limiter, limit, bucket)


def rate_limit_react_generator(
    generator: ReactGenerator,
    *,
    limiter: RateLimiter,
    limit: RateLimit,
    bucket: str,
) -> ReactGenerator:
    return _RateLimitedReactGenerator(generator, limiter, limit, bucket)
