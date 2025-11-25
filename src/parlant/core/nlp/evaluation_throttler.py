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
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping


from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.loggers import Logger
from parlant.core.nlp.embedding import Embedder, EmbeddingResult
from parlant.core.nlp.generation import SchematicGenerator, SchematicGenerationResult, T
from parlant.core.nlp.tokenization import EstimatingTokenizer


@dataclass(frozen=True)
class RateLimits:
    """Flexible rate limits configuration inspired by OpenAI's approach.

    Users can specify any combination of limits they need:
    - RPM (requests per minute)
    - TPM (tokens per minute)
    - RPH (requests per hour)
    - RPD (requests per day)
    - RPM_MONTHLY (requests per month)
    - TPD (tokens per day)
    - TPM_MONTHLY (tokens per month)
    - MAX_CONCURRENT (maximum concurrent requests)

    None values mean no limit for that metric.
    """

    # Requests per minute
    rpm: int | None = None

    # Tokens per minute
    tpm: int | None = None

    # Requests per hour
    rph: int | None = None

    # Requests per day
    rpd: int | None = None

    # Requests per month (30 days)
    rpm_monthly: int | None = None

    # Tokens per day
    tpd: int | None = None

    # Tokens per month (30 days)
    tpm_monthly: int | None = None

    # Maximum concurrent requests
    max_concurrent: int | None = None

    @classmethod
    def create_custom(
        cls,
        rpm: int | None = None,
        tpm: int | None = None,
        rph: int | None = None,
        rpd: int | None = None,
        rpm_monthly: int | None = None,
        tpd: int | None = None,
        tpm_monthly: int | None = None,
        max_concurrent: int | None = None,
    ) -> "RateLimits":
        """Create custom rate limits with specified parameters."""
        return cls(
            rpm=rpm,
            tpm=tpm,
            rph=rph,
            rpd=rpd,
            rpm_monthly=rpm_monthly,
            tpd=tpd,
            tpm_monthly=tpm_monthly,
            max_concurrent=max_concurrent,
        )

    @classmethod
    def free_tier(cls) -> "RateLimits":
        """Predefined free tier limits (similar to OpenAI free tier)."""
        return cls(
            rpm=3,
            tpm=40_000,
            rph=200,
            rpd=1_000,
            max_concurrent=1,
        )

    @classmethod
    def basic_tier(cls) -> "RateLimits":
        """Predefined basic tier limits."""
        return cls(
            rpm=60,
            tpm=60_000,
            rph=3_600,
            rpd=50_000,
            max_concurrent=5,
        )

    @classmethod
    def premium_tier(cls) -> "RateLimits":
        """Predefined premium tier limits."""
        return cls(
            rpm=5_000,
            tpm=300_000,
            rph=300_000,
            rpd=1_000_000,
            max_concurrent=20,
        )

    @classmethod
    def enterprise_tier(cls) -> "RateLimits":
        """Predefined enterprise tier limits."""
        return cls(
            rpm=10_000,
            tpm=2_000_000,
            rph=600_000,
            rpd=5_000_000,
            max_concurrent=100,
        )


class ThrottlingException(Exception):
    """Exception raised when evaluation requests are throttled due to rate limits."""

    def __init__(self, message: str, rate_limits: RateLimits, retry_after: float | None = None):
        super().__init__(message)
        self.rate_limits = rate_limits
        self.retry_after = retry_after


class ServiceExhaustedException(Exception):
    """Exception raised when the underlying NLP service is exhausted or unavailable."""

    def __init__(self, message: str, service_id: str, retry_after: float | None = None):
        super().__init__(message)
        self.service_id = service_id
        self.retry_after = retry_after


class SlidingWindow:
    """Efficient sliding window for tracking requests and tokens in a time period."""

    def __init__(self, window_seconds: float):
        self.window_seconds = window_seconds
        self.requests: deque[tuple[float, int]] = deque()  # (timestamp, token_count)
        self.request_count = 0
        self.token_count = 0

    def add_request(self, timestamp: float, tokens: int) -> None:
        """Add a request to the sliding window."""
        self._expire_old_requests(timestamp)
        self.requests.append((timestamp, tokens))
        self.request_count += 1
        self.token_count += tokens

    def can_add_request(
        self, timestamp: float, tokens: int, max_requests: int | None, max_tokens: int | None
    ) -> tuple[bool, float | None]:
        """Check if a request can be added without exceeding limits."""
        self._expire_old_requests(timestamp)

        # Check request limit
        if max_requests is not None and self.request_count >= max_requests:
            if self.requests:
                oldest_timestamp = self.requests[0][0]
                retry_after = self.window_seconds - (timestamp - oldest_timestamp)
                return False, max(retry_after, 0.1)
            return False, self.window_seconds

        # Check token limit
        if max_tokens is not None and self.token_count + tokens > max_tokens:
            if self.requests:
                oldest_timestamp = self.requests[0][0]
                retry_after = self.window_seconds - (timestamp - oldest_timestamp)
                return False, max(retry_after, 0.1)
            return False, self.window_seconds

        return True, None

    def _expire_old_requests(self, current_time: float) -> None:
        """Remove requests outside the sliding window."""
        cutoff = current_time - self.window_seconds
        while self.requests and self.requests[0][0] <= cutoff:
            _, tokens = self.requests.popleft()
            self.request_count -= 1
            self.token_count -= tokens


class RateLimitTracker:
    """Efficient sliding window-based rate limit tracker."""

    def __init__(self) -> None:
        self._minute_window = SlidingWindow(60.0)  # 1 minute
        self._hour_window = SlidingWindow(3600.0)  # 1 hour
        self._day_window = SlidingWindow(86400.0)  # 1 day
        self._month_window = SlidingWindow(30 * 86400.0)  # 30 days
        self._concurrent_count = 0
        self._lock = asyncio.Lock()

    async def can_make_request(
        self, limits: RateLimits, estimated_tokens: int = 1
    ) -> tuple[bool, float | None]:
        """Check if a request can be made within the current limits.

        Args:
            limits: The rate limits to check against
            estimated_tokens: Estimated number of tokens for this request

        Returns:
            tuple: (can_make_request, retry_after_seconds)
        """
        async with self._lock:
            current_time = time.time()

            # Check concurrent requests
            if (
                limits.max_concurrent is not None
                and self._concurrent_count >= limits.max_concurrent
            ):
                return False, 1.0  # Retry after 1 second

            # Check minute limits (RPM and TPM)
            if limits.rpm is not None or limits.tpm is not None:
                can_add, retry_after = self._minute_window.can_add_request(
                    current_time, estimated_tokens, limits.rpm, limits.tpm
                )
                if not can_add:
                    return False, retry_after

            # Check hour limits (RPH)
            if limits.rph is not None:
                can_add, retry_after = self._hour_window.can_add_request(
                    current_time, estimated_tokens, limits.rph, None
                )
                if not can_add:
                    return False, retry_after

            # Check day limits (RPD and TPD)
            if limits.rpd is not None or limits.tpd is not None:
                can_add, retry_after = self._day_window.can_add_request(
                    current_time, estimated_tokens, limits.rpd, limits.tpd
                )
                if not can_add:
                    return False, retry_after

            # Check monthly limits (RPM_monthly and TPM_monthly)
            if limits.rpm_monthly is not None or limits.tpm_monthly is not None:
                can_add, retry_after = self._month_window.can_add_request(
                    current_time, estimated_tokens, limits.rpm_monthly, limits.tpm_monthly
                )
                if not can_add:
                    return False, retry_after

            return True, None

    async def record_request_start(self, tokens: int = 1) -> None:
        """Record the start of a request with token count."""
        async with self._lock:
            current_time = time.time()

            # Add to all relevant sliding windows
            self._minute_window.add_request(current_time, tokens)
            self._hour_window.add_request(current_time, tokens)
            self._day_window.add_request(current_time, tokens)
            self._month_window.add_request(current_time, tokens)

            self._concurrent_count += 1

    async def record_request_end(self) -> None:
        """Record the end of a request."""
        async with self._lock:
            self._concurrent_count = max(0, self._concurrent_count - 1)


class UserProvider(ABC):
    """Abstract interface for providing user information."""

    @abstractmethod
    async def get_user_id(self, hints: Mapping[str, Any]) -> str | None:
        """Extract user ID from hints."""
        pass

    @abstractmethod
    async def get_user_limits(self, user_id: str) -> RateLimits:
        """Get rate limits for a specific user."""
        pass


class DefaultUserProvider(UserProvider):
    """Default user provider that uses hints to determine user limits."""

    def __init__(self, default_limits: RateLimits | None = None):
        self._default_limits = default_limits or RateLimits.free_tier()

    async def get_user_id(self, hints: Mapping[str, Any]) -> str | None:
        """Extract user ID from hints."""
        return hints.get("user_id") or hints.get("customer_id") or hints.get("session_id")

    async def get_user_limits(self, user_id: str) -> RateLimits:
        """Get rate limits for a specific user. Override this method to integrate with your user management system."""
        return self._default_limits


class EvaluationThrottler(SchematicGenerator[T]):
    """A throttling wrapper for SchematicGenerator that enforces flexible rate limits."""

    _shared_global_tracker: RateLimitTracker | None = None

    def __init__(
        self,
        generator: SchematicGenerator[T],
        user_provider: UserProvider,
        logger: Logger,
        global_limits: RateLimits | None = None,
        shared_global_tracker: RateLimitTracker | None = None,
    ) -> None:
        self._generator = generator
        self._user_provider = user_provider
        self._logger = logger
        self._global_limits = global_limits
        self._user_trackers: dict[str, RateLimitTracker] = {}

        # Use shared global tracker if provided, otherwise create a new one
        if shared_global_tracker is not None:
            self._global_tracker = shared_global_tracker
        elif global_limits is not None:
            # Create a class-level shared tracker if one doesn't exist
            if EvaluationThrottler._shared_global_tracker is None:
                EvaluationThrottler._shared_global_tracker = RateLimitTracker()
            self._global_tracker = EvaluationThrottler._shared_global_tracker
        else:
            self._global_tracker = RateLimitTracker()
        self._lock = asyncio.Lock()

    async def _get_user_tracker(self, user_id: str) -> RateLimitTracker:
        """Get or create a rate limit tracker for a user."""
        async with self._lock:
            if user_id not in self._user_trackers:
                self._user_trackers[user_id] = RateLimitTracker()
            return self._user_trackers[user_id]

    async def _estimate_tokens(self, prompt: str | PromptBuilder) -> int:
        """Estimate the number of tokens in a prompt."""
        try:
            if isinstance(prompt, PromptBuilder):
                prompt_text = str(prompt)
            else:
                prompt_text = prompt
            return await self._generator.tokenizer.estimate_token_count(prompt_text)
        except Exception:
            # Fallback to character count / 4 as rough estimate
            prompt_text = str(prompt)
            return len(prompt_text) // 4

    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        user_id = await self._user_provider.get_user_id(hints)

        if not user_id:
            self._logger.warning("No user ID found in hints, using anonymous user")
            user_id = "anonymous"

        # Get user limits
        user_limits = await self._user_provider.get_user_limits(user_id)
        user_tracker = await self._get_user_tracker(user_id)

        # Estimate tokens for this request
        estimated_tokens = await self._estimate_tokens(prompt)

        # Check user limits
        can_proceed, retry_after = await user_tracker.can_make_request(
            user_limits, estimated_tokens
        )
        if not can_proceed:
            raise ThrottlingException(
                f"User {user_id} rate limit exceeded",
                user_limits,
                retry_after,
            )

        # Check global limits if configured
        if self._global_limits:
            can_proceed, retry_after = await self._global_tracker.can_make_request(
                self._global_limits, estimated_tokens
            )
            if not can_proceed:
                raise ServiceExhaustedException(
                    "Global service rate limit exceeded",
                    self._generator.id,
                    retry_after,
                )

        # Record request start
        await user_tracker.record_request_start(estimated_tokens)
        if self._global_limits:
            await self._global_tracker.record_request_start(estimated_tokens)

        try:
            # Make the actual request
            result = await self._generator.generate(prompt, hints)
            return result
        finally:
            # Record request end
            await user_tracker.record_request_end()
            if self._global_limits:
                await self._global_tracker.record_request_end()

    @property
    def schema(self) -> type[T]:
        """Return the schema from the wrapped generator."""
        return self._generator.schema

    @property
    def id(self) -> str:
        return f"throttled({self._generator.id})"

    @property
    def max_tokens(self) -> int:
        return self._generator.max_tokens

    @property
    def tokenizer(self) -> EstimatingTokenizer:
        return self._generator.tokenizer


class EmbedderThrottler(Embedder):
    """A throttling wrapper for Embedder that enforces flexible rate limits."""

    def __init__(
        self,
        embedder: Embedder,
        user_provider: UserProvider,
        logger: Logger,
        global_limits: RateLimits | None = None,
    ) -> None:
        self._embedder = embedder
        self._user_provider = user_provider
        self._logger = logger
        self._global_limits = global_limits
        self._user_trackers: dict[str, RateLimitTracker] = {}
        self._global_tracker = RateLimitTracker()
        self._lock = asyncio.Lock()

    async def _get_user_tracker(self, user_id: str) -> RateLimitTracker:
        """Get or create a rate limit tracker for a user."""
        async with self._lock:
            if user_id not in self._user_trackers:
                self._user_trackers[user_id] = RateLimitTracker()
            return self._user_trackers[user_id]

    async def _estimate_tokens(self, texts: list[str]) -> int:
        """Estimate the total number of tokens for embedding texts."""
        try:
            total_chars = sum(len(text) for text in texts)
            # Rough estimate: 1 token per 4 characters
            return total_chars // 4
        except Exception:
            # Fallback
            return len(texts) * 100

    async def embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        user_id = await self._user_provider.get_user_id(hints)

        if not user_id:
            self._logger.warning("No user ID found in hints, using anonymous user")
            user_id = "anonymous"

        # Get user limits
        user_limits = await self._user_provider.get_user_limits(user_id)
        user_tracker = await self._get_user_tracker(user_id)

        # Estimate tokens for this request
        estimated_tokens = await self._estimate_tokens(texts)

        # Check user limits
        can_proceed, retry_after = await user_tracker.can_make_request(
            user_limits, estimated_tokens
        )
        if not can_proceed:
            raise ThrottlingException(
                f"User {user_id} rate limit exceeded",
                user_limits,
                retry_after,
            )

        # Check global limits if configured
        if self._global_limits:
            can_proceed, retry_after = await self._global_tracker.can_make_request(
                self._global_limits, estimated_tokens
            )
            if not can_proceed:
                raise ServiceExhaustedException(
                    "Global service rate limit exceeded",
                    self._embedder.id,
                    retry_after,
                )

        # Record request start
        await user_tracker.record_request_start(estimated_tokens)
        if self._global_limits:
            await self._global_tracker.record_request_start(estimated_tokens)

        try:
            # Make the actual request
            result = await self._embedder.embed(texts, hints)
            return result
        finally:
            # Record request end
            await user_tracker.record_request_end()
            if self._global_limits:
                await self._global_tracker.record_request_end()

    @property
    def id(self) -> str:
        return f"throttled({self._embedder.id})"

    @property
    def max_tokens(self) -> int:
        return self._embedder.max_tokens

    @property
    def tokenizer(self) -> EstimatingTokenizer:
        return self._embedder.tokenizer

    @property
    def dimensions(self) -> int:
        return self._embedder.dimensions


# Convenience functions for creating throttlers with predefined limits


def create_throttled_generator(
    generator: SchematicGenerator[T],
    logger: Logger,
    rate_limits: RateLimits | None = None,
    global_limits: RateLimits | None = None,
) -> EvaluationThrottler[T]:
    """Create a throttled generator with default user provider."""
    user_provider = DefaultUserProvider(rate_limits or RateLimits.free_tier())
    return EvaluationThrottler(generator, user_provider, logger, global_limits)


def create_throttled_embedder(
    embedder: Embedder,
    logger: Logger,
    rate_limits: RateLimits | None = None,
    global_limits: RateLimits | None = None,
) -> EmbedderThrottler:
    """Create a throttled embedder with default user provider."""
    user_provider = DefaultUserProvider(rate_limits or RateLimits.free_tier())
    return EmbedderThrottler(embedder, user_provider, logger, global_limits)
