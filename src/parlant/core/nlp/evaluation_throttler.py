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
from dataclasses import dataclass
from typing import Any, Mapping, Optional
from typing_extensions import override

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
    rpm: Optional[int] = None

    # Tokens per minute
    tpm: Optional[int] = None

    # Requests per hour
    rph: Optional[int] = None

    # Requests per day
    rpd: Optional[int] = None

    # Requests per month (30 days)
    rpm_monthly: Optional[int] = None

    # Tokens per day
    tpd: Optional[int] = None

    # Tokens per month (30 days)
    tpm_monthly: Optional[int] = None

    # Maximum concurrent requests
    max_concurrent: Optional[int] = None

    @classmethod
    def create_custom(
        cls,
        rpm: Optional[int] = None,
        tpm: Optional[int] = None,
        rph: Optional[int] = None,
        rpd: Optional[int] = None,
        rpm_monthly: Optional[int] = None,
        tpd: Optional[int] = None,
        tpm_monthly: Optional[int] = None,
        max_concurrent: Optional[int] = None,
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

    def __init__(self, message: str, rate_limits: RateLimits, retry_after: Optional[float] = None):
        super().__init__(message)
        self.rate_limits = rate_limits
        self.retry_after = retry_after


class ServiceExhaustedException(Exception):
    """Exception raised when the underlying NLP service is exhausted or unavailable."""

    def __init__(self, message: str, service_id: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.service_id = service_id
        self.retry_after = retry_after


class RateLimitTracker:
    """Tracks rate limits for different time windows with flexible limits."""

    def __init__(self):
        self._requests: list[tuple[float, int]] = []  # (timestamp, token_count)
        self._concurrent_count = 0
        self._lock = asyncio.Lock()

    async def can_make_request(
        self, limits: RateLimits, estimated_tokens: int = 1
    ) -> tuple[bool, Optional[float]]:
        """Check if a request can be made within the current limits.

        Args:
            limits: The rate limits to check against
            estimated_tokens: Estimated number of tokens for this request

        Returns:
            tuple: (can_make_request, retry_after_seconds)
        """
        async with self._lock:
            current_time = time.time()

            # Clean up old requests
            self._cleanup_old_requests(current_time)

            # Check concurrent requests
            if (
                limits.max_concurrent is not None
                and self._concurrent_count >= limits.max_concurrent
            ):
                return False, 1.0  # Retry after 1 second

            # Check requests per minute
            if limits.rpm is not None:
                minute_requests = len([t for t, _ in self._requests if current_time - t <= 60])
                if minute_requests >= limits.rpm:
                    oldest_in_minute = min([t for t, _ in self._requests if current_time - t <= 60])
                    return False, 60 - (current_time - oldest_in_minute)

            # Check tokens per minute
            if limits.tpm is not None:
                minute_tokens = sum(
                    [tokens for t, tokens in self._requests if current_time - t <= 60]
                )
                if minute_tokens + estimated_tokens > limits.tpm:
                    oldest_in_minute = min([t for t, _ in self._requests if current_time - t <= 60])
                    return False, 60 - (current_time - oldest_in_minute)

            # Check requests per hour
            if limits.rph is not None:
                hour_requests = len([t for t, _ in self._requests if current_time - t <= 3600])
                if hour_requests >= limits.rph:
                    oldest_in_hour = min([t for t, _ in self._requests if current_time - t <= 3600])
                    return False, 3600 - (current_time - oldest_in_hour)

            # Check requests per day
            if limits.rpd is not None:
                day_requests = len([t for t, _ in self._requests if current_time - t <= 86400])
                if day_requests >= limits.rpd:
                    oldest_in_day = min([t for t, _ in self._requests if current_time - t <= 86400])
                    return False, 86400 - (current_time - oldest_in_day)

            # Check tokens per day
            if limits.tpd is not None:
                day_tokens = sum(
                    [tokens for t, tokens in self._requests if current_time - t <= 86400]
                )
                if day_tokens + estimated_tokens > limits.tpd:
                    oldest_in_day = min([t for t, _ in self._requests if current_time - t <= 86400])
                    return False, 86400 - (current_time - oldest_in_day)

            # Check requests per month (30 days)
            if limits.rpm_monthly is not None:
                month_seconds = 30 * 86400
                month_requests = len(
                    [t for t, _ in self._requests if current_time - t <= month_seconds]
                )
                if month_requests >= limits.rpm_monthly:
                    oldest_in_month = min(
                        [t for t, _ in self._requests if current_time - t <= month_seconds]
                    )
                    return False, month_seconds - (current_time - oldest_in_month)

            # Check tokens per month (30 days)
            if limits.tpm_monthly is not None:
                month_seconds = 30 * 86400
                month_tokens = sum(
                    [tokens for t, tokens in self._requests if current_time - t <= month_seconds]
                )
                if month_tokens + estimated_tokens > limits.tpm_monthly:
                    oldest_in_month = min(
                        [t for t, _ in self._requests if current_time - t <= month_seconds]
                    )
                    return False, month_seconds - (current_time - oldest_in_month)

            return True, None

    async def record_request_start(self, tokens: int = 1):
        """Record the start of a request with token count."""
        async with self._lock:
            current_time = time.time()
            self._requests.append((current_time, tokens))
            self._concurrent_count += 1

    async def record_request_end(self):
        """Record the end of a request."""
        async with self._lock:
            self._concurrent_count = max(0, self._concurrent_count - 1)

    def _cleanup_old_requests(self, current_time: float):
        """Remove requests older than 30 days to prevent memory leaks."""
        month_seconds = 30 * 86400
        cutoff = current_time - month_seconds
        self._requests = [(t, tokens) for t, tokens in self._requests if t > cutoff]


class UserProvider(ABC):
    """Abstract interface for providing user information."""

    @abstractmethod
    async def get_user_id(self, hints: Mapping[str, Any]) -> Optional[str]:
        """Extract user ID from hints."""
        pass

    @abstractmethod
    async def get_user_limits(self, user_id: str) -> RateLimits:
        """Get rate limits for a specific user."""
        pass


class DefaultUserProvider(UserProvider):
    """Default user provider that uses hints to determine user limits."""

    def __init__(self, default_limits: Optional[RateLimits] = None):
        self._default_limits = default_limits or RateLimits.free_tier()

    @override
    async def get_user_id(self, hints: Mapping[str, Any]) -> Optional[str]:
        """Extract user ID from hints."""
        return hints.get("user_id") or hints.get("customer_id") or hints.get("session_id")

    @override
    async def get_user_limits(self, user_id: str) -> RateLimits:
        """Get rate limits for a specific user. Override this method to integrate with your user management system."""
        return self._default_limits


class EvaluationThrottler(SchematicGenerator[T]):
    """A throttling wrapper for SchematicGenerator that enforces flexible rate limits."""

    def __init__(
        self,
        generator: SchematicGenerator[T],
        user_provider: UserProvider,
        logger: Logger,
        global_limits: Optional[RateLimits] = None,
    ):
        self._generator = generator
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

    async def _estimate_tokens(self, prompt: str | PromptBuilder) -> int:
        """Estimate the number of tokens in a prompt."""
        try:
            if isinstance(prompt, PromptBuilder):
                prompt_text = str(prompt)
            else:
                prompt_text = prompt
            return self._generator.tokenizer.estimate_tokens(prompt_text)
        except Exception:
            # Fallback to character count / 4 as rough estimate
            prompt_text = str(prompt)
            return len(prompt_text) // 4

    @override
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
    @override
    def id(self) -> str:
        return f"throttled({self._generator.id})"

    @property
    @override
    def max_tokens(self) -> int:
        return self._generator.max_tokens

    @property
    @override
    def tokenizer(self) -> EstimatingTokenizer:
        return self._generator.tokenizer


class EmbedderThrottler(Embedder):
    """A throttling wrapper for Embedder that enforces flexible rate limits."""

    def __init__(
        self,
        embedder: Embedder,
        user_provider: UserProvider,
        logger: Logger,
        global_limits: Optional[RateLimits] = None,
    ):
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

    @override
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
    @override
    def id(self) -> str:
        return f"throttled({self._embedder.id})"

    @property
    @override
    def max_tokens(self) -> int:
        return self._embedder.max_tokens

    @property
    @override
    def tokenizer(self) -> EstimatingTokenizer:
        return self._embedder.tokenizer

    @property
    @override
    def dimensions(self) -> int:
        return self._embedder.dimensions


# Convenience functions for creating throttlers with predefined limits


def create_throttled_generator(
    generator: SchematicGenerator[T],
    logger: Logger,
    rate_limits: Optional[RateLimits] = None,
    global_limits: Optional[RateLimits] = None,
) -> EvaluationThrottler[T]:
    """Create a throttled generator with default user provider."""
    user_provider = DefaultUserProvider(rate_limits or RateLimits.free_tier())
    return EvaluationThrottler(generator, user_provider, logger, global_limits)


def create_throttled_embedder(
    embedder: Embedder,
    logger: Logger,
    rate_limits: Optional[RateLimits] = None,
    global_limits: Optional[RateLimits] = None,
) -> EmbedderThrottler:
    """Create a throttled embedder with default user provider."""
    user_provider = DefaultUserProvider(rate_limits or RateLimits.free_tier())
    return EmbedderThrottler(embedder, user_provider, logger, global_limits)
