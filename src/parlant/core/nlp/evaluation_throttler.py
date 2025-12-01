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

from __future__ import annotations
import asyncio
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from limits import RateLimitItemPerMinute, RateLimitItemPerHour, RateLimitItemPerDay
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter

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
    - TPD (tokens per day)
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

    # Tokens per day
    tpd: int | None = None

    # Maximum concurrent requests
    max_concurrent: int | None = None


class RateLimiterProtocol(Protocol):
    """Protocol for rate limiting implementations."""

    async def can_make_request(self, estimated_tokens: int = 1) -> tuple[bool, float | None]:
        """Check if a request can be made within the current limits.

        Args:
            estimated_tokens: Estimated number of tokens for this request

        Returns:
            tuple: (can_make_request, retry_after_seconds)
        """
        ...

    async def record_request_start(self, tokens: int = 1) -> None:
        """Record the start of a request (for concurrent tracking)."""
        ...

    async def record_request_end(self) -> None:
        """Record the end of a request."""
        ...


class NullRateLimiter:
    """A no-op rate limiter that allows all requests through."""

    def __init__(self, limits: RateLimits | None = None) -> None:
        """Initialize null rate limiter (limits parameter ignored)."""
        pass

    async def can_make_request(self, estimated_tokens: int = 1) -> tuple[bool, float | None]:
        """Always allows requests through."""
        return True, None

    async def record_request_start(self, tokens: int = 1) -> None:
        """No-op implementation."""
        pass

    async def record_request_end(self) -> None:
        """No-op implementation."""
        pass


class ThrottlingException(Exception):
    def __init__(self, message: str, rate_limits: RateLimits, retry_after: float | None = None):
        super().__init__(message)
        self.rate_limits = rate_limits
        self.retry_after = retry_after


class ServiceExhaustedException(Exception):
    def __init__(self, message: str, service_id: str, retry_after: float | None = None):
        super().__init__(message)
        self.service_id = service_id
        self.retry_after = retry_after


class RateLimiter:
    """Active rate limiter using the limits package for sophisticated rate limiting."""

    def __init__(self, limits: RateLimits, storage: MemoryStorage | None = None) -> None:
        self._limits = limits
        self._storage = storage or MemoryStorage()
        self._limiter = MovingWindowRateLimiter(self._storage)
        self._concurrent_count = 0
        self._lock = asyncio.Lock()

    async def can_make_request(self, estimated_tokens: int = 1) -> tuple[bool, float | None]:
        """Check if a request can be made within the current limits.

        Args:
            estimated_tokens: Estimated number of tokens for this request

        Returns:
            tuple: (can_make_request, retry_after_seconds)
        """
        async with self._lock:
            # Check concurrent requests
            if (
                self._limits.max_concurrent is not None
                and self._concurrent_count >= self._limits.max_concurrent
            ):
                return False, 1.0  # Retry after 1 second

            # Check minute limits (RPM and TPM)
            if self._limits.rpm is not None:
                rpm_item = RateLimitItemPerMinute(self._limits.rpm)
                # Test first without consuming the limit
                if not self._limiter.test(rpm_item, "rpm"):
                    # Rate limit exceeded - for moving window, retry after a shorter interval
                    # Since we don't know exactly when the window will clear, use a reasonable estimate
                    return False, min(
                        60.0 / self._limits.rpm, 60.0
                    )  # At most 1 minute, but usually much less
                # If test passed, actually consume the limit
                self._limiter.hit(rpm_item, "rpm")

            if self._limits.tpm is not None:
                tpm_item = RateLimitItemPerMinute(self._limits.tpm)
                # For tokens, check if we have enough quota for all estimated tokens
                # We approximate by testing multiple times
                tokens_can_fit = 0
                for _ in range(estimated_tokens):
                    if self._limiter.test(tpm_item, "tpm"):
                        tokens_can_fit += 1
                    else:
                        break

                if tokens_can_fit < estimated_tokens:
                    # Not enough token quota - calculate smart retry
                    tokens_needed = estimated_tokens - tokens_can_fit
                    retry_time = min(60.0 * tokens_needed / self._limits.tpm, 60.0)
                    return False, max(retry_time, 1.0)  # At least 1 second

                # We have enough quota, consume the tokens
                for _ in range(estimated_tokens):
                    self._limiter.hit(tpm_item, "tpm")

            # Check hour limits (RPH)
            if self._limits.rph is not None:
                rph_item = RateLimitItemPerHour(self._limits.rph)
                if not self._limiter.test(rph_item, "rph"):
                    # For hourly limits, use a smarter retry - typically much less than full hour
                    return False, min(3600.0 / self._limits.rph, 300.0)  # At most 5 minutes
                self._limiter.hit(rph_item, "rph")

            # Check day limits (RPD and TPD)
            if self._limits.rpd is not None:
                rpd_item = RateLimitItemPerDay(self._limits.rpd)
                if not self._limiter.test(rpd_item, "rpd"):
                    # For daily limits, use smarter retry - usually much less than full day
                    return False, min(86400.0 / self._limits.rpd, 3600.0)  # At most 1 hour
                self._limiter.hit(rpd_item, "rpd")

            if self._limits.tpd is not None:
                tpd_item = RateLimitItemPerDay(self._limits.tpd)
                # Check token availability for the day
                tokens_can_fit = 0
                for _ in range(estimated_tokens):
                    if self._limiter.test(tpd_item, "tpd"):
                        tokens_can_fit += 1
                    else:
                        break

                if tokens_can_fit < estimated_tokens:
                    # Not enough daily token quota
                    tokens_needed = estimated_tokens - tokens_can_fit
                    retry_time = min(86400.0 * tokens_needed / self._limits.tpd, 3600.0)
                    return False, max(retry_time, 1.0)

                # Consume the daily token quota
                for _ in range(estimated_tokens):
                    self._limiter.hit(tpd_item, "tpd")

            return True, None

    async def record_request_start(self, tokens: int = 1) -> None:
        """Record the start of a request (for concurrent tracking)."""
        async with self._lock:
            self._concurrent_count += 1

    async def record_request_end(self) -> None:
        """Record the end of a request."""
        async with self._lock:
            self._concurrent_count = max(0, self._concurrent_count - 1)


class ThrottledSchematicGenerator(SchematicGenerator[T]):
    """A throttling wrapper for SchematicGenerator that enforces flexible rate limits."""

    def __init__(
        self,
        generator: SchematicGenerator[T],
        rate_limiter: RateLimiterProtocol,
        logger: Logger | None = None,
    ) -> None:
        self._generator = generator
        self._rate_limiter = rate_limiter
        self._logger = logger

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
        # Estimate tokens for this request
        estimated_tokens = await self._estimate_tokens(prompt)

        # Check rate limits
        can_proceed, retry_after = await self._rate_limiter.can_make_request(estimated_tokens)
        if not can_proceed:
            raise ThrottlingException(
                "Rate limit exceeded",
                getattr(self._rate_limiter, "_limits", RateLimits()),
                retry_after,
            )

        # Record request start
        await self._rate_limiter.record_request_start(estimated_tokens)

        try:
            # Make the actual request
            result = await self._generator.generate(prompt, hints)
            return result
        finally:
            # Record request end
            await self._rate_limiter.record_request_end()

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


class ThrottledEmbedder(Embedder):
    """A throttling wrapper for Embedder that enforces flexible rate limits."""

    def __init__(
        self,
        embedder: Embedder,
        rate_limiter: RateLimiterProtocol,
        logger: Logger | None = None,
    ) -> None:
        self._embedder = embedder
        self._rate_limiter = rate_limiter
        self._logger = logger

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
        # Estimate tokens for this request
        estimated_tokens = await self._estimate_tokens(texts)

        # Check rate limits
        can_proceed, retry_after = await self._rate_limiter.can_make_request(estimated_tokens)
        if not can_proceed:
            raise ThrottlingException(
                "Rate limit exceeded",
                getattr(self._rate_limiter, "_limits", RateLimits()),
                retry_after,
            )

        # Record request start
        await self._rate_limiter.record_request_start(estimated_tokens)

        try:
            # Make the actual request
            result = await self._embedder.embed(texts, hints)
            return result
        finally:
            # Record request end
            await self._rate_limiter.record_request_end()

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


# Convenience functions for creating rate limiters and throttled generators


def create_rate_limiter(
    rpm: int | None = None,
    tpm: int | None = None,
    rph: int | None = None,
    rpd: int | None = None,
    tpd: int | None = None,
    max_concurrent: int | None = None,
) -> RateLimiterProtocol:
    """Create a rate limiter with the specified limits.

    Args:
        rpm: Requests per minute
        tpm: Tokens per minute
        rph: Requests per hour
        rpd: Requests per day
        tpd: Tokens per day
        max_concurrent: Maximum concurrent requests

    Returns:
        RateLimiter: Ready to use for throttling
    """
    rate_limits = RateLimits(
        rpm=rpm,
        tpm=tpm,
        rph=rph,
        rpd=rpd,
        tpd=tpd,
        max_concurrent=max_concurrent,
    )

    # If no limits are specified, return a null limiter
    if all(limit is None for limit in [rpm, tpm, rph, rpd, tpd, max_concurrent]):
        return NullRateLimiter(rate_limits)

    return RateLimiter(rate_limits)


def create_null_rate_limiter() -> RateLimiterProtocol:
    """Create a no-op rate limiter that allows all requests through.

    Returns:
        RateLimiter: A null rate limiter
    """
    return NullRateLimiter(RateLimits())


def throttle_generator(
    generator: SchematicGenerator[T],
    rate_limiter: RateLimiterProtocol,
    logger: Logger | None = None,
) -> ThrottledSchematicGenerator[T]:
    """Wrap a generator with throttling.

    Example:
        # Create rate limiter with 10 RPM, 1000 TPM
        limiter = create_rate_limiter(rpm=10, tpm=1000)

        # Wrap generator
        throttled = throttle_generator(my_generator, limiter)
    """
    return ThrottledSchematicGenerator(generator, rate_limiter, logger)


def throttle_embedder(
    embedder: Embedder,
    rate_limiter: RateLimiterProtocol,
    logger: Logger | None = None,
) -> ThrottledEmbedder:
    """Wrap an embedder with throttling.

    Example:
        # Create rate limiter with 100 RPM
        limiter = create_rate_limiter(rpm=100)

        # Wrap embedder
        throttled = throttle_embedder(my_embedder, limiter)
    """
    return ThrottledEmbedder(embedder, rate_limiter, logger)


# Predefined common rate limit configurations


def openai_free_tier_limits() -> RateLimiterProtocol:
    """OpenAI free tier typical limits (approximate)."""
    return create_rate_limiter(rpm=3, tpm=40_000, rpd=200)


def openai_tier_1_limits() -> RateLimiterProtocol:
    """OpenAI Tier 1 typical limits (approximate)."""
    return create_rate_limiter(rpm=500, tpm=200_000, rpd=10_000)


def openai_tier_2_limits() -> RateLimiterProtocol:
    """OpenAI Tier 2 typical limits (approximate)."""
    return create_rate_limiter(rpm=5_000, tpm=450_000, rpd=None)


def conservative_limits() -> RateLimiterProtocol:
    """Conservative limits suitable for most development scenarios."""
    return create_rate_limiter(rpm=10, tpm=50_000, max_concurrent=3)
