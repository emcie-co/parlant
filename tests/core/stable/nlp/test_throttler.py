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
from typing import Any, Mapping
from unittest.mock import AsyncMock
from lagom import Container

from pytest import raises

from parlant.core.common import DefaultBaseModel
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.loggers import Logger
from parlant.core.nlp.embedding import Embedder, EmbeddingResult
from parlant.core.nlp.evaluation_throttler import (
    EmbedderThrottler,
    EvaluationThrottler,
    RateLimits,
    RateLimitTracker,
    ThrottlingException,
    ServiceExhaustedException,
    DefaultUserProvider,
    create_throttled_embedder,
    create_throttled_generator,
)
from parlant.core.nlp.generation import SchematicGenerator, SchematicGenerationResult
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.nlp.tokenization import EstimatingTokenizer, ZeroEstimatingTokenizer


class DummySchema(DefaultBaseModel):
    result: str


class MockGenerator(SchematicGenerator[DummySchema]):
    """Mock generator for testing purposes."""

    def __init__(self) -> None:
        self._tokenizer = ZeroEstimatingTokenizer()
        self.call_count = 0

    @property
    def id(self) -> str:
        return "mock-generator"

    @property
    def max_tokens(self) -> int:
        return 4096

    @property
    def tokenizer(self) -> EstimatingTokenizer:
        return self._tokenizer

    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[DummySchema]:
        self.call_count += 1
        await asyncio.sleep(0.01)  # Simulate processing time

        return SchematicGenerationResult(
            content=DummySchema(result=f"Generated response #{self.call_count}"),
            info=GenerationInfo(
                schema_name="DummySchema",
                model="mock-model",
                duration=10,
                usage=UsageInfo(
                    input_tokens=10,
                    output_tokens=20,
                ),
            ),
        )


class MockEmbedder(Embedder):
    """Mock embedder for testing purposes."""

    def __init__(self) -> None:
        self._tokenizer = ZeroEstimatingTokenizer()
        self.call_count = 0

    @property
    def id(self) -> str:
        return "mock-embedder"

    @property
    def max_tokens(self) -> int:
        return 8192

    @property
    def tokenizer(self) -> EstimatingTokenizer:
        return self._tokenizer

    @property
    def dimensions(self) -> int:
        return 1536

    async def embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        self.call_count += 1
        await asyncio.sleep(0.01)  # Simulate processing time

        vectors = [[0.1] * self.dimensions for _ in texts]
        return EmbeddingResult(vectors=vectors)


async def test_evaluation_throttler_allows_requests_within_limits(container: Container) -> None:
    """Test that throttler allows requests when within rate limits."""
    mock_generator = MockGenerator()

    # Very permissive limits
    rate_limits = RateLimits(rpm=100, tpm=10000, max_concurrent=10)

    user_provider = DefaultUserProvider(rate_limits)
    throttler = EvaluationThrottler(
        generator=mock_generator,
        user_provider=user_provider,
        logger=container[Logger],
    )

    # Should succeed
    result = await throttler.generate("test prompt")

    assert result.content.result == "Generated response #1"
    assert mock_generator.call_count == 1


async def test_evaluation_throttler_blocks_requests_exceeding_rpm_limit(
    container: Container,
) -> None:
    """Test that throttler blocks requests when RPM limit is exceeded."""
    mock_generator = MockGenerator()

    # Very restrictive RPM limit
    rate_limits = RateLimits(
        rpm=1,  # Only 1 request per minute
        max_concurrent=5,
    )

    user_provider = DefaultUserProvider(rate_limits)
    throttler = EvaluationThrottler(
        generator=mock_generator,
        user_provider=user_provider,
        logger=container[Logger],
    )

    # First request should succeed
    result1 = await throttler.generate("test prompt 1")
    assert result1.content.result == "Generated response #1"

    # Second request should be throttled
    with raises(ThrottlingException) as exc_info:
        await throttler.generate("test prompt 2")

    assert "rate limit exceeded" in str(exc_info.value).lower()
    assert mock_generator.call_count == 1  # Only first request went through


async def test_evaluation_throttler_blocks_requests_exceeding_concurrent_limit(
    container: Container,
) -> None:
    """Test that throttler blocks requests when concurrent limit is exceeded."""
    mock_generator = AsyncMock(spec=SchematicGenerator[DummySchema])

    # Mock generator that takes some time
    async def slow_generate(*args: Any, **kwargs: Any) -> SchematicGenerationResult[DummySchema]:
        await asyncio.sleep(0.1)
        return SchematicGenerationResult(
            content=DummySchema(result="Success"),
            info=GenerationInfo(
                schema_name="DummySchema",
                model="mock-model",
                duration=100,
                usage=UsageInfo(input_tokens=10, output_tokens=20),
            ),
        )

    mock_generator.generate = slow_generate
    mock_generator.id = "slow-generator"
    mock_generator.max_tokens = 4096
    mock_generator.tokenizer = ZeroEstimatingTokenizer()

    # Very restrictive concurrent limit
    rate_limits = RateLimits(
        rpm=100,
        max_concurrent=1,  # Only 1 concurrent request
    )

    user_provider = DefaultUserProvider(rate_limits)
    throttler: EvaluationThrottler[DummySchema] = EvaluationThrottler(
        generator=mock_generator,
        user_provider=user_provider,
        logger=container[Logger],
    )

    # Start two requests concurrently
    task1 = asyncio.create_task(throttler.generate("prompt 1"))
    await asyncio.sleep(0.01)  # Let first request start

    # Second request should be throttled due to concurrent limit
    with raises(ThrottlingException) as exc_info:
        await throttler.generate("prompt 2")

    assert "rate limit exceeded" in str(exc_info.value).lower()

    # Wait for first request to complete
    result1 = await task1
    assert result1.content.result == "Success"


async def test_embedder_throttler_allows_requests_within_limits(container: Container) -> None:
    """Test that embedder throttler allows requests when within rate limits."""
    mock_embedder = MockEmbedder()

    # Very permissive limits
    rate_limits = RateLimits(rpm=100, tpm=10000, max_concurrent=10)

    user_provider = DefaultUserProvider(rate_limits)
    throttler = EmbedderThrottler(
        embedder=mock_embedder,
        user_provider=user_provider,
        logger=container[Logger],
    )

    # Should succeed
    result = await throttler.embed(["test text"])

    assert len(result.vectors) == 1
    assert len(result.vectors[0]) == mock_embedder.dimensions
    assert mock_embedder.call_count == 1


async def test_embedder_throttler_blocks_requests_exceeding_limits(container: Container) -> None:
    """Test that embedder throttler blocks requests when rate limits are exceeded."""
    mock_embedder = MockEmbedder()

    # Very restrictive limits
    rate_limits = RateLimits(
        rpm=1,  # Only 1 request per minute
        max_concurrent=1,
    )

    user_provider = DefaultUserProvider(rate_limits)
    throttler = EmbedderThrottler(
        embedder=mock_embedder,
        user_provider=user_provider,
        logger=container[Logger],
    )

    # First request should succeed
    result1 = await throttler.embed(["test text 1"])
    assert len(result1.vectors) == 1

    # Second request should be throttled
    with raises(ThrottlingException) as exc_info:
        await throttler.embed(["test text 2"])

    assert "rate limit exceeded" in str(exc_info.value).lower()
    assert mock_embedder.call_count == 1  # Only first request went through


async def test_global_limits_are_enforced_across_users(container: Container) -> None:
    """Test that global limits are enforced across all users."""
    mock_generator = MockGenerator()

    # Permissive user limits but restrictive global limits
    user_limits = RateLimits(rpm=100, max_concurrent=10)
    global_limits = RateLimits(rpm=1, max_concurrent=1)  # Very restrictive

    user_provider = DefaultUserProvider(user_limits)
    throttler = EvaluationThrottler(
        generator=mock_generator,
        user_provider=user_provider,
        logger=container[Logger],
        global_limits=global_limits,
    )

    # First request should succeed
    result1 = await throttler.generate("test prompt 1")
    assert result1.content.result == "Generated response #1"

    # Second request should be blocked by global limits
    with raises(ServiceExhaustedException) as exc_info:
        await throttler.generate("test prompt 2")

    assert "global service rate limit exceeded" in str(exc_info.value).lower()
    assert mock_generator.call_count == 1


async def test_token_limits_are_enforced(container: Container) -> None:
    """Test that token-based limits are enforced."""
    mock_generator = AsyncMock(spec=SchematicGenerator[DummySchema])

    # Mock tokenizer that returns high token count
    mock_tokenizer = AsyncMock(spec=EstimatingTokenizer)
    mock_tokenizer.estimate_token_count.return_value = 400  # High token count

    mock_generator.generate.return_value = SchematicGenerationResult(
        content=DummySchema(result="Success"),
        info=GenerationInfo(
            schema_name="DummySchema",
            model="mock-model",
            duration=10,
            usage=UsageInfo(input_tokens=400, output_tokens=20),
        ),
    )
    mock_generator.id = "token-heavy-generator"
    mock_generator.max_tokens = 4096
    mock_generator.tokenizer = mock_tokenizer

    # Token-based limits (very restrictive)
    rate_limits = RateLimits(
        rpm=100,  # Plenty of requests allowed
        tpm=500,  # But very few tokens per minute - 400 + 400 > 500
        max_concurrent=10,
    )

    user_provider = DefaultUserProvider(rate_limits)
    throttler: EvaluationThrottler[DummySchema] = EvaluationThrottler(
        generator=mock_generator,
        user_provider=user_provider,
        logger=container[Logger],
    )

    # First request should succeed but consume most tokens (400 tokens)
    result1 = await throttler.generate("high token prompt")
    assert result1.content.result == "Success"

    # Second request should be blocked due to token limits (400 + 400 > 500)
    with raises(ThrottlingException) as exc_info:
        await throttler.generate("another high token prompt")

    assert "rate limit exceeded" in str(exc_info.value).lower()


async def test_convenience_functions_create_working_throttlers(container: Container) -> None:
    """Test that convenience functions create properly working throttlers."""
    mock_generator = MockGenerator()
    mock_embedder = MockEmbedder()

    # Test create_throttled_generator
    rate_limits = RateLimits(rpm=10, tpm=1000, max_concurrent=5)
    throttled_gen = create_throttled_generator(
        generator=mock_generator,
        logger=container[Logger],
        rate_limits=rate_limits,
    )

    result = await throttled_gen.generate("test prompt")
    assert result.content.result == "Generated response #1"

    # Test create_throttled_embedder
    throttled_emb = create_throttled_embedder(
        embedder=mock_embedder,
        logger=container[Logger],
        rate_limits=rate_limits,
    )

    embed_result = await throttled_emb.embed(["test text"])
    assert len(embed_result.vectors) == 1


async def test_predefined_tier_limits() -> None:
    """Test that predefined tier limits have reasonable values."""
    free_limits = RateLimits.free_tier()
    assert free_limits.rpm == 3
    assert free_limits.tpm == 40_000
    assert free_limits.rph == 200
    assert free_limits.rpd == 1_000
    assert free_limits.max_concurrent == 1

    premium_limits = RateLimits.premium_tier()
    assert premium_limits.rpm == 5_000
    assert premium_limits.tpm == 300_000
    assert premium_limits.rph == 300_000
    assert premium_limits.rpd == 1_000_000
    assert premium_limits.max_concurrent == 20

    enterprise_limits = RateLimits.enterprise_tier()
    assert enterprise_limits.rpm == 10_000
    assert enterprise_limits.tpm == 2_000_000  # Fixed: should be 2_000_000
    assert enterprise_limits.rph == 600_000
    assert enterprise_limits.rpd == 5_000_000  # Fixed: should be 5_000_000 not None
    assert enterprise_limits.max_concurrent == 100


async def test_custom_rate_limits_configuration() -> None:
    """Test that custom rate limits can be configured flexibly."""
    # Test OpenAI-style limits
    openai_limits = RateLimits(rpm=500, tpm=30_000, rph=30_000, rpd=500_000, max_concurrent=10)

    assert openai_limits.rpm == 500
    assert openai_limits.tpm == 30_000
    assert openai_limits.rph == 30_000
    assert openai_limits.rpd == 500_000
    assert openai_limits.max_concurrent == 10

    # Test token-only limits
    token_only = RateLimits(tpm=50_000, tpd=1_000_000, max_concurrent=3)

    assert token_only.rpm is None  # No request limit
    assert token_only.tpm == 50_000
    assert token_only.tpd == 1_000_000
    assert token_only.max_concurrent == 3


async def test_throttler_preserves_generator_interface(container: Container) -> None:
    """Test that throttler preserves the original generator interface."""
    mock_generator = MockGenerator()

    rate_limits = RateLimits(rpm=100, max_concurrent=10)
    user_provider = DefaultUserProvider(rate_limits)
    throttler = EvaluationThrottler(
        generator=mock_generator,
        user_provider=user_provider,
        logger=container[Logger],
    )

    # Test that throttler exposes generator properties correctly
    assert throttler.id == f"throttled({mock_generator.id})"
    assert throttler.max_tokens == mock_generator.max_tokens
    assert throttler.tokenizer == mock_generator.tokenizer

    # Test that generate method works with all parameter types
    result1 = await throttler.generate("string prompt")
    assert result1.content.result == "Generated response #1"

    result2 = await throttler.generate("prompt with hints", hints={"user": "test"})
    assert result2.content.result == "Generated response #2"


async def test_throttler_preserves_embedder_interface(container: Container) -> None:
    """Test that throttler preserves the original embedder interface."""
    mock_embedder = MockEmbedder()

    rate_limits = RateLimits(rpm=100, max_concurrent=10)
    user_provider = DefaultUserProvider(rate_limits)
    throttler = EmbedderThrottler(
        embedder=mock_embedder,
        user_provider=user_provider,
        logger=container[Logger],
    )

    # Test that throttler exposes embedder properties correctly
    assert throttler.id == f"throttled({mock_embedder.id})"
    assert throttler.max_tokens == mock_embedder.max_tokens
    assert throttler.tokenizer == mock_embedder.tokenizer
    assert throttler.dimensions == mock_embedder.dimensions

    # Test that embed method works with all parameter types
    result1 = await throttler.embed(["test text"])
    assert len(result1.vectors) == 1

    result2 = await throttler.embed(["text1", "text2"], hints={"user": "test"})
    assert len(result2.vectors) == 2


async def test_user_provider_can_provide_different_limits_per_user(container: Container) -> None:
    """Test that UserProvider can provide different limits based on user."""
    from parlant.core.nlp.evaluation_throttler import UserProvider

    class CustomUserProvider(UserProvider):
        async def get_user_id(self, hints: Mapping[str, Any]) -> str:
            return str(hints.get("user_id", "anonymous"))

        async def get_user_limits(self, user_id: str) -> RateLimits:
            if user_id == "premium_user":
                return RateLimits.premium_tier()
            else:
                return RateLimits.free_tier()

    mock_generator = MockGenerator()
    throttler = EvaluationThrottler(
        generator=mock_generator,
        user_provider=CustomUserProvider(),
        logger=container[Logger],
    )

    # Test with free user (should have restrictive limits)
    result1 = await throttler.generate("prompt", hints={"user_id": "free_user"})
    assert result1.content.result == "Generated response #1"

    # Test with premium user (should have more generous limits)
    result2 = await throttler.generate("prompt", hints={"user_id": "premium_user"})
    assert result2.content.result == "Generated response #2"

    assert mock_generator.call_count == 2


async def test_multiple_throttlers_share_global_limits(container: Container) -> None:
    """Test that multiple throttlers can share the same global limits."""
    mock_gen1 = MockGenerator()
    mock_gen2 = MockGenerator()

    user_limits = RateLimits(rpm=100, max_concurrent=10)
    global_limits = RateLimits(rpm=1, max_concurrent=1)  # Very restrictive

    user_provider1 = DefaultUserProvider(user_limits)
    user_provider2 = DefaultUserProvider(user_limits)

    # Create a shared global tracker
    shared_tracker = RateLimitTracker()

    throttler1 = EvaluationThrottler(
        generator=mock_gen1,
        user_provider=user_provider1,
        logger=container[Logger],
        global_limits=global_limits,
        shared_global_tracker=shared_tracker,
    )

    throttler2 = EvaluationThrottler(
        generator=mock_gen2,
        user_provider=user_provider2,
        logger=container[Logger],
        global_limits=global_limits,
        shared_global_tracker=shared_tracker,
    )

    # First request on throttler1 should succeed
    result1 = await throttler1.generate("test 1")
    assert result1.content.result == "Generated response #1"

    # Request on throttler2 should be blocked by shared global limits
    with raises(ServiceExhaustedException):
        await throttler2.generate("test 2")

    assert mock_gen1.call_count == 1
    assert mock_gen2.call_count == 0


async def test_throttling_exception_contains_useful_information() -> None:
    """Test that ThrottlingException contains useful debugging information."""
    rate_limits = RateLimits(rpm=10, tpm=1000)

    exc = ThrottlingException(
        message="Rate limit exceeded",
        rate_limits=rate_limits,
        retry_after=60.5,
    )

    assert "Rate limit exceeded" in str(exc)
    assert exc.rate_limits == rate_limits
    assert exc.retry_after == 60.5


async def test_service_exhausted_exception_contains_useful_information() -> None:
    """Test that ServiceExhaustedException contains useful debugging information."""
    exc = ServiceExhaustedException(
        message="Global service limits exceeded",
        service_id="test-service",
        retry_after=30.0,
    )

    assert "Global service limits exceeded" in str(exc)
    assert exc.service_id == "test-service"
    assert exc.retry_after == 30.0
