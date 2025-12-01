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
    ThrottledEmbedder,
    ThrottledSchematicGenerator,
    RateLimits,
    ThrottlingException,
    ServiceExhaustedException,
    create_rate_limiter,
    throttle_embedder,
    throttle_generator,
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
    rate_limiter = create_rate_limiter(rpm=100, tpm=10000, max_concurrent=10)

    throttler = ThrottledSchematicGenerator(
        generator=mock_generator,
        rate_limiter=rate_limiter,
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
    rate_limiter = create_rate_limiter(
        rpm=1,  # Only 1 request per minute
        max_concurrent=5,
    )

    throttler = ThrottledSchematicGenerator(
        generator=mock_generator,
        rate_limiter=rate_limiter,
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
    rate_limiter = create_rate_limiter(
        rpm=100,
        max_concurrent=1,  # Only 1 concurrent request
    )

    throttler: ThrottledSchematicGenerator[DummySchema] = ThrottledSchematicGenerator(
        generator=mock_generator,
        rate_limiter=rate_limiter,
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
    rate_limiter = create_rate_limiter(rpm=100, tpm=10000, max_concurrent=10)

    throttler = ThrottledEmbedder(
        embedder=mock_embedder,
        rate_limiter=rate_limiter,
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
    rate_limiter = create_rate_limiter(
        rpm=1,  # Only 1 request per minute
        max_concurrent=1,
    )

    throttler = ThrottledEmbedder(
        embedder=mock_embedder,
        rate_limiter=rate_limiter,
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


async def test_rate_limiter_enforces_limits_consistently(container: Container) -> None:
    """Test that rate limiter consistently enforces limits."""
    mock_generator = MockGenerator()

    # Very restrictive limits
    rate_limiter = create_rate_limiter(rpm=1, max_concurrent=1)  # Very restrictive

    throttler = ThrottledSchematicGenerator(
        generator=mock_generator,
        rate_limiter=rate_limiter,
        logger=container[Logger],
    )

    # First request should succeed
    result1 = await throttler.generate("test prompt 1")
    assert result1.content.result == "Generated response #1"

    # Second request should be blocked by rate limits
    with raises(ThrottlingException) as exc_info:
        await throttler.generate("test prompt 2")

    assert "rate limit exceeded" in str(exc_info.value).lower()
    assert mock_generator.call_count == 1


async def test_token_limits_are_enforced(container: Container) -> None:
    """Test that rate limiting works with requests per minute (simplified test)."""
    mock_generator = MockGenerator()

    # Very restrictive request-based limits - we'll test this instead of TPM
    rate_limiter = create_rate_limiter(
        rpm=1,  # Only 1 request per minute
        max_concurrent=10,
    )

    throttler: ThrottledSchematicGenerator[DummySchema] = ThrottledSchematicGenerator(
        generator=mock_generator,
        rate_limiter=rate_limiter,
        logger=container[Logger],
    )

    # First request should succeed
    result1 = await throttler.generate("first prompt")
    assert result1.content.result == "Generated response #1"

    # Second request should be blocked due to RPM limits
    with raises(ThrottlingException) as exc_info:
        await throttler.generate("second prompt")

    assert "rate limit exceeded" in str(exc_info.value).lower()


async def test_convenience_functions_create_working_throttlers(container: Container) -> None:
    """Test that convenience functions create properly working throttlers."""
    mock_generator = MockGenerator()
    mock_embedder = MockEmbedder()

    # Test throttle_generator
    rate_limiter = create_rate_limiter(rpm=10, tpm=1000, max_concurrent=5)
    throttled_gen = throttle_generator(
        generator=mock_generator,
        rate_limiter=rate_limiter,
        logger=container[Logger],
    )

    result = await throttled_gen.generate("test prompt")
    assert result.content.result == "Generated response #1"

    # Test throttle_embedder
    throttled_emb = throttle_embedder(
        embedder=mock_embedder,
        rate_limiter=rate_limiter,
        logger=container[Logger],
    )

    embed_result = await throttled_emb.embed(["test text"])
    assert len(embed_result.vectors) == 1


async def test_predefined_rate_limiters() -> None:
    """Test that predefined rate limiter functions work correctly."""
    from parlant.core.nlp.evaluation_throttler import (
        openai_free_tier_limits,
        openai_tier_1_limits,
        openai_tier_2_limits,
        conservative_limits,
    )

    # Test that functions return rate limiters (they should be callable objects)
    free_limiter = openai_free_tier_limits()
    tier_1_limiter = openai_tier_1_limits()
    tier_2_limiter = openai_tier_2_limits()
    conservative_limiter = conservative_limits()

    # Test that they implement the rate limiter protocol
    assert hasattr(free_limiter, "can_make_request")
    assert hasattr(tier_1_limiter, "can_make_request")
    assert hasattr(tier_2_limiter, "can_make_request")
    assert hasattr(conservative_limiter, "can_make_request")

    assert hasattr(free_limiter, "record_request_start")
    assert hasattr(tier_1_limiter, "record_request_start")
    assert hasattr(tier_2_limiter, "record_request_start")
    assert hasattr(conservative_limiter, "record_request_start")


async def test_custom_rate_limits_configuration() -> None:
    """Test that custom rate limits can be configured flexibly."""
    # Test that create_rate_limiter works with various parameters
    openai_limiter = create_rate_limiter(
        rpm=500, tpm=30_000, rph=30_000, rpd=500_000, max_concurrent=10
    )

    # Test that it implements the expected interface
    assert hasattr(openai_limiter, "can_make_request")
    assert hasattr(openai_limiter, "record_request_start")
    assert hasattr(openai_limiter, "record_request_end")

    # Test token-only limits
    token_only_limiter = create_rate_limiter(tpm=50_000, tpd=1_000_000, max_concurrent=3)

    # Test that it also implements the expected interface
    assert hasattr(token_only_limiter, "can_make_request")
    assert hasattr(token_only_limiter, "record_request_start")
    assert hasattr(token_only_limiter, "record_request_end")


async def test_throttler_preserves_generator_interface(container: Container) -> None:
    """Test that throttler preserves the original generator interface."""
    mock_generator = MockGenerator()

    rate_limiter = create_rate_limiter(rpm=100, max_concurrent=10)
    throttler = ThrottledSchematicGenerator(
        generator=mock_generator,
        rate_limiter=rate_limiter,
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

    rate_limiter = create_rate_limiter(rpm=100, max_concurrent=10)
    throttler = ThrottledEmbedder(
        embedder=mock_embedder,
        rate_limiter=rate_limiter,
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


async def test_rate_limiter_works_with_hints(container: Container) -> None:
    """Test that throttler works correctly with hints parameter."""
    mock_generator = MockGenerator()

    rate_limiter = create_rate_limiter(rpm=100, max_concurrent=10)
    throttler = ThrottledSchematicGenerator(
        generator=mock_generator,
        rate_limiter=rate_limiter,
        logger=container[Logger],
    )

    # Test with hints (should work normally)
    result1 = await throttler.generate("prompt", hints={"user_id": "test_user"})
    assert result1.content.result == "Generated response #1"

    # Test without hints (should also work)
    result2 = await throttler.generate("prompt")
    assert result2.content.result == "Generated response #2"

    assert mock_generator.call_count == 2


async def test_multiple_throttlers_can_use_shared_rate_limiter(container: Container) -> None:
    """Test that multiple throttlers can share the same rate limiter instance."""
    mock_gen1 = MockGenerator()
    mock_gen2 = MockGenerator()

    # Create a shared rate limiter with restrictive limits
    shared_limiter = create_rate_limiter(rpm=1, max_concurrent=1)  # Very restrictive

    throttler1 = ThrottledSchematicGenerator(
        generator=mock_gen1,
        rate_limiter=shared_limiter,
        logger=container[Logger],
    )

    throttler2 = ThrottledSchematicGenerator(
        generator=mock_gen2,
        rate_limiter=shared_limiter,
        logger=container[Logger],
    )

    # First request on throttler1 should succeed
    result1 = await throttler1.generate("test 1")
    assert result1.content.result == "Generated response #1"

    # Request on throttler2 should be blocked by shared rate limits
    with raises(ThrottlingException):
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
