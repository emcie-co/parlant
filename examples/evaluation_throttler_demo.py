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

"""
Example demonstrating the flexible EvaluationThrottler and EmbedderThrottler
with OpenAI-inspired rate limiting.
"""

import asyncio
import logging

from parlant.core.nlp.evaluation_throttler import (
    EvaluationThrottler,
    EmbedderThrottler,
    RateLimits,
    DefaultUserProvider,
    ThrottlingException,
    ServiceExhaustedException,
    create_throttled_generator,
    create_throttled_embedder,
)


# Mock implementations for demonstration
class MockSchematicGenerator:
    """Mock generator for testing."""

    def __init__(self, name: str = "mock-generator"):
        self.name = name

    @property
    def id(self) -> str:
        return self.name

    @property
    def max_tokens(self) -> int:
        return 4096

    @property
    def tokenizer(self):
        # Mock tokenizer
        class MockTokenizer:
            def estimate_tokens(self, text: str) -> int:
                return len(text) // 4

        return MockTokenizer()

    async def generate(self, prompt, hints=None):
        # Mock response
        class MockResult:
            def __init__(self):
                self.content = "Mock generated content"
                self.info = {"tokens_used": 10}

        # Simulate processing time
        await asyncio.sleep(0.1)
        return MockResult()


class MockEmbedder:
    """Mock embedder for testing."""

    def __init__(self, name: str = "mock-embedder"):
        self.name = name

    @property
    def id(self) -> str:
        return self.name

    @property
    def max_batch_size(self) -> int:
        return 100

    @property
    def dimensions(self) -> int:
        return 1536

    async def embed(self, texts, hints=None):
        # Mock response
        class MockResult:
            def __init__(self, text_count: int):
                self.vectors = [[0.1] * 1536 for _ in range(text_count)]
                self.info = {"tokens_used": sum(len(text) for text in texts) // 4}

        # Simulate processing time
        await asyncio.sleep(0.05)
        return MockResult(len(texts))


class CustomUserProvider(DefaultUserProvider):
    """Custom user provider that demonstrates different rate limits per user."""

    def __init__(self):
        super().__init__()
        # Define different limits for different users
        self._user_limits = {
            "free_user": RateLimits.free_tier(),
            "basic_user": RateLimits.basic_tier(),
            "premium_user": RateLimits.premium_tier(),
            "enterprise_user": RateLimits.enterprise_tier(),
            "custom_user": RateLimits.create_custom(
                rpm=100, tpm=150_000, rph=5000, rpd=50_000, max_concurrent=10
            ),
        }

    async def get_user_limits(self, user_id: str) -> RateLimits:
        """Return custom limits based on user ID."""
        return self._user_limits.get(user_id, RateLimits.free_tier())


async def demonstrate_flexible_rate_limits():
    """Demonstrate the new flexible rate limiting system."""
    print("🚀 Demonstrating Flexible Rate Limiting System")
    print("=" * 60)

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Create mock services
    generator = MockSchematicGenerator("gpt-4")
    embedder = MockEmbedder("text-embedding-ada-002")

    # 1. Basic usage with predefined tiers
    print("\n1. Predefined Tier Examples")
    print("-" * 30)

    # Free tier throttler
    free_throttler = create_throttled_generator(
        generator, logger, rate_limits=RateLimits.free_tier()
    )
    print(f"✅ Created free tier throttler: {free_throttler.id}")
    print(f"   Limits: RPM=3, TPM=40k, RPH=200, RPD=1k, concurrent=1")

    # Premium tier throttler
    premium_throttler = create_throttled_generator(
        generator, logger, rate_limits=RateLimits.premium_tier()
    )
    print(f"✅ Created premium tier throttler: {premium_throttler.id}")
    print(f"   Limits: RPM=5k, TPM=300k, RPH=300k, RPD=1M, concurrent=20")

    # 2. Custom rate limits
    print("\n2. Custom Rate Limits Examples")
    print("-" * 30)

    # OpenAI GPT-4 style limits
    gpt4_limits = RateLimits.create_custom(
        rpm=500,  # 500 requests per minute
        tpm=30_000,  # 30k tokens per minute
        rph=30_000,  # 30k requests per hour
        rpd=500_000,  # 500k requests per day
        max_concurrent=10,  # Max 10 concurrent requests
    )

    gpt4_throttler = create_throttled_generator(generator, logger, gpt4_limits)
    print(f"✅ Created GPT-4 style throttler: {gpt4_throttler.id}")
    print(f"   Limits: RPM=500, TPM=30k, RPH=30k, RPD=500k, concurrent=10")

    # Claude style limits (different pattern)
    claude_limits = RateLimits.create_custom(
        rpm=60,  # 60 requests per minute
        tpm=100_000,  # 100k tokens per minute
        rph=3_600,  # 3.6k requests per hour
        tpd=2_000_000,  # 2M tokens per day
        max_concurrent=5,  # Max 5 concurrent requests
    )

    claude_throttler = create_throttled_generator(generator, logger, claude_limits)
    print(f"✅ Created Claude style throttler: {claude_throttler.id}")
    print(f"   Limits: RPM=60, TPM=100k, RPH=3.6k, TPD=2M, concurrent=5")

    # Token-only limits (no request limits)
    token_only_limits = RateLimits.create_custom(
        tpm=50_000,  # Only token limit per minute
        tpd=1_000_000,  # Only token limit per day
        max_concurrent=3,  # And concurrent limit
        # No RPM, RPH, RPD limits
    )

    token_throttler = create_throttled_generator(generator, logger, token_only_limits)
    print(f"✅ Created token-only throttler: {token_throttler.id}")
    print(f"   Limits: Only TPM=50k, TPD=1M, concurrent=3")

    # 3. Embedder throttling
    print("\n3. Embedder Throttling Examples")
    print("-" * 30)

    embedding_limits = RateLimits.create_custom(
        rpm=3000,  # High request rate for embeddings
        tpm=1_000_000,  # 1M tokens per minute
        rph=150_000,  # 150k requests per hour
        max_concurrent=50,  # High concurrency for batch processing
    )

    embedding_throttler = create_throttled_embedder(embedder, logger, embedding_limits)
    print(f"✅ Created embedding throttler: {embedding_throttler.id}")
    print(f"   Limits: RPM=3k, TPM=1M, RPH=150k, concurrent=50")

    # 4. Custom user provider with different limits per user
    print("\n4. Custom User Provider Examples")
    print("-" * 30)

    custom_provider = CustomUserProvider()

    # Create throttler with custom user provider
    multi_tier_throttler = EvaluationThrottler(generator, custom_provider, logger)

    print(f"✅ Created multi-tier throttler: {multi_tier_throttler.id}")
    print("   Different limits per user based on user_id in hints")

    # 5. Testing rate limiting in action
    print("\n5. Rate Limiting in Action")
    print("-" * 30)

    # Use a very restrictive throttler for demo
    restrictive_limits = RateLimits.create_custom(
        rpm=2,  # Only 2 requests per minute
        tpm=1000,  # 1000 tokens per minute
        max_concurrent=1,  # Only 1 concurrent request
    )

    restrictive_throttler = create_throttled_generator(generator, logger, restrictive_limits)

    print(f"Testing with restrictive limits (RPM=2, TPM=1000, concurrent=1)...")

    try:
        # First request should succeed
        result1 = await restrictive_throttler.generate("Hello world", {"user_id": "test_user"})
        print("✅ Request 1: Success")

        # Second request should succeed
        result2 = await restrictive_throttler.generate("Hello again", {"user_id": "test_user"})
        print("✅ Request 2: Success")

        # Third request should be throttled (exceeds RPM=2)
        try:
            result3 = await restrictive_throttler.generate(
                "Hello third time", {"user_id": "test_user"}
            )
            print("❌ Request 3: Unexpected success")
        except ThrottlingException as e:
            print(f"✅ Request 3: Correctly throttled - {e}")
            print(f"   Retry after: {e.retry_after:.1f} seconds")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    # 6. Global limits demonstration
    print("\n6. Global Service Limits")
    print("-" * 30)

    # Set up global limits for the entire service
    global_limits = RateLimits.create_custom(
        rpm=10,  # Service-wide limit
        tpm=50_000,  # Service-wide token limit
        max_concurrent=5,  # Service-wide concurrency limit
    )

    global_throttler = EvaluationThrottler(
        generator,
        DefaultUserProvider(RateLimits.premium_tier()),  # User has premium limits
        logger,
        global_limits=global_limits,  # But service has global limits
    )

    print(f"✅ Created throttler with global limits: {global_throttler.id}")
    print(f"   User limits: Premium tier")
    print(f"   Global limits: RPM=10, TPM=50k, concurrent=5")

    print("\n🎉 Demonstration complete!")
    print("\nKey Benefits of the New System:")
    print("• ✅ Flexible: Define any combination of RPM, TPM, RPH, RPD, etc.")
    print("• ✅ OpenAI-compatible: Easy to mirror OpenAI's rate limiting")
    print("• ✅ Per-user: Different limits for different users")
    print("• ✅ Global limits: Service-wide throttling")
    print("• ✅ Token-aware: Considers both requests and token usage")
    print("• ✅ Concurrent limiting: Controls simultaneous requests")
    print("• ✅ Multiple timeframes: Minute, hour, day, month limits")


if __name__ == "__main__":
    asyncio.run(demonstrate_flexible_rate_limits())
