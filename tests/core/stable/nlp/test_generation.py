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
from dataclasses import dataclass
from typing import Any, Mapping
from typing_extensions import override
from parlant.core.container import Container

from pytest import raises

from parlant.core.common import DefaultBaseModel
from parlant.core.engines.compass.prompt_builder import (
    BuiltInSection,
    PromptBuilder,
    PromptSection,
    SectionStatus,
)
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.nlp.embedding import EmbeddingResult
from parlant.core.nlp.generation import (
    HEDGE_TIMEOUT_HINT,
    BaseSchematicGenerator,
    FallbackSchematicGenerator,
    SchematicGenerationResult,
    SchematicGenerator,
)
from parlant.core.nlp.common import UsageInfo
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.tokenization import EstimatingTokenizer, ZeroEstimatingTokenizer
from parlant.core.tracer import Tracer
from parlant.core.usage_reporter import UsageReporter


class DummySchema(DefaultBaseModel):
    result: str


class FirstException(Exception):
    pass


class SecondException(Exception):
    pass


def _success_result(result: str = "Success") -> SchematicGenerationResult[DummySchema]:
    return SchematicGenerationResult(
        content=DummySchema(result=result),
        info=GenerationInfo(
            schema_name="DummySchema",
            model="not-real-model",
            duration=1,
            usage=UsageInfo(input_tokens=1, output_tokens=1),
        ),
    )


class ScriptedSchematicGenerator(SchematicGenerator[DummySchema]):
    """A hand-written scripted generator: the Nth call to `generate` yields the
    Nth scripted item, either returning it (a result) or raising it (an
    exception). Records every call's prompt and hints."""

    def __init__(
        self,
        script: list[SchematicGenerationResult[DummySchema] | Exception],
    ) -> None:
        self._script = list(script)
        self.calls: list[tuple[str | PromptBuilder, Mapping[str, Any]]] = []

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @override
    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[DummySchema]:
        self.calls.append((prompt, dict(hints)))
        assert self._script, "ScriptedSchematicGenerator called more times than scripted"
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    @override
    @property
    def id(self) -> str:
        return "scripted-schematic-generator"

    @override
    @property
    def max_tokens(self) -> int:
        return 1000

    @override
    @property
    def tokenizer(self) -> EstimatingTokenizer:
        return ZeroEstimatingTokenizer()


async def test_that_fallback_generation_uses_the_first_working_generator(
    container: Container,
) -> None:
    first_generator = ScriptedSchematicGenerator([_success_result()])
    second_generator = ScriptedSchematicGenerator([])

    fallback_generator = FallbackSchematicGenerator[DummySchema](
        first_generator,
        second_generator,
        logger=container[Logger],
    )

    schema_generation_result = await fallback_generator.generate(
        prompt="test prompt", hints={"a": 1}
    )

    assert first_generator.calls == [("test prompt", {"a": 1})]
    assert second_generator.calls == []

    assert schema_generation_result.content.result == "Success"


async def test_that_fallback_generation_falls_back_to_the_next_generator_when_encountering_an_error_in_the_first_one(
    container: Container,
) -> None:
    first_generator = ScriptedSchematicGenerator([Exception("Failure")])
    second_generator = ScriptedSchematicGenerator([_success_result()])

    fallback_generator = FallbackSchematicGenerator[DummySchema](
        first_generator,
        second_generator,
        logger=container[Logger],
    )

    schema_generation_result = await fallback_generator.generate(
        prompt="test prompt", hints={"a": 1}
    )

    assert first_generator.calls == [("test prompt", {"a": 1})]
    assert second_generator.calls == [("test prompt", {"a": 1})]

    assert schema_generation_result.content.result == "Success"


async def test_that_fallback_generation_raises_an_error_when_all_generators_fail(
    container: Container,
) -> None:
    first_generator = ScriptedSchematicGenerator([Exception("Failure")])
    second_generator = ScriptedSchematicGenerator([Exception("Failure")])

    dummy_generator: SchematicGenerator[DummySchema] = FallbackSchematicGenerator(
        first_generator,
        second_generator,
        logger=container[Logger],
    )

    with raises(Exception):
        await dummy_generator.generate("test prompt")

    assert first_generator.calls == [("test prompt", {})]
    assert second_generator.calls == [("test prompt", {})]


async def test_that_retry_succeeds_on_first_attempt(
    container: Container,
) -> None:
    scripted_generator = ScriptedSchematicGenerator([_success_result()])

    @policy([retry(exceptions=(FirstException))])
    async def generate(
        prompt: str, hints: Mapping[str, Any]
    ) -> SchematicGenerationResult[DummySchema]:
        return await scripted_generator.generate(prompt=prompt, hints=hints)

    result = await generate(prompt="test prompt", hints={"a": 1})

    assert scripted_generator.calls == [("test prompt", {"a": 1})]
    assert result.content.result == "Success"


async def test_that_retry_succeeds_after_failures(
    container: Container,
) -> None:
    scripted_generator = ScriptedSchematicGenerator(
        [
            FirstException("First failure"),
            FirstException("Second failure"),
            _success_result(),
        ]
    )

    @policy([retry(exceptions=(FirstException))])
    async def generate(
        prompt: str, hints: Mapping[str, Any]
    ) -> SchematicGenerationResult[DummySchema]:
        return await scripted_generator.generate(prompt=prompt, hints=hints)

    result = await generate(prompt="test prompt", hints={"a": 1})

    assert scripted_generator.call_count == 3
    assert scripted_generator.calls[-1] == ("test prompt", {"a": 1})
    assert result.content.result == "Success"


async def test_that_retry_handles_multiple_exception_types(container: Container) -> None:
    class AnotherException(Exception):
        pass

    scripted_generator = ScriptedSchematicGenerator(
        [
            FirstException("First error"),
            AnotherException("Second error"),
            _success_result(),
        ]
    )

    @policy([retry(exceptions=(FirstException, AnotherException), max_exceptions=3)])
    async def generate(
        prompt: str, hints: Mapping[str, Any] = {}
    ) -> SchematicGenerationResult[DummySchema]:
        return await scripted_generator.generate(prompt, hints)

    result = await generate(prompt="test prompt")

    assert scripted_generator.call_count == 3
    assert result.content.result == "Success"


async def test_that_retry_doesnt_catch_unspecified_exceptions(container: Container) -> None:
    class UnexpectedException(Exception):
        pass

    scripted_generator = ScriptedSchematicGenerator([UnexpectedException("Unexpected error")])

    @policy([retry(exceptions=(FirstException), max_exceptions=3)])
    async def generate(
        prompt: str, hints: Mapping[str, Any] = {}
    ) -> SchematicGenerationResult[DummySchema]:
        return await scripted_generator.generate(prompt, hints)

    with raises(UnexpectedException):
        await generate(prompt="test prompt")

    assert scripted_generator.call_count == 1


class ScriptedEmbedder:
    """A scripted embedding callable: the Nth call yields the Nth scripted item,
    returning it (a result) or raising it (an exception)."""

    def __init__(self, script: list[EmbeddingResult | Exception]) -> None:
        self._script = list(script)
        self.call_count = 0

    async def __call__(self, text: str) -> EmbeddingResult:
        self.call_count += 1
        assert self._script, "ScriptedEmbedder called more times than scripted"
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


async def test_that_stacked_retry_decorators_exceed_max_attempts(container: Container) -> None:
    success_result = EmbeddingResult(vectors=[[0.1, 0.2, 0.3]])

    scripted_embedder = ScriptedEmbedder(
        [
            SecondException("First failure"),
            FirstException("Second failure"),
            FirstException("Third failure"),
            SecondException("Fourth failure"),
            FirstException("Fifth failure"),
            success_result,
        ]
    )

    @policy([retry(SecondException, max_exceptions=3), retry(FirstException, max_exceptions=3)])
    async def embed(text: str) -> EmbeddingResult:
        return await scripted_embedder(text=text)

    with raises(FirstException) as exc_info:
        await embed(text="test text")

    assert scripted_embedder.call_count == 5
    assert str(exc_info.value) == "Fifth failure"


async def test_that_prompt_builder_edits_are_reflected_in_generation() -> None:
    class MockSchematicGenerator(SchematicGenerator[DummySchema]):
        def __init__(self) -> None:
            self.last_prompt: str | None = None

        @override
        @property
        def id(self) -> str:
            return "mock-nlp-service"

        @override
        @property
        def max_tokens(self) -> int:
            return 1000

        @override
        @property
        def tokenizer(self) -> EstimatingTokenizer:
            return ZeroEstimatingTokenizer()

        def _build_agent_identity(self, section: PromptSection) -> PromptSection:
            new_section = PromptSection(
                template="You are NOT {name}",
                props=section.props,
                status=section.status,
            )

            return new_section

        @override
        async def generate(
            self,
            prompt: str | PromptBuilder,
            hints: Mapping[str, Any] = {},
        ) -> SchematicGenerationResult[DummySchema]:
            if isinstance(prompt, PromptBuilder):
                prompt.edit_section(
                    name=BuiltInSection.AGENT_IDENTITY,
                    editor_func=self._build_agent_identity,
                )

                prompt = prompt.build()

            return SchematicGenerationResult(
                content=DummySchema(result=prompt),
                info=GenerationInfo(
                    schema_name="DummySchema",
                    model="mock-model",
                    duration=1,
                    usage=UsageInfo(input_tokens=1, output_tokens=1),
                ),
            )

    mock_service = MockSchematicGenerator()
    builder = PromptBuilder()

    builder.add_section(
        name=BuiltInSection.AGENT_IDENTITY,
        template="You are {name}",
        props={"name": "Bob"},
        status=SectionStatus.ACTIVE,
    )

    result = await mock_service.generate(builder.build())
    assert result.content.result == "You are Bob"


async def test_that_retry_succeeds_after_failures_with_higher_concurrency(
    container: Container,
) -> None:
    concurrency = 10

    @policy(retry(exceptions=(FirstException,)))
    async def generate(
        generator: ScriptedSchematicGenerator,
        prompt: str,
        hints: Mapping[str, Any],
    ) -> SchematicGenerationResult[DummySchema]:
        return await generator.generate(prompt=prompt, hints=hints)

    # Create tasks, each with a different scripted generator
    tasks = []
    scripted_generators = []

    for i in range(concurrency):
        scripted_generator = ScriptedSchematicGenerator(
            [
                FirstException("First failure"),
                FirstException("Second failure"),
                _success_result(),
            ]
        )

        scripted_generators.append(scripted_generator)

        tasks.append(generate(generator=scripted_generator, prompt="test prompt", hints={"a": i}))

    results = await asyncio.gather(*tasks)

    for i in range(concurrency):
        assert scripted_generators[i].call_count == 3
        assert scripted_generators[i].calls[-1] == ("test prompt", {"a": i})
        assert results[i].content.result == "Success"


# ============================================================================
# BaseSchematicGenerator Hedging Tests
# ============================================================================


@dataclass
class _HedgeAttempt:
    """Scripts one `do_generate` invocation: how long it takes, and whether it
    ultimately succeeds (``result``) or raises (``exception``)."""

    delay: float
    result: str | None = None
    exception: BaseException | None = None


class TestableBaseSchematicGenerator(BaseSchematicGenerator[DummySchema]):
    """A BaseSchematicGenerator whose `do_generate` is scripted per attempt, so the
    base class's hedging behavior can be exercised deterministically. The Nth call
    to `do_generate` runs the Nth scripted attempt (the last one repeats)."""

    def __init__(
        self,
        container: Container,
        attempts: list[_HedgeAttempt],
        usage_reporter: UsageReporter | None = None,
    ) -> None:
        super().__init__(
            logger=container[Logger],
            tracer=container[Tracer],
            meter=container[Meter],
            model_name="test-model",
            usage_reporter=usage_reporter,
            provider_name="test",
        )
        self._attempts = attempts
        self.call_count = 0
        self.cancelled_indices: list[int] = []

    @property
    @override
    def schema(self) -> type[DummySchema]:
        return DummySchema

    @override
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[DummySchema]:
        index = self.call_count
        self.call_count += 1
        spec = self._attempts[min(index, len(self._attempts) - 1)]

        try:
            await asyncio.sleep(spec.delay)
        except asyncio.CancelledError:
            self.cancelled_indices.append(index)
            raise

        if spec.exception is not None:
            raise spec.exception

        return SchematicGenerationResult(
            content=DummySchema(
                result=spec.result if spec.result is not None else f"attempt-{index}"
            ),
            info=GenerationInfo(
                schema_name="DummySchema",
                model="test-model",
                duration=spec.delay,
                usage=UsageInfo(input_tokens=1, output_tokens=1),
            ),
        )

    @property
    @override
    def id(self) -> str:
        return "testable-schematic-generator"

    @property
    @override
    def max_tokens(self) -> int:
        return 1000

    @property
    @override
    def tokenizer(self) -> EstimatingTokenizer:
        return ZeroEstimatingTokenizer()


async def test_that_a_request_slower_than_the_hedge_delay_triggers_a_second_attempt(
    container: Container,
) -> None:
    generator = TestableBaseSchematicGenerator(
        container,
        attempts=[
            _HedgeAttempt(delay=5.0, result="slow"),  # first attempt — never returns in time
            _HedgeAttempt(delay=0.01, result="fast"),  # hedge — wins
        ],
    )

    result = await generator.generate("prompt", hints={HEDGE_TIMEOUT_HINT: 0.05})

    assert generator.call_count == 2
    assert result.content.result == "fast"


async def test_that_a_request_faster_than_the_hedge_delay_is_not_hedged(
    container: Container,
) -> None:
    generator = TestableBaseSchematicGenerator(
        container,
        attempts=[_HedgeAttempt(delay=0.01, result="quick")],
    )

    result = await generator.generate("prompt", hints={HEDGE_TIMEOUT_HINT: 0.5})

    assert generator.call_count == 1
    assert result.content.result == "quick"


async def test_that_no_hedge_is_attempted_without_the_hint(
    container: Container,
) -> None:
    generator = TestableBaseSchematicGenerator(
        container,
        attempts=[_HedgeAttempt(delay=0.05, result="only")],
    )

    result = await generator.generate("prompt")

    assert generator.call_count == 1
    assert result.content.result == "only"


async def test_that_the_losing_hedge_attempt_is_cancelled(
    container: Container,
) -> None:
    generator = TestableBaseSchematicGenerator(
        container,
        attempts=[
            _HedgeAttempt(delay=5.0, result="slow"),
            _HedgeAttempt(delay=0.01, result="fast"),
        ],
    )

    result = await generator.generate("prompt", hints={HEDGE_TIMEOUT_HINT: 0.05})

    assert result.content.result == "fast"
    assert generator.cancelled_indices == [0]


async def test_that_all_hedge_attempts_are_included_in_reported_usage(
    container: Container,
) -> None:
    reporter = container[UsageReporter]
    generator = TestableBaseSchematicGenerator(
        container,
        attempts=[
            _HedgeAttempt(delay=0.05, result="slow"),
            _HedgeAttempt(delay=0.001, result="fast"),
        ],
        usage_reporter=reporter,
    )

    result = await generator.generate("prompt", hints={HEDGE_TIMEOUT_HINT: 0.01})
    await reporter.wait_for_pending()

    usage = reporter.get_usage()["test:test-model"]
    assert result.content.result == "fast"
    assert generator.cancelled_indices == []
    assert usage.input_tokens == 2
    assert usage.output_tokens == 2
    assert usage.requests == 2


async def test_that_a_hedge_returns_the_first_successful_result_even_if_another_attempt_fails(
    container: Container,
) -> None:
    generator = TestableBaseSchematicGenerator(
        container,
        attempts=[
            _HedgeAttempt(delay=0.15, result="slow-success"),  # first — slow but succeeds
            _HedgeAttempt(delay=0.01, exception=FirstException()),  # hedge — fails fast
        ],
    )

    result = await generator.generate("prompt", hints={HEDGE_TIMEOUT_HINT: 0.05})

    assert generator.call_count == 2
    assert result.content.result == "slow-success"


async def test_that_a_hedge_raises_when_all_attempts_fail(
    container: Container,
) -> None:
    generator = TestableBaseSchematicGenerator(
        container,
        attempts=[
            _HedgeAttempt(delay=0.15, exception=FirstException("first")),
            _HedgeAttempt(delay=0.01, exception=SecondException("second")),
        ],
    )

    with raises(FirstException):
        await generator.generate("prompt", hints={HEDGE_TIMEOUT_HINT: 0.05})

    assert generator.call_count == 2
