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
from typing import Any, Mapping, cast
from typing_extensions import override
from lagom import Container
from unittest.mock import AsyncMock

from pytest import raises

from parlant.core.common import DefaultBaseModel
from parlant.core.engines.alpha.prompt_builder import (
    BuiltInSection,
    PromptBuilder,
    PromptSection,
    SectionStatus,
)
from parlant.core.loggers import Logger
from parlant.core.nlp.embedding import EmbeddingResult
from parlant.core.nlp.generation import (
    FallbackSchematicGenerator,
    SchematicGenerationResult,
    SchematicGenerator,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.tokenization import EstimatingTokenizer, ZeroEstimatingTokenizer


class DummySchema(DefaultBaseModel):
    result: str


class FirstException(Exception):
    pass


class SecondException(Exception):
    pass


async def test_that_fallback_generation_uses_the_first_working_generator(
    container: Container,
) -> None:
    mock_first_generator = AsyncMock(spec=SchematicGenerator[DummySchema])
    mock_first_generator.generate.return_value = SchematicGenerationResult(
        content=DummySchema(result="Success"),
        info=GenerationInfo(
            schema_name="DummySchema",
            model="not-real-model",
            duration=1,
            usage=UsageInfo(
                input_tokens=1,
                output_tokens=1,
            ),
        ),
    )

    mock_second_generator = AsyncMock(spec=SchematicGenerator[DummySchema])

    fallback_generator = FallbackSchematicGenerator[DummySchema](
        mock_first_generator,
        mock_second_generator,
        logger=container[Logger],
    )

    schema_generation_result = await fallback_generator.generate(
        prompt="test prompt", hints={"a": 1}
    )

    mock_first_generator.generate.assert_awaited_once_with(prompt="test prompt", hints={"a": 1})
    mock_second_generator.generate.assert_not_called()

    assert schema_generation_result.content.result == "Success"


async def test_that_fallback_generation_falls_back_to_the_next_generator_when_encountering_an_error_in_the_first_one(
    container: Container,
) -> None:
    mock_first_generator = AsyncMock(spec=SchematicGenerator[DummySchema])
    mock_first_generator.generate.side_effect = Exception("Failure")

    mock_second_generator = AsyncMock(spec=SchematicGenerator[DummySchema])
    mock_second_generator.generate.return_value = SchematicGenerationResult(
        content=DummySchema(result="Success"),
        info=GenerationInfo(
            schema_name="DummySchema",
            model="not-real-model",
            duration=1,
            usage=UsageInfo(
                input_tokens=1,
                output_tokens=1,
            ),
        ),
    )

    fallback_generator = FallbackSchematicGenerator[DummySchema](
        mock_first_generator,
        mock_second_generator,
        logger=container[Logger],
    )

    schema_generation_result = await fallback_generator.generate(
        prompt="test prompt", hints={"a": 1}
    )

    mock_first_generator.generate.assert_awaited_once_with(prompt="test prompt", hints={"a": 1})
    mock_second_generator.generate.assert_awaited_once_with(prompt="test prompt", hints={"a": 1})

    assert schema_generation_result.content.result == "Success"


async def test_that_fallback_generation_raises_an_error_when_all_generators_fail(
    container: Container,
) -> None:
    mock_first_generator = AsyncMock(spec=SchematicGenerator[DummySchema])
    mock_first_generator.generate.side_effect = Exception("Failure")

    mock_second_generator = AsyncMock(spec=SchematicGenerator[DummySchema])
    mock_second_generator.generate.side_effect = Exception("Failure")

    dummy_generator: SchematicGenerator[DummySchema] = FallbackSchematicGenerator(
        mock_first_generator,
        mock_second_generator,
        logger=container[Logger],
    )

    with raises(Exception):
        await dummy_generator.generate("test prompt")

    mock_first_generator.generate.assert_awaited_once_with(prompt="test prompt", hints={})
    mock_second_generator.generate.assert_awaited_once_with(prompt="test prompt", hints={})


async def test_that_retry_succeeds_on_first_attempt(
    container: Container,
) -> None:
    mock_generator = AsyncMock(spec=SchematicGenerator[DummySchema])
    mock_generator.generate.return_value = SchematicGenerationResult(
        content=DummySchema(result="Success"),
        info=GenerationInfo(
            schema_name="DummySchema",
            model="not-real-model",
            duration=1,
            usage=UsageInfo(input_tokens=1, output_tokens=1),
        ),
    )

    @policy([retry(exceptions=(FirstException))])
    async def generate(
        prompt: str, hints: Mapping[str, Any]
    ) -> SchematicGenerationResult[DummySchema]:
        return cast(
            SchematicGenerationResult[DummySchema],
            await mock_generator.generate(prompt=prompt, hints=hints),
        )

    result = await generate(prompt="test prompt", hints={"a": 1})

    mock_generator.generate.assert_awaited_once_with(prompt="test prompt", hints={"a": 1})
    assert result.content.result == "Success"


async def test_that_retry_succeeds_after_failures(
    container: Container,
) -> None:
    mock_generator = AsyncMock(spec=SchematicGenerator[DummySchema])
    success_result = SchematicGenerationResult(
        content=DummySchema(result="Success"),
        info=GenerationInfo(
            schema_name="DummySchema",
            model="not-real-model",
            duration=1,
            usage=UsageInfo(input_tokens=1, output_tokens=1),
        ),
    )

    mock_generator.generate.side_effect = [
        FirstException("First failure"),
        FirstException("Second failure"),
        success_result,
    ]

    @policy([retry(exceptions=(FirstException))])
    async def generate(
        prompt: str, hints: Mapping[str, Any]
    ) -> SchematicGenerationResult[DummySchema]:
        return cast(
            SchematicGenerationResult[DummySchema],
            await mock_generator.generate(prompt=prompt, hints=hints),
        )

    result = await generate(prompt="test prompt", hints={"a": 1})

    assert mock_generator.generate.await_count == 3
    mock_generator.generate.assert_awaited_with(prompt="test prompt", hints={"a": 1})
    assert result.content.result == "Success"


async def test_that_retry_handles_multiple_exception_types(container: Container) -> None:
    class AnotherException(Exception):
        pass

    mock_generator = AsyncMock(spec=SchematicGenerator[DummySchema])
    success_result = SchematicGenerationResult(
        content=DummySchema(result="Success"),
        info=GenerationInfo(
            schema_name="DummySchema",
            model="not-real-model",
            duration=1,
            usage=UsageInfo(input_tokens=1, output_tokens=1),
        ),
    )

    mock_generator.generate.side_effect = [
        FirstException("First error"),
        AnotherException("Second error"),
        success_result,
    ]

    @policy([retry(exceptions=(FirstException, AnotherException), max_exceptions=3)])
    async def generate(
        prompt: str, hints: Mapping[str, Any] = {}
    ) -> SchematicGenerationResult[DummySchema]:
        return cast(
            SchematicGenerationResult[DummySchema], await mock_generator.generate(prompt, hints)
        )

    result = await generate(prompt="test prompt")

    assert mock_generator.generate.await_count == 3
    assert result.content.result == "Success"


async def test_that_retry_doesnt_catch_unspecified_exceptions(container: Container) -> None:
    class UnexpectedException(Exception):
        pass

    mock_generator = AsyncMock(spec=SchematicGenerator[DummySchema])
    mock_generator.generate.side_effect = UnexpectedException("Unexpected error")

    @policy([retry(exceptions=(FirstException), max_exceptions=3)])
    async def generate(
        prompt: str, hints: Mapping[str, Any] = {}
    ) -> SchematicGenerationResult[DummySchema]:
        return cast(
            SchematicGenerationResult[DummySchema], await mock_generator.generate(prompt, hints)
        )

    with raises(UnexpectedException):
        await generate(prompt="test prompt")

    mock_generator.generate.assert_awaited_once()


async def test_that_stacked_retry_decorators_exceed_max_attempts(container: Container) -> None:
    mock_embedder = AsyncMock(spec=EmbeddingResult)
    success_result = EmbeddingResult(vectors=[[0.1, 0.2, 0.3]])

    mock_embedder.side_effect = [
        SecondException("First failure"),
        FirstException("Second failure"),
        FirstException("Third failure"),
        SecondException("Fourth failure"),
        FirstException("Fifth failure"),
        success_result,
    ]

    @policy([retry(SecondException, max_exceptions=3), retry(FirstException, max_exceptions=3)])
    async def embed(text: str) -> EmbeddingResult:
        return cast(EmbeddingResult, await mock_embedder(text=text))

    with raises(FirstException) as exc_info:
        await embed(text="test text")

    assert mock_embedder.await_count == 5
    assert str(exc_info.value) == "Fifth failure"


async def test_that_prompt_builder_edits_are_reflected_in_generation() -> None:
    class MockNLPService(SchematicGenerator[DummySchema]):
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

    mock_service = MockNLPService()
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

    success_result = SchematicGenerationResult(
        content=DummySchema(result="Success"),
        info=GenerationInfo(
            schema_name="DummySchema",
            model="not-real-model",
            duration=1,
            usage=UsageInfo(input_tokens=1, output_tokens=1),
        ),
    )

    private_side_effects = [
        FirstException("First failure"),
        FirstException("Second failure"),
        success_result,
    ]

    @policy(retry(exceptions=(FirstException,)))
    async def generate(
        mock_object: AsyncMock,
        prompt: str,
        hints: Mapping[str, Any],
    ) -> SchematicGenerationResult[DummySchema]:
        return cast(
            SchematicGenerationResult[DummySchema],
            await mock_object.generate(prompt=prompt, hints=hints),
        )

    # Create 5 tasks, each with a different mock object
    tasks = []
    mock_generators = []

    for i in range(concurrency):
        mock_generator = AsyncMock(spec=SchematicGenerator[DummySchema])
        mock_generator.generate.side_effect = private_side_effects

        mock_generators.append(mock_generator)

        tasks.append(generate(mock_object=mock_generator, prompt="test prompt", hints={"a": i}))

    results = await asyncio.gather(*tasks)

    for i in range(concurrency):
        assert mock_generators[i].generate.await_count == 3
        mock_generators[i].generate.assert_awaited_with(prompt="test prompt", hints={"a": i})
        assert results[i].content.result == "Success"
