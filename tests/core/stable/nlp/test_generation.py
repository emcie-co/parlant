# Copyright 2024 Emcie Co Ltd.
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

from typing import Any, Mapping, cast
from lagom import Container
from unittest.mock import AsyncMock

from pytest import raises

from parlant.core.common import DefaultBaseModel
from parlant.core.logging import Logger
from parlant.core.nlp.embedding import EmbeddingResult
from parlant.core.nlp.generation import (
    FallbackSchematicGenerator,
    GenerationInfo,
    SchematicGenerationResult,
    SchematicGenerator,
    UsageInfo,
)
from parlant.core.nlp.policies import policy, retry


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

    @policy([retry(exceptions=(FirstException, AnotherException), max_attempts=3)])
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

    @policy([retry(exceptions=(FirstException), max_attempts=3)])
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

    @policy([retry(SecondException, max_attempts=3), retry(FirstException, max_attempts=3)])
    async def embed(text: str) -> EmbeddingResult:
        return cast(EmbeddingResult, await mock_embedder(text=text))

    with raises(FirstException) as exc_info:
        await embed(text="test text")

    assert mock_embedder.await_count == 5
    assert str(exc_info.value) == "Fifth failure"
