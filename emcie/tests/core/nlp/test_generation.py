from lagom import Container
from unittest.mock import AsyncMock

from pytest import raises

from emcie.server.core.common import DefaultBaseModel
from emcie.server.core.logging import Logger
from emcie.server.core.nlp.generation import (
    FallbackSchematicGenerator,
    SchematicGenerationResult,
    SchematicGenerator,
)


class DummySchema(DefaultBaseModel):
    result: str


async def test_that_fallback_generation_uses_the_first_working_generator(
    container: Container,
) -> None:
    mock_first_generator = AsyncMock(spec=SchematicGenerator[DummySchema])
    mock_first_generator.generate.return_value = SchematicGenerationResult(
        content=DummySchema(result="Success")
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
        content=DummySchema(result="Success")
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


async def test_that_fallback_generation_raise_an_error_when_all_generators_fail(
    container: Container,
) -> None:
    mock_first_generator = AsyncMock(spec=SchematicGenerator[DummySchema])
    mock_first_generator.generate.side_effect = Exception("Failure")

    mock_second_generator = AsyncMock(spec=SchematicGenerator[DummySchema])
    mock_second_generator.generate.side_effect = Exception("Failure")

    fallback_generator: SchematicGenerator[DummySchema] = FallbackSchematicGenerator(
        mock_first_generator,
        mock_second_generator,
        logger=container[Logger],
    )

    with raises(Exception):
        await fallback_generator.generate("test prompt")

    mock_first_generator.generate.assert_awaited_once_with(prompt="test prompt", hints={})
    mock_second_generator.generate.assert_awaited_once_with(prompt="test prompt", hints={})
