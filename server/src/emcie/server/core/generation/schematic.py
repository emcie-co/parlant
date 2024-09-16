from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
import json
import os
from typing import Any, Generic, Mapping, TypeVar, cast, get_args

import jsonfinder  # type: ignore
from openai import AsyncClient
from pydantic import ValidationError
from together import AsyncTogether  # type: ignore

from emcie.server.base_models import DefaultBaseModel
from emcie.server.logger import Logger

T = TypeVar("T", bound=DefaultBaseModel)


@dataclass(frozen=True)
class SchematicGenerationResult(Generic[T]):
    content: T


class SchematicGenerator(ABC, Generic[T]):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]: ...


class BaseSchematicGenerator(SchematicGenerator[T]):
    @cached_property
    def schema(self) -> type[T]:
        orig_class = getattr(self, "__orig_class__")
        generic_args = get_args(orig_class)
        return cast(type[T], generic_args[0])


class OpenAISchematicGenerator(BaseSchematicGenerator[T]):
    supported_openai_params = ["temperature", "logit_bias", "max_tokens"]
    supported_hints = supported_openai_params + ["strict"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
    ) -> None:
        self.model_name = model_name
        self._logger = logger
        self._client = AsyncClient(api_key=os.environ["OPENAI_API_KEY"])

    async def generate(
        self,
        prompt: str,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        openai_api_arguments = {k: v for k, v in hints.items() if k in self.supported_openai_params}

        if hints.get("strict", False):
            response = await self._client.beta.chat.completions.parse(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                response_format=self.schema,
                **openai_api_arguments,
            )

            parsed_object = response.choices[0].message.parsed
            assert parsed_object

            return SchematicGenerationResult[T](content=parsed_object)

        else:
            response = await self._client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                response_format={"type": "json_object"},
                **openai_api_arguments,
            )

            raw_content = response.choices[0].message.content or "{}"

            try:
                json_content = json.loads(raw_content)
            except json.JSONDecodeError:
                self._logger.warning(f"Invalid JSON returned by {self.model_name}:\n{raw_content}")
                json_content = jsonfinder.only_json(raw_content)[2]
                self._logger.warning("Found JSON content within model response; continuing...")

            try:
                content = self.schema.model_validate(json_content)
                return SchematicGenerationResult(content=content)
            except ValidationError:
                self._logger.error(
                    f"JSON content returned by {self.model_name} does not match expected schema:\n{raw_content}"
                )
                raise


class GPT_4o(OpenAISchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(model_name="gpt-4o-2024-08-06", logger=logger)


class GPT_4o_Mini(OpenAISchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(model_name="gpt-4o-mini", logger=logger)


class TogetherAISchematicGenerator(BaseSchematicGenerator[T]):
    supported_hints = ["temperature"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
    ) -> None:
        self.model_name = model_name
        self._logger = logger
        self._client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

    async def generate(
        self,
        prompt: str,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        together_api_arguments = {k: v for k, v in hints.items() if k in self.supported_hints}

        response = await self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            **together_api_arguments,
        )

        raw_content = response.choices[0].message.content or "{}"

        try:
            json_content = jsonfinder.only_json(raw_content)[2]
        except Exception:
            self._logger.error(
                f"Failed to extract JSON returned by {self.model_name}:\n{raw_content}"
            )
            raise

        try:
            content = self.schema.model_validate(json_content)
            return SchematicGenerationResult(content=content)
        except ValidationError:
            self._logger.error(
                f"JSON content returned by {self.model_name} does not match expected schema:\n{raw_content}"
            )
            raise


class Llama3_1_8B(TogetherAISchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            logger=logger,
        )
