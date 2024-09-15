from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
import json
import os
from typing import Any, Generic, Mapping, TypeVar, cast, get_args

import jsonfinder  # type: ignore
from openai import AsyncClient
from together import AsyncTogether  # type: ignore

from emcie.server.base_models import DefaultBaseModel

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
    supported_arguments = ["temperature", "logit_bias", "max_tokens"]

    def __init__(
        self,
        model_name: str,
    ) -> None:
        self.model_name = model_name
        self._client = AsyncClient(api_key=os.environ["OPENAI_API_KEY"])

    async def generate(
        self,
        prompt: str,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        filtered_hints = {k: v for k, v in hints.items() if k in self.supported_arguments}

        response = await self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            response_format={
                "type": "json_object",
            },
            **filtered_hints,
        )

        raw_content = response.choices[0].message.content or "{}"

        try:
            json_content = json.loads(raw_content)
        except json.JSONDecodeError:
            json_content = jsonfinder.only_json(raw_content)[2]

        content = self.schema.model_validate(json_content)
        return SchematicGenerationResult(content=content)


class GPT_4o(OpenAISchematicGenerator[T]):
    def __init__(self) -> None:
        super().__init__(model_name="gpt-4o")


class GPT_4o_Mini(OpenAISchematicGenerator[T]):
    def __init__(self) -> None:
        super().__init__(model_name="gpt-4o-mini")


class TogetherAISchematicGenerator(BaseSchematicGenerator[T]):
    supported_arguments = ["temperature"]

    def __init__(
        self,
        model_name: str,
    ) -> None:
        self.model_name = model_name
        self._client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

    async def generate(
        self,
        prompt: str,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        filtered_hints = {k: v for k, v in hints.items() if k in self.supported_arguments}

        response = await self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            **filtered_hints,
        )

        raw_content = response.choices[0].message.content or "{}"

        json_content = jsonfinder.only_json(raw_content)[2]

        content = self.schema.model_validate(json_content)
        return SchematicGenerationResult(content=content)


class Llama3_1_8B(TogetherAISchematicGenerator[T]):
    def __init__(self) -> None:
        super().__init__(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
