from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import os
from typing import Any, Generic, Optional, Type, TypeVar

import jsonfinder  # type: ignore
from openai import AsyncClient
from together import AsyncTogether  # type: ignore

from emcie.server.base_models import DefaultBaseModel
from emcie.server.logger import Logger

T = TypeVar("T", bound=DefaultBaseModel)


@dataclass(frozen=True)
class SchematicGenerationResult(Generic[T]):
    content: T


class BaseSchematicGenerator(ABC, Generic[T]):
    supported_arguments: list[str] = []

    def __init__(
        self,
        logger: Logger,
        schema: Type[T],
    ) -> None:
        self.logger = logger
        self._schema = schema

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        hints: Optional[dict[str, Any]],
    ) -> SchematicGenerationResult[T]: ...


class OpenAIBaseSchematicGenerator(BaseSchematicGenerator[T], ABC):
    supported_arguments = ["temperature", "logit_bias", "max_tokens"]

    def __init__(
        self,
        logger: Logger,
        schema: Type[T],
    ) -> None:
        super().__init__(logger=logger, schema=schema)
        self._client = AsyncClient(api_key=os.environ["OPENAI_API_KEY"])

    @abstractmethod
    def _get_model_name(self) -> str:
        pass

    async def generate(
        self,
        prompt: str,
        hints: Optional[dict[str, Any]] = None,
    ) -> SchematicGenerationResult[T]:
        filtered_hints = {}
        if hints:
            for k, v in hints.items():
                if k not in self.supported_arguments:
                    self.logger.warning(
                        f"Key '{k}' is not supported in the provided model. Skipping..."
                    )
                    continue
                filtered_hints[k] = v

        response = await self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self._get_model_name(),
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

        content = self._schema.model_validate(json_content)
        return SchematicGenerationResult(content=content)


class TogetherAIBaseSchematicGenerator(BaseSchematicGenerator[T], ABC):
    supported_arguments = ["temperature"]

    def __init__(
        self,
        logger: Logger,
        schema: Type[T],
    ) -> None:
        self.logger = logger
        self._schema = schema
        self._client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

    @abstractmethod
    def _get_model_name(self) -> str:
        pass

    async def generate(
        self,
        prompt: str,
        hints: Optional[dict[str, Any]] = None,
    ) -> SchematicGenerationResult[T]:
        filtered_hints = {}
        if hints:
            for k, v in hints.items():
                if k not in self.supported_arguments:
                    self.logger.warning(
                        f"Key '{k}' is not supported in the provided model. Skipping..."
                    )
                    continue
                filtered_hints[k] = v

        response = await self._client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=self._get_model_name(),
            **filtered_hints,
        )

        raw_content = response.choices[0].message.content or "{}"

        try:
            json_content = json.loads(raw_content)
        except json.JSONDecodeError:
            json_content = jsonfinder.only_json(raw_content)[2]

        content = self._schema.model_validate(json_content)
        return SchematicGenerationResult(content=content)


class GPT4o(OpenAIBaseSchematicGenerator[T]):
    def _get_model_name(self) -> str:
        return "gpt-4o"


class GPT4oMini(OpenAIBaseSchematicGenerator[T]):
    def _get_model_name(self) -> str:
        return "gpt-4o-mini"


class Llama3_1_8B(TogetherAIBaseSchematicGenerator[T]):
    def _get_model_name(self) -> str:
        return "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
