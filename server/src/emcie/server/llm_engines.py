from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import jsonfinder  # type: ignore
import os
from typing import Any, Generic, Type, TypeVar

from openai import AsyncClient
from together import AsyncTogether  # type: ignore

from emcie.server.base_models import DefaultBaseModel

T = TypeVar("T", bound=DefaultBaseModel)


@dataclass(frozen=True)
class JSONGenerationResult(Generic[T]):
    content: T


class JSONGenerator(ABC, Generic[T]):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        args: dict[str, Any],
    ) -> JSONGenerationResult[T]: ...


class GPT4o(JSONGenerator[T]):
    def __init__(self, schema: Type[T]) -> None:
        self._llm_client = AsyncClient(api_key=os.environ["OPENAI_API_KEY"])
        self._schema = schema

    async def generate(
        self,
        prompt: str,
        args: dict[str, Any],
    ) -> JSONGenerationResult[T]:
        response = await self._llm_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o",
            response_format={"type": "json_object"},
            **args,
        )

        raw_content = response.choices[0].message.content or "{}"

        json_content = json.loads(raw_content)

        content = self._schema.model_validate(json_content)

        return JSONGenerationResult(content=content)


class Llama3_1_8B(JSONGenerator[T]):
    def __init__(self, schema: Type[T]) -> None:
        self._llm_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))
        self._schema = schema

    async def generate(
        self,
        prompt: str,
        args: dict[str, Any],
    ) -> JSONGenerationResult[T]:
        response = await self._llm_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            response_format={
                "type": "json_object",
            },
            **args,
        )

        raw_content = response.choices[0].message.content or "{}"

        json_content = jsonfinder.only_json(raw_content)[2]

        content = self._schema.model_validate(json_content)

        return JSONGenerationResult(content=content)
