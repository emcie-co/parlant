from abc import ABC, abstractmethod
import os
from typing import Any

from openai import AsyncClient


class TextGenerator(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        args: dict[str, Any],
    ) -> str: ...


class GPT4o(TextGenerator):
    def __init__(self) -> None:
        self._llm_client = AsyncClient(api_key=os.environ["OPENAI_API_KEY"])

    async def generate(
        self,
        prompt: str,
        args: dict[str, Any],
    ) -> str:
        response = await self._llm_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o",
            **args,
        )

        return str(response.choices[0].message.content)
