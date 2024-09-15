from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from lagom import Container
from together import AsyncTogether  # type: ignore
from typing import Any, Sequence
import openai
import os


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: Sequence[Sequence[float]]


class Embedder(ABC):
    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        pass


class EmbedderFactory:
    def __init__(self, container: Container):
        self._container = container

    def create_embedder(self, embedder_type: type[Embedder]) -> Embedder:
        return self._container[embedder_type]


class OpenAIEmbedder(Embedder):
    supported_arguments = ["dimensions"]

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._client = openai.AsyncClient(api_key=os.environ["OPENAI_API_KEY"])

    async def embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        filtered_hints = {k: v for k, v in hints.items() if k in self.supported_arguments}

        response = await self._client.embeddings.create(
            model=self.model_name,
            input=texts,
            **filtered_hints,
        )

        vectors = [data_point.embedding for data_point in response.data]
        return EmbeddingResult(vectors=vectors)


class OpenAITextEmbedding3Large(OpenAIEmbedder):
    def __init__(self) -> None:
        super().__init__(model_name="text-embedding-3-large")


class OpenAITextEmbedding3Small(OpenAIEmbedder):
    def __init__(self) -> None:
        super().__init__(model_name="text-embedding-3-small")


class TogetherAIEmbedder(Embedder, ABC):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

    async def embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        _ = hints

        response = await self._client.embeddings.create(
            model=self.model_name,
            input=texts,
        )

        vectors = [data_point.embedding for data_point in response.data]
        return EmbeddingResult(vectors=vectors)


class M2Bert32K(TogetherAIEmbedder):
    def __init__(self) -> None:
        super().__init__(model_name="togethercomputer/m2-bert-80M-32k-retrieval")
