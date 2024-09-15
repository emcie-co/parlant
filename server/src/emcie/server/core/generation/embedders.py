from abc import ABC, abstractmethod
from dataclasses import dataclass
from lagom import Container
from together import AsyncTogether  # type: ignore
from typing import Any, Optional, Sequence
import openai
import os

from emcie.server.logger import Logger


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: Sequence[Sequence[float]]


class Embedder(ABC):
    supported_arguments: list[str] = []

    def __init__(self, logger: Logger) -> None:
        self.logger = logger

    @abstractmethod
    async def embed(
        self,
        texts: list[str],
        hints: Optional[dict[str, Any]] = None,
    ) -> EmbeddingResult:
        pass


class EmbedderFactory:
    def __init__(self, container: Container):
        self._container = container

    def create_embedder(self, embedder_type: type[Embedder]) -> Embedder:
        return self._container[embedder_type]


class OpenAIEmbedder(Embedder):
    supported_arguments = ["dimensions"]

    def __init__(self, logger: Logger) -> None:
        super().__init__(logger=logger)
        self._client = openai.AsyncClient(api_key=os.environ["OPENAI_API_KEY"])

    @abstractmethod
    def _get_model_name(self) -> str:
        pass

    async def embed(
        self,
        texts: list[str],
        hints: Optional[dict[str, Any]] = None,
    ) -> EmbeddingResult:
        filtered_hints = {}
        if hints:
            for k, v in hints.items():
                if k not in self.supported_arguments:
                    self.logger.warning(
                        f"Key '{k}' is not supported in the provided embedding model. Skipping..."
                    )
                    continue
                filtered_hints[k] = v

        response = await self._client.embeddings.create(
            model=self._get_model_name(),
            input=texts,
            **filtered_hints,
        )

        vectors = [data_point.embedding for data_point in response.data]
        return EmbeddingResult(vectors=vectors)


class TogetherAIEmbedder(Embedder, ABC):
    supported_arguments = []

    def __init__(self, logger: Logger) -> None:
        super().__init__(logger=logger)
        self._client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

    @abstractmethod
    def _get_model_name(self) -> str:
        pass

    async def embed(
        self,
        texts: list[str],
        hints: Optional[dict[str, Any]] = None,
    ) -> EmbeddingResult:
        filtered_hints = {}
        if hints:
            for k, v in hints.items():
                if k not in self.supported_arguments:
                    self.logger.warning(
                        f"Key '{k}' is not supported in the the provided embedding model. Skipping..."
                    )
                    continue
                filtered_hints[k] = v

        response = await self._client.embeddings.create(
            model=self._get_model_name(),
            input=texts,
            **filtered_hints,
        )

        vectors = [data_point.embedding for data_point in response.data]
        return EmbeddingResult(vectors=vectors)


class Ada002Embedder(OpenAIEmbedder):
    def _get_model_name(self) -> str:
        return "text-embedding-ada-002"


class Large3Embedder(OpenAIEmbedder):
    def _get_model_name(self) -> str:
        return "text-embedding-3-large"


class Small3Embedder(OpenAIEmbedder):
    def _get_model_name(self) -> str:
        return "text-embedding-3-small"


class M2Bert32K(TogetherAIEmbedder):
    def _get_model_name(self) -> str:
        return "togethercomputer/m2-bert-80M-32k-retrieval"
