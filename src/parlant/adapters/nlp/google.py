import os
import time
import google.generativeai as genai  # type: ignore
from typing import Any, Mapping
import jsonfinder  # type: ignore
from pydantic import ValidationError
from parlant.core.nlp.moderation import ModerationService, NoModeration
from parlant.core.nlp.service import NLPService
from vertexai.preview import tokenization  # type: ignore

from parlant.core.logging import Logger
from parlant.core.nlp.embedding import Embedder, EmbeddingResult
from parlant.core.nlp.generation import T, BaseSchematicGenerator, GenerationInfo, SchematicGenerationResult, TokenEstimator


genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


class GoogleTokenEstimator(TokenEstimator):
    def __init__(self, model_name: str) -> None:
        self._tokenizer = tokenization.get_tokenizer_for_model(model_name)

    async def estimate_token_count(self, prompt: str) -> int:
        result = self._tokenizer.count_tokens(prompt)
        return int(result.total_tokens)


class GeminiSchematicGenerator(BaseSchematicGenerator[T]):
    supported_hints = ["temperature"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
    ) -> None:
        self.model_name = model_name
        self._logger = logger

        self._model = genai.GenerativeModel(model_name)

        self._token_estimator = GoogleTokenEstimator(model_name=self.model_name)

    @property
    def id(self) -> str:
        return f"google/{self.model_name}"

    @property
    def token_estimator(self) -> TokenEstimator:
        return self._token_estimator

    async def generate(
        self,
        prompt: str,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        gemini_api_arguments = {k: v for k, v in hints.items() if k in self.supported_hints}

        t_start = time.time()
        response = await self._model.generate_content_async(
            contents=prompt,
            generation_config={"temperature": gemini_api_arguments.pop("temperature")},
            **gemini_api_arguments,
        )
        t_end = time.time()

        raw_content = response.text

        try:
            json_start = max(0, raw_content.find("```json"))

            if json_start != -1:
                json_start = json_start + 7

            json_end = raw_content[json_start:].find("```")

            if json_end == -1:
                json_end = len(raw_content[json_start:])

            json_content = raw_content[json_start : json_start + json_end]

            json_content = json_content.replace("“", '"').replace("”", '"')

            json_object = jsonfinder.only_json(json_content)[2]
        except Exception:
            self._logger.error(
                f"Failed to extract JSON returned by {self.model_name}:\n{raw_content}"
            )
            raise

        try:
            model_content = self.schema.model_validate(json_object)
            return SchematicGenerationResult(
                content=model_content,
                info=GenerationInfo(
                    schema_name=self.schema.__name__, model=self.id, duration=(t_end - t_start)
                ),
            )
        except ValidationError:
            self._logger.error(
                f"JSON content returned by {self.model_name} does not match expected schema:\n{raw_content}"
            )
            raise


class Gemini_1_5_Flash(GeminiSchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(
            model_name="gemini-1.5-flash",
            logger=logger,
        )

    @property
    def max_tokens(self) -> int:
        return 1000000


class Gemini_1_5_Pro(GeminiSchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(
            model_name="gemini-1.5-pro",
            logger=logger,
        )

    @property
    def max_tokens(self) -> int:
        return 2000000


class GeminiEmbedder(Embedder):
    supported_hints = ["title", "task_type"]

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._token_estimator = GoogleTokenEstimator(model_name=self.model_name)

    @property
    def id(self) -> str:
        return f"google/{self.model_name}"

    @property
    def token_estimator(self) -> TokenEstimator:
        return self._token_estimator

    async def embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        gemini_api_arguments = {k: v for k, v in hints.items() if k in self.supported_hints}

        response = await genai.embed_content_async(
            model=self.model_name,
            content=texts,
            **gemini_api_arguments,
        )

        vectors = [data_point for data_point in response["embedding"]]
        return EmbeddingResult(vectors=vectors)


class GeminiTextEmbedding_004(GeminiEmbedder):
    def __init__(self) -> None:
        super().__init__(model_name="models/text-embedding-004")

    @property
    def max_tokens(self) -> int:
        return 2000000


class GoogleService(NLPService):
    def __init__(
        self,
        logger: Logger,
    ) -> None:
        self._logger = logger

    async def get_schematic_generator(self, t: type[T]) -> GeminiSchematicGenerator[T]:
        return Gemini_1_5_Pro[T](self._logger)

    async def get_embedder(self) -> Embedder:
        return GeminiTextEmbedding_004()

    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
