import os
import google.generativeai as genai  # type: ignore
from typing import Any, Mapping
import jsonfinder  # type: ignore
from pydantic import ValidationError

from emcie.server.core.logging import Logger
from emcie.server.core.nlp.embedding import Embedder, EmbeddingResult
from emcie.server.core.nlp.generation import T, BaseSchematicGenerator, SchematicGenerationResult


genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


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

    async def generate(
        self,
        prompt: str,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        gemini_api_arguments = {k: v for k, v in hints.items() if k in self.supported_hints}

        response = await self._model.generate_content_async(
            contents=prompt,
            generation_config={"temperature": gemini_api_arguments.pop("temperature")},
            **gemini_api_arguments,
        )

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
            return SchematicGenerationResult(content=model_content)
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


class Gemini_1_5_Pro(GeminiSchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(
            model_name="gemini-1.5-pro",
            logger=logger,
        )


class GeminiEmbedder(Embedder):
    supported_hints = ["title", "task_type"]

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

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
