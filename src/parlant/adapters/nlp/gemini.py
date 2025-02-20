# Copyright 2024 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import google.generativeai as genai  # type: ignore
from google.api_core.exceptions import NotFound, TooManyRequests, ResourceExhausted, ServerError
from typing import Any, Mapping
from typing_extensions import override
import jsonfinder  # type: ignore
from pydantic import ValidationError
from vertexai.preview import tokenization  # type: ignore

from parlant.adapters.nlp.common import normalize_json_output
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.moderation import ModerationService, NoModeration
from parlant.core.nlp.service import NLPService
from parlant.core.nlp.embedding import Embedder, EmbeddingResult
from parlant.core.nlp.generation import (
    T,
    SchematicGenerator,
    FallbackSchematicGenerator,
    GenerationInfo,
    SchematicGenerationResult,
    UsageInfo,
)
from parlant.core.logging import Logger


genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))  # type: ignore


class GoogleEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, model_name: str) -> None:
        self._tokenizer = tokenization.get_tokenizer_for_model("gemini-1.5-pro")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        result = self._tokenizer.count_tokens(prompt)
        return int(result.total_tokens)


class GeminiSchematicGenerator(SchematicGenerator[T]):
    supported_hints = ["temperature"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
    ) -> None:
        self.model_name = model_name
        self._logger = logger

        self._model = genai.GenerativeModel(model_name)  # type: ignore

        self._tokenizer = GoogleEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"google/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> EstimatingTokenizer:
        return self._tokenizer

    @policy(
        [
            retry(
                exceptions=(
                    NotFound,
                    TooManyRequests,
                    ResourceExhausted,
                )
            ),
            retry(ServerError, max_attempts=2, wait_times=(1.0, 5.0)),
        ]
    )
    @override
    async def generate(
        self,
        prompt: str,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        gemini_api_arguments = {k: v for k, v in hints.items() if k in self.supported_hints}

        t_start = time.time()
        response = await self._model.generate_content_async(
            contents=prompt,
            generation_config=gemini_api_arguments,  # type: ignore
        )
        t_end = time.time()

        raw_content = response.text

        try:
            json_content = normalize_json_output(raw_content)
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
                    schema_name=self.schema.__name__,
                    model=self.id,
                    duration=(t_end - t_start),
                    usage=UsageInfo(
                        input_tokens=response.usage_metadata.prompt_token_count,
                        output_tokens=response.usage_metadata.candidates_token_count,
                        extra={
                            "cached_input_tokens": response.usage_metadata.cached_content_token_count
                        },
                    ),
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
    @override
    def max_tokens(self) -> int:
        return 1024 * 1024


class Gemini_2_0_Flash(GeminiSchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(
            model_name="gemini-2.0-flash",
            logger=logger,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 1024 * 1024


class Gemini_1_5_Pro(GeminiSchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(
            model_name="gemini-1.5-pro",
            logger=logger,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 2 * 1024 * 1024


class GoogleEmbedder(Embedder):
    supported_hints = ["title", "task_type"]

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._tokenizer = GoogleEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"google/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> GoogleEstimatingTokenizer:
        return self._tokenizer

    @policy(
        [
            retry(
                exceptions=(
                    NotFound,
                    TooManyRequests,
                    ResourceExhausted,
                )
            ),
            retry(ServerError, max_attempts=2, wait_times=(1.0, 5.0)),
        ]
    )
    @override
    async def embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        gemini_api_arguments = {k: v for k, v in hints.items() if k in self.supported_hints}

        response = await genai.embed_content_async(  # type: ignore
            model=self.model_name,
            content=texts,
            **gemini_api_arguments,
        )

        vectors = [data_point for data_point in response["embedding"]]
        return EmbeddingResult(vectors=vectors)


class GeminiTextEmbedding_004(GoogleEmbedder):
    def __init__(self) -> None:
        super().__init__(model_name="models/text-embedding-004")

    @property
    @override
    def max_tokens(self) -> int:
        return 8000

    @property
    def dimensions(self) -> int:
        return 768


class GeminiService(NLPService):
    def __init__(
        self,
        logger: Logger,
    ) -> None:
        self._logger = logger
        self._logger.info("Initialized GeminiService")

    @override
    async def get_schematic_generator(self, t: type[T]) -> GeminiSchematicGenerator[T]:
        return FallbackSchematicGenerator[t](  # type: ignore
            Gemini_2_0_Flash[t](self._logger),  # type: ignore
            Gemini_1_5_Pro[t](self._logger),  # type: ignore
            logger=self._logger,
        )

    @override
    async def get_embedder(self) -> Embedder:
        return GeminiTextEmbedding_004()

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
