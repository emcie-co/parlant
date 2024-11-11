from __future__ import annotations
from itertools import chain
import time
from openai import AsyncClient
from typing import Any, Mapping, Type
import json
import jsonfinder  # type: ignore
import os

from pydantic import ValidationError
import tiktoken

from parlant.core.engines.alpha.tool_caller import ToolCallInferenceSchema
from parlant.core.logging import Logger
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.service import NLPService
from parlant.core.nlp.embedding import Embedder, EmbeddingResult
from parlant.core.nlp.generation import (
    T,
    BaseSchematicGenerator,
    FallbackSchematicGenerator,
    GenerationInfo,
    SchematicGenerationResult,
    UsageInfo,
)
from parlant.core.nlp.moderation import ModerationCheck, ModerationService, ModerationTag


class OpenAIEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model(model_name)

    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return len(tokens)


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

        self._tokenizer = OpenAIEstimatingTokenizer(model_name=self.model_name)

    @property
    def id(self) -> str:
        return f"openai/{self.model_name}"

    @property
    def tokenizer(self) -> OpenAIEstimatingTokenizer:
        return self._tokenizer

    async def generate(
        self,
        prompt: str,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        with self._logger.operation(f"OpenAI LLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    async def _do_generate(
        self,
        prompt: str,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        openai_api_arguments = {k: v for k, v in hints.items() if k in self.supported_openai_params}

        if hints.get("strict", False):
            t_start = time.time()
            response = await self._client.beta.chat.completions.parse(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                response_format=self.schema,
                **openai_api_arguments,
            )
            t_end = time.time()

            if response.usage:
                self._logger.debug(response.usage.model_dump_json(indent=2))

            parsed_object = response.choices[0].message.parsed
            assert parsed_object

            assert response.usage
            assert response.usage.prompt_tokens_details

            return SchematicGenerationResult[T](
                content=parsed_object,
                info=GenerationInfo(
                    schema_name=self.schema.__name__,
                    model=self.id,
                    duration=(t_end - t_start),
                    usage=UsageInfo(
                        input_tokens=response.usage.prompt_tokens,
                        output_tokens=response.usage.completion_tokens,
                        extra={
                            "cached_input_tokens": response.usage.prompt_tokens_details.cached_tokens
                            or 0
                        },
                    ),
                ),
            )

        else:
            t_start = time.time()
            response = await self._client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                response_format={"type": "json_object"},
                **openai_api_arguments,
            )
            t_end = time.time()

            if response.usage:
                self._logger.debug(response.usage.model_dump_json(indent=2))

            raw_content = response.choices[0].message.content or "{}"

            try:
                json_content = json.loads(raw_content)
            except json.JSONDecodeError:
                self._logger.warning(f"Invalid JSON returned by {self.model_name}:\n{raw_content}")
                json_content = jsonfinder.only_json(raw_content)[2]
                self._logger.warning("Found JSON content within model response; continuing...")

            try:
                content = self.schema.model_validate(json_content)

                assert response.usage
                assert response.usage.prompt_tokens_details

                return SchematicGenerationResult(
                    content=content,
                    info=GenerationInfo(
                        schema_name=self.schema.__name__,
                        model=self.id,
                        duration=(t_end - t_start),
                        usage=UsageInfo(
                            input_tokens=response.usage.prompt_tokens,
                            output_tokens=response.usage.completion_tokens,
                            extra={
                                "cached_input_tokens": response.usage.prompt_tokens_details.cached_tokens
                                or 0
                            },
                        ),
                    ),
                )
            except ValidationError:
                self._logger.error(
                    f"JSON content returned by {self.model_name} does not match expected schema:\n{raw_content}"
                )
                raise


class GPT_4o(OpenAISchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(model_name="gpt-4o-2024-08-06", logger=logger)

    @property
    def max_tokens(self) -> int:
        return 128_000


class GPT_4o_Mini(OpenAISchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(model_name="gpt-4o-mini", logger=logger)
        self._token_estimator = OpenAIEstimatingTokenizer(model_name=self.model_name)

    @property
    def max_tokens(self) -> int:
        return 128_000


class OpenAIEmbedder(Embedder):
    supported_arguments = ["dimensions"]

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._client = AsyncClient(api_key=os.environ["OPENAI_API_KEY"])
        self._tokenizer = OpenAIEstimatingTokenizer(model_name=self.model_name)

    @property
    def id(self) -> str:
        return f"openai/{self.model_name}"

    @property
    def tokenizer(self) -> OpenAIEstimatingTokenizer:
        return self._tokenizer

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

    @property
    def max_tokens(self) -> int:
        return 8192


class OpenAITextEmbedding3Small(OpenAIEmbedder):
    def __init__(self) -> None:
        super().__init__(model_name="text-embedding-3-small")

    @property
    def max_tokens(self) -> int:
        return 8192


class OpenAIModerationService(ModerationService):
    def __init__(self, model_name: str, logger: Logger) -> None:
        self.model_name = model_name
        self._logger = logger

        self._client = AsyncClient(api_key=os.environ["OPENAI_API_KEY"])

    async def check(self, content: str) -> ModerationCheck:
        def extract_tags(category: str) -> list[ModerationTag]:
            mapping: dict[str, list[ModerationTag]] = {
                "sexual": ["sexual"],
                "sexual_minors": ["sexual", "illicit"],
                "harassment": ["harassment"],
                "harassment_threatening": ["harassment", "illicit"],
                "hate": ["hate"],
                "hate_threatening": ["hate", "illicit"],
                "illicit": ["illicit"],
                "illicit_violent": ["illicit", "violence"],
                "self_harm": ["self-harm"],
                "self_harm_intent": ["self-harm", "violence"],
                "self_harm_instructions": ["self-harm", "illicit"],
                "violence": ["violence"],
                "violence_graphic": ["violence", "harassment"],
            }

            return mapping.get(category.replace("/", "_").replace("-", "_"), [])

        with self._logger.operation("OpenAI Moderation Request"):
            response = await self._client.moderations.create(
                input=content,
                model=self.model_name,
            )

        result = response.results[0]

        return ModerationCheck(
            flagged=result.flagged,
            tags=list(
                set(
                    chain.from_iterable(
                        extract_tags(category)
                        for category, detected in result.categories
                        if detected
                    )
                )
            ),
        )


class OmniModeration(OpenAIModerationService):
    def __init__(self, logger: Logger) -> None:
        super().__init__(model_name="omni-moderation-latest", logger=logger)


class OpenAIService(NLPService):
    def __init__(
        self,
        logger: Logger,
    ) -> None:
        self._logger = logger

    async def get_schematic_generator(self, t: Type[T]) -> OpenAISchematicGenerator[T]:
        if t == ToolCallInferenceSchema:
            return FallbackSchematicGenerator(
                GPT_4o_Mini[t](self._logger),  # type: ignore
                GPT_4o[t](self._logger),  # type: ignore
                logger=self._logger,
            )
        return GPT_4o[t](self._logger)  # type: ignore

    async def get_embedder(self) -> Embedder:
        return OpenAITextEmbedding3Large()

    async def get_moderation_service(self) -> ModerationService:
        return OmniModeration(self._logger)
