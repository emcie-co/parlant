# Copyright 2025 Emcie Co Ltd.
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

from __future__ import annotations
import json
import os
import time
from typing import Any, Mapping, TypeVar
from typing_extensions import override
from openai import AsyncClient, APIConnectionError, APITimeoutError, RateLimitError, BadRequestError
from pydantic import ValidationError
from sentence_transformers import SentenceTransformer
from pydantic.json_schema import model_json_schema

from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.loggers import Logger
from parlant.core.nlp.embedding import Embedder, EmbeddingResult
from parlant.core.nlp.generation import SchematicGenerator, SchematicGenerationResult, T
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.nlp.moderation import ModerationCheck, ModerationService
from parlant.core.nlp.service import NLPService
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.adapters.nlp.common import normalize_json_output
import jsonfinder

class OllamaEstimatingTokenizer(EstimatingTokenizer):
    """A simple tokenizer for estimating token count for local models."""
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.chars_per_token = 4

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        return max(1, len(prompt) // self.chars_per_token)

class OllamaSchematicGenerator(SchematicGenerator[T]):
    """Schematic generator for Ollama models."""
    supported_ollama_params = ["temperature", "max_tokens"]
    supported_hints = supported_ollama_params + ["strict"]

    def __init__(self, model_name: str, logger: Logger, base_url: str = "http://localhost:11434/v1") -> None:
        self.model_name = model_name
        self._logger = logger
        self._client = AsyncClient(
            base_url=os.environ.get("OLLAMA_API_BASE", base_url),
            api_key="ollama"
        )
        self._tokenizer = OllamaEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"ollama/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> OllamaEstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

    @override
    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        with self._logger.scope("OllamaSchematicGenerator"):
            with self._logger.operation(f"LLM Request ({self.schema.__name__})"):
                return await self._do_generate(prompt, hints)

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        ollama_api_arguments = {k: v for k, v in hints.items() if k in self.supported_ollama_params}

        try:
            t_start = time.time()
            response = await self._client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                response_format={"type": "json_object"},
                **ollama_api_arguments,
            )
            t_end = time.time()
        except (APIConnectionError, APITimeoutError, RateLimitError, BadRequestError) as e:
            self._logger.error(f"Ollama API error: {str(e)}")
            raise

        raw_content = response.choices[0].message.content or "{}"

        try:
            json_content = json.loads(normalize_json_output(raw_content))
        except json.JSONDecodeError:
            self._logger.warning(f"Invalid JSON returned by {self.model_name}:\n{raw_content}")
            json_content = jsonfinder.only_json(raw_content)[2]
            self._logger.warning("Found JSON content within model response; continuing...")

        try:
            content = self.schema.model_validate(json_content)
            input_tokens = await self._tokenizer.estimate_token_count(prompt)
            output_tokens = await self._tokenizer.estimate_token_count(raw_content)

            return SchematicGenerationResult(
                content=content,
                info=GenerationInfo(
                    schema_name=self.schema.__name__,
                    model=self.id,
                    duration=(t_end - t_start),
                    usage=UsageInfo(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        extra={},
                    ),
                ),
            )
        except ValidationError as e:
            self._logger.error(
                f"Error: {e.json(indent=2)}\nJSON content returned by {self.model_name} does not match expected schema:\n{raw_content}"
            )
            raise

class LMStudioSchematicGenerator(OllamaSchematicGenerator[T]):
    """Schematic generator for LM Studio models."""
    def __init__(self, model_name: str, logger: Logger) -> None:
        super().__init__(model_name=model_name, logger=logger, base_url="http://localhost:1234/v1")

    @property
    @override
    def id(self) -> str:
        return f"lmstudio/{self.model_name}"

    @override
    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        # Format prompt for Gemma 3
        formatted_prompt = (
            f"<start_of_turn>user\n{prompt}\n<end_of_turn>\n<start_of_turn>assistant\n"
        )

        lmstudio_api_arguments = {k: v for k, v in hints.items() if k in self.supported_ollama_params}

        # Simplified JSON schema
        json_schema = {
            "name": self.schema.__name__,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "produced_reply": {"type": "boolean"},
                    "produced_reply_rationale": {"type": ["string", "null"]},
                    "revisions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "revision_number": {"type": "integer"},
                                "content": {"type": "string"},
                                "is_repeat_message": {"type": "boolean"},
                                "followed_all_instructions": {"type": "boolean"},
                                "all_facts_and_services_sourced_from_prompt": {"type": "boolean"},
                                "further_revisions_required": {"type": "boolean"}
                            },
                            "required": ["revision_number", "content", "is_repeat_message", "followed_all_instructions"],
                            "additionalProperties": True
                        }
                    }
                },
                "required": ["produced_reply", "revisions"],
                "additionalProperties": True
            }
        }

        try:
            t_start = time.time()
            self._logger.debug(f"Sending LM Studio request: model={self.model_name}, prompt={formatted_prompt}, schema={json.dumps(json_schema, indent=2)}")
            response = await self._client.chat.completions.create(
                messages=[{"role": "user", "content": formatted_prompt}],
                model=self.model_name,
                response_format={"type": "json_schema", "json_schema": json_schema},
                **lmstudio_api_arguments,
            )
            t_end = time.time()

            raw_content = response.choices[0].message.content or "{}"
            self._logger.debug(f"Raw response: {raw_content}")

            try:
                json_content = json.loads(normalize_json_output(raw_content))
            except json.JSONDecodeError:
                self._logger.warning(f"Invalid JSON returned by {self.model_name}:\n{raw_content}")
                json_content = jsonfinder.only_json(raw_content)[2]
                self._logger.warning("Found JSON content within model response; continuing...")

            try:
                content = self.schema.model_validate(json_content)
                input_tokens = response.usage.prompt_tokens if response.usage else await self._tokenizer.estimate_token_count(formatted_prompt)
                output_tokens = response.usage.completion_tokens if response.usage else await self._tokenizer.estimate_token_count(raw_content)

                return SchematicGenerationResult(
                    content=content,
                    info=GenerationInfo(
                        schema_name=self.schema.__name__,
                        model=self.id,
                        duration=(t_end - t_start),
                        usage=UsageInfo(
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            extra={},
                        ),
                    ),
                )
            except ValidationError as e:
                self._logger.error(
                    f"Validation error: {e.json(indent=2)}\nJSON content: {json.dumps(json_content, indent=2)}"
                )
                raise
        except (APIConnectionError, APITimeoutError, RateLimitError, BadRequestError) as e:
            self._logger.error(f"LM Studio API error: {str(e)}, response: {getattr(e, 'response', 'No response')}")
            raise

class OllamaLlama3Generator(OllamaSchematicGenerator[T]):
    """Schematic generator for Llama 3 model via Ollama."""
    def __init__(self, logger: Logger) -> None:
        super().__init__(model_name="llama3", logger=logger)

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

class LMStudioGemma3Generator(LMStudioSchematicGenerator[T]):
    """Schematic generator for Gemma 3 4B Instruct QAT model via LM Studio."""
    def __init__(self, logger: Logger) -> None:
        super().__init__(model_name="gemma-3-4b-it-qat", logger=logger)

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

class OllamaEmbedder(Embedder):
    """Base embedder class using sentence-transformers for local embeddings."""
    supported_arguments = ["dimensions"]

    def __init__(self, model_name: str, logger: Logger) -> None:
        self.model_name = model_name
        self._logger = logger
        self._model = SentenceTransformer(model_name)
        self._tokenizer = OllamaEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"sentence-transformers/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> OllamaEstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        return 8192

    @property
    @override
    def dimensions(self) -> int:
        return 384

    @override
    async def embed(self, texts: list[str], hints: Mapping[str, Any] = {}) -> EmbeddingResult:
        filtered_hints = {k: v for k, v in hints.items() if k in self.supported_arguments}
        try:
            vectors = self._model.encode(texts, convert_to_numpy=True, **filtered_hints).tolist()
            return EmbeddingResult(vectors=vectors)
        except Exception as e:
            self._logger.error(f"Embedding error: {str(e)}")
            raise

class OllamaMiniLMEmbedder(OllamaEmbedder):
    """Embedder for all-MiniLM-L6-v2 model."""
    def __init__(self, logger: Logger) -> None:
        super().__init__(model_name="all-MiniLM-L6-v2", logger=logger)

    @property
    @override
    def dimensions(self) -> int:
        return 384

class OllamaModerationService(ModerationService):
    """Placeholder moderation service for Ollama/LM Studio."""
    def __init__(self, logger: Logger) -> None:
        self._logger = logger
        self._logger.warning(
            "Local models do not provide a moderation service. Consider integrating a dedicated moderation model."
        )

    @override
    async def check(self, content: str) -> ModerationCheck:
        self._logger.error("Moderation not supported by local models.")
        raise NotImplementedError("Local models do not support moderation. Consider using a dedicated moderation service.")

class OllamaService(NLPService):
    """NLPService implementation for Ollama or LM Studio."""
    def __init__(self, logger: Logger, use_lm_studio: bool = False) -> None:
        self._logger = logger
        self._use_lm_studio = use_lm_studio or os.environ.get("USE_LM_STUDIO", "false").lower() == "true"
        self._logger.info(f"Initialized OllamaService (use_lm_studio={self._use_lm_studio})")

    @override
    async def get_schematic_generator(self, t: type[T]) -> OllamaSchematicGenerator[T]:
        if self._use_lm_studio:
            return LMStudioGemma3Generator[t](self._logger)  # type: ignore
        return OllamaLlama3Generator[t](self._logger)  # type: ignore

    @override
    async def get_embedder(self) -> Embedder:
        return OllamaMiniLMEmbedder(self._logger)

    @override
    async def get_moderation_service(self) -> ModerationService:
        return OllamaModerationService(self._logger)