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
import time
from typing import Any, Mapping # Added Mapping
from typing_extensions import override # Ensured override is imported
import json
import jsonfinder  # type: ignore
import os

from pydantic import ValidationError
import tiktoken

import litellm # Ensured litellm is imported

from parlant.adapters.nlp.common import normalize_json_output
# from parlant.adapters.nlp.hugging_face import JinaAIEmbedder # REMOVED JinaAIEmbedder import
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.loggers import Logger # Ensured Logger is imported
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.service import NLPService
from parlant.core.nlp.embedding import Embedder, EmbeddingResult # Ensured EmbeddingResult is imported
from parlant.core.nlp.generation import (
    T, # Ensured T is imported
    SchematicGenerator,
    SchematicGenerationResult,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.nlp.moderation import (
    ModerationService, # Ensured ModerationService is imported
    NoModeration, # Ensured NoModeration is imported
)

RATE_LIMIT_ERROR_MESSAGE = (
    "LiteLLM to provider API rate limit exceeded. Possible reasons:\n"
    "1. Your account may have insufficient API credits.\n"
    "2. You may be using a free-tier account with limited request capacity.\n"
    "3. You might have exceeded the requests-per-minute limit for your account.\n\n"
    "Recommended actions:\n"
    "- Check your LLM Provider account balance and billing status.\n"
    "- Review your API usage limits in Provider's dashboard.\n"
    "- For more details on rate limits and usage tiers, visit:\n"
    "  Your Provider's API documentation."
)


class LiteLLMEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return len(tokens)


class LiteLLMSchematicGenerator(SchematicGenerator[T]):
    supported_litellm_params = [
        "temperature",
        "max_tokens",
        "logit_bias",
        "adapter_id",
        "adapter_soruce", # Typo "soruce" exists in original, keeping it to match
    ]
    supported_hints = supported_litellm_params + ["strict"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
    ) -> None:
        self.model_name = model_name
        self._logger = logger

        self._client = litellm

        self._tokenizer = LiteLLMEstimatingTokenizer(model_name=self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"litellm/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> LiteLLMEstimatingTokenizer:
        return self._tokenizer

    @override
    async def generate(
        self,
        prompt: PromptBuilder | str,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        with self._logger.operation(f"LiteLLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        litellm_api_arguments = {
            k: v for k, v in hints.items() if k in self.supported_litellm_params
        }

        t_start = time.time()

        # Make sure LITELLM_PROVIDER_API_KEY is present, otherwise LiteLLM might error or use other fallbacks
        api_key = os.environ.get("LITELLM_PROVIDER_API_KEY")
        if not api_key:
            self._logger.error("LITELLM_PROVIDER_API_KEY environment variable not set for LiteLLM completion.")
            # Potentially raise an error or handle as appropriate
            # For now, proceeding will likely cause LiteLLM to fail if the selected model needs a key

        response = self._client.completion( # This is a synchronous call
            api_key=api_key, # Use fetched api_key
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            max_tokens=5000, # Consider making this configurable
            response_format={"type": "json_object"},
            # api_base=os.environ.get("LITELLM_API_BASE"), # Could be useful if provider needs a specific base
            **litellm_api_arguments,
        )

        t_end = time.time()

        if response.usage:
            self._logger.debug(response.usage.model_dump_json(indent=2))

        raw_content = response.choices[0].message.content or "{}"

        try:
            json_content = json.loads(normalize_json_output(raw_content))
        except json.JSONDecodeError:
            self._logger.warning(
                f"Invalid JSON returned by litellm/{self.model_name}:\n{raw_content})"
            )
            # Attempt to find JSON within the raw content if direct parsing fails
            json_matches = list(jsonfinder.jsonfinder(raw_content, multiline=True))
            if json_matches:
                json_content = json_matches[0][0] # Take the first found JSON object
                self._logger.warning("Found JSON content within model response using jsonfinder; continuing...")
            else:
                self._logger.error(f"No valid JSON found in response from litellm/{self.model_name}")
                raise # Re-raise if no JSON can be extracted

        try:
            content = self.schema.model_validate(json_content)
            assert response.usage is not None # Ensure usage is not None

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
                            "cached_input_tokens": getattr(
                                response.usage, # Accessing usage directly
                                "prompt_cache_hit_tokens", # This attribute might not always exist
                                0,
                            )
                        },
                    ),
                ),
            )
        except ValidationError:
            self._logger.error(
                f"JSON content returned by litellm/{self.model_name} does not match expected schema:\n{raw_content}"
            )
            raise


class LiteLLM_Default(LiteLLMSchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        # Ensure LITELLM_PROVIDER_MODEL_NAME is available, provide a fallback or raise if critical
        model_name = os.environ.get("LITELLM_PROVIDER_MODEL_NAME")
        if not model_name:
            logger.critical("LITELLM_PROVIDER_MODEL_NAME environment variable is not set.")
            # This will likely cause issues. Consider raising an exception or having a safe default.
            # For now, let's proceed, but this needs to be set for the service to work.
            model_name = "unknown-litellm-model" # Placeholder
        super().__init__(
            model_name=model_name,
            logger=logger,
        )

    @property
    @override
    def max_tokens(self) -> int:
        # This should ideally be model-specific if possible, or a safe general value
        return 4096 # Adjusted from 5000 to a more common value


# --- START OF OllamaEmbedder DEFINITION ---
class OllamaEmbedder(Embedder):
    def __init__(self, logger: Logger):
        self.logger = logger
        raw_model_name = os.environ.get("OLLAMA_EMBEDDING_MODEL_NAME", "nomic-embed-text") 
        if not raw_model_name.startswith("ollama/"):
            self.model_name = f"ollama/{raw_model_name}"
        else:
            self.model_name = raw_model_name
            
        self.api_base = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
        
        self.logger.info(f"Initialized OllamaEmbedder with model: {self.model_name} and api_base: {self.api_base}")
        # Tokenizer for Ollama is not straightforward with tiktoken used elsewhere for OpenAI.
        # For now, we assume the Embedder interface does not strictly require a functional tokenizer
        # for the `embed` method to work, especially when LiteLLM handles the call.
        # self._tokenizer = LiteLLMEstimatingTokenizer(model_name=self.model_name) # Avoid using OpenAI-specific tokenizer

    @property
    @override
    def id(self) -> str:
        return f"ollama/{self.model_name.replace('ollama/', '')}"

    # @property
    # @override
    # def tokenizer(self) -> EstimatingTokenizer:
    #     # If a generic tokenizer or one suitable for Ollama models via LiteLLM is available,
    #     # it would be returned here. For now, not implemented to avoid incompatibility.
    #     raise NotImplementedError("Tokenizer not implemented for OllamaEmbedder.")

    # @property
    # @override
    # def max_tokens(self) -> int:
    #     # Max tokens for embedding models can vary. 8192 is a common figure.
    #     return 8192 

    @override
    async def embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {}, 
    ) -> EmbeddingResult:
        valid_texts = [text for text in texts if text is not None and text.strip() != ""]
        if not valid_texts:
            self.logger.warning("Received no valid texts for embedding after filtering. Returning empty vectors.")
            return EmbeddingResult(vectors=[])

        with self.logger.operation(f"Ollama Embedding Request (Model: {self.model_name}, API Base: {self.api_base})"):
            try:
                # Use await for the asynchronous call
                response = await litellm.aembedding(
                    model=self.model_name,
                    input=valid_texts,
                    api_base=self.api_base,
                    # Ollama typically does not require an API key when run locally.
                    # If your Ollama instance is secured, you might need to pass additional parameters.
                    # Example: api_key=os.environ.get("OLLAMA_API_KEY") if needed.
                )
                
                # LiteLLM's aembedding returns a ModelResponse object.
                # The embeddings are in response.data, which is a list of dicts.
                vectors = [data_point['embedding'] for data_point in response.data]
                return EmbeddingResult(vectors=vectors)
            except Exception as e:
                self.logger.error(f"Ollama embedding call failed for model {self.model_name} using API base {self.api_base}: {e}")
                # Depending on how Parlant expects errors, either re-raise or return empty.
                # Returning empty vectors for robustness, as per original thought.
                return EmbeddingResult(vectors=[])
# --- END OF OllamaEmbedder DEFINITION ---


class LiteLLMService(NLPService):
    def __init__(
        self,
        logger: Logger,
    ) -> None:
        # Ensure LITELLM_PROVIDER_MODEL_NAME is available, provide a fallback or raise if critical
        self._model_name = os.environ.get("LITELLM_PROVIDER_MODEL_NAME")
        if not self._model_name:
            logger.critical("LITELLM_PROVIDER_MODEL_NAME environment variable is not set for LiteLLMService.")
            # This is critical for the schematic generator.
            # Defaulting to a placeholder which will likely fail if not configured.
            self._model_name = "placeholder-model-name-not-set"
        self._logger = logger
        self._logger.info(f"Initialized LiteLLMService with provider model: {self._model_name}")

    @override
    async def get_schematic_generator(self, t: type[T]) -> LiteLLMSchematicGenerator[T]:
        # LiteLLM_Default internally uses LITELLM_PROVIDER_MODEL_NAME from env var.
        return LiteLLM_Default[t](self._logger) # type: ignore

    @override
    async def get_embedder(self) -> Embedder:
        self._logger.info("Using OllamaEmbedder for embeddings.")
        return OllamaEmbedder(logger=self._logger) # Pass the service's logger instance

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
