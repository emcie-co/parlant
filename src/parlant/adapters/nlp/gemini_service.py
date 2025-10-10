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

import enum
import inspect
import enum
import inspect
import os
import time
import types
import types
from google.api_core.exceptions import NotFound, TooManyRequests, ResourceExhausted, ServerError
import google.genai  # type: ignore
import google.genai.types  # type: ignore
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from typing import Any, List, Literal, Mapping, Optional, Sequence, Union, cast
from typing_extensions import get_args, get_origin, override
from pydantic import BaseModel, Field, ValidationError
from pydantic.fields import FieldInfo

from parlant.core.common import DefaultBaseModel
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.moderation import ModerationService, NoModeration
from parlant.core.nlp.service import NLPService
from parlant.core.nlp.embedding import Embedder, EmbeddingResult
from parlant.core.nlp.generation import (
    T,
    SchematicGenerator,
    FallbackSchematicGenerator,
    SchematicGenerationResult,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.loggers import Logger

RATE_LIMIT_ERROR_MESSAGE = (
    "Google API rate limit exceeded.\n\n"
    "Possible reasons:\n"
    "1. Insufficient API credits in your account.\n"
    "2. Using a free-tier account with limited request capacity.\n"
    "3. Exceeded the requests-per-minute limit for your account.\n\n"
    "Recommended actions:\n"
    "- Check your Google API account balance and billing status.\n"
    "- Review your API usage limits in the Google Cloud Console.\n"
    "- Learn more about quotas and limits:\n"
    "  https://cloud.google.com/docs/quota-and-billing/quotas/quotas-overview"
)


class GoogleEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, client: google.genai.Client, model_name: str) -> None:
        self._client = client
        self._model_name = model_name

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        model_approximation = {
            "text-embedding-004": "gemini-2.5-flash",
        }.get(self._model_name, self._model_name)

        result = await self._client.aio.models.count_tokens(
            model=model_approximation,
            contents=prompt,
        )

        return int(result.total_tokens or 0)


class GeminiSchematicGenerator(SchematicGenerator[T]):
    supported_hints = ["temperature", "thinking_config"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
    ) -> None:
        self.model_name = model_name
        self._logger = logger

        self._client = google.genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

        self._tokenizer = GoogleEstimatingTokenizer(client=self._client, model_name=self.model_name)

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
            retry(ServerError, max_exceptions=2, wait_times=(1.0, 5.0)),
        ]
    )
    @override
    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        gemini_api_arguments = {k: v for k, v in hints.items() if k in self.supported_hints}

        fd = self._get_schema_function_declaration()

        config = google.genai.types.GenerateContentConfig(
            tools=[google.genai.types.Tool(function_declarations=[fd])],
            tool_config=google.genai.types.ToolConfig(
                function_calling_config=google.genai.types.FunctionCallingConfig(
                    mode=google.genai.types.FunctionCallingConfigMode.ANY,
                    allowed_function_names=[fd.name],
                )
            ),
            **gemini_api_arguments,  # type: ignore
        )

        fd = self._get_schema_function_declaration()

        config = google.genai.types.GenerateContentConfig(
            tools=[google.genai.types.Tool(function_declarations=[fd])],
            tool_config=google.genai.types.ToolConfig(
                function_calling_config=google.genai.types.FunctionCallingConfig(
                    mode=google.genai.types.FunctionCallingConfigMode.ANY,
                    allowed_function_names=[fd.name],
                )
            ),
            **gemini_api_arguments,  # type: ignore
        )

        t_start = time.time()
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
                config=config,
            )
        except TooManyRequests:
            self._logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise

        t_end = time.time()

        assert response.candidates
        assert response.candidates[0].content
        assert response.candidates[0].content.parts
        assert response.candidates[0].content.parts[0].function_call
        assert response.candidates[0].content.parts[0].function_call.args

        json_result = (
            response.candidates[0].content.parts[0].function_call.args.get("log_data", {}) or {}
        )
        assert response.candidates
        assert response.candidates[0].content
        assert response.candidates[0].content.parts
        assert response.candidates[0].content.parts[0].function_call
        assert response.candidates[0].content.parts[0].function_call.args

        json_result = (
            response.candidates[0].content.parts[0].function_call.args.get("log_data", {}) or {}
        )

        if response.usage_metadata:
            self._logger.trace(response.usage_metadata.model_dump_json(indent=2))

        try:
            model_content = self.schema.model_validate(json_result)
            model_content = self.schema.model_validate(json_result)

            return SchematicGenerationResult(
                content=model_content,
                info=GenerationInfo(
                    schema_name=self.schema.__name__,
                    model=self.id,
                    duration=(t_end - t_start),
                    usage=UsageInfo(
                        input_tokens=response.usage_metadata.prompt_token_count or 0,
                        output_tokens=response.usage_metadata.candidates_token_count or 0,
                        extra={
                            "cached_input_tokens": (
                                response.usage_metadata.cached_content_token_count
                                if response.usage_metadata
                                else 0
                            )
                            or 0
                        },
                    )
                    if response.usage_metadata
                    else UsageInfo(input_tokens=0, output_tokens=0, extra={}),
                ),
            )
        except ValidationError:
            self._logger.error(
                f"JSON content returned by {self.model_name} does not match expected schema:\n{json_result}"
            )
            raise

    def _get_schema_function_declaration(self) -> google.genai.types.FunctionDeclaration:
        # Create a signature from parameters
        sig = inspect.Signature(
            parameters=[
                inspect.Parameter(
                    name="log_data",
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=convert_model_to_gemini_compatible_schema(self.schema),
                )
            ],
            return_annotation=bool,
        )

        # Create a fake callable
        def log_data() -> None:
            pass

        # Attach the signature
        log_data.__signature__ = sig  # type: ignore

        fd = google.genai.types.FunctionDeclaration.from_callable(
            callable=log_data,
            client=self._client,  # type: ignore
        )

        return fd

    def _get_schema_function_declaration(self) -> google.genai.types.FunctionDeclaration:
        # Create a signature from parameters
        sig = inspect.Signature(
            parameters=[
                inspect.Parameter(
                    name="log_data",
                    kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=convert_model_to_gemini_compatible_schema(self.schema),
                )
            ],
            return_annotation=bool,
        )

        # Create a fake callable
        def log_data() -> None:
            pass

        # Attach the signature
        log_data.__signature__ = sig  # type: ignore

        fd = google.genai.types.FunctionDeclaration.from_callable(
            callable=log_data,
            client=self._client,  # type: ignore
        )

        return fd


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


class Gemini_2_0_Flash_Lite(GeminiSchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(
            model_name="gemini-2.0-flash-lite-preview-02-05",
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


class Gemini_2_5_Flash(GeminiSchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(
            model_name="gemini-2.5-flash",
            logger=logger,
        )

    @override
    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        return await super().generate(
            prompt,
            {"thinking_config": {"thinking_budget": 0}, **hints},
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 1024 * 1024


class Gemini_2_5_Pro(GeminiSchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(
            model_name="gemini-2.5-pro",
            logger=logger,
        )

    @property
    @override
    def max_tokens(self) -> int:
        return 1024 * 1024


class GoogleEmbedder(Embedder):
    supported_hints = ["title", "task_type"]

    def __init__(self, model_name: str, logger: Logger) -> None:
        self.model_name = model_name

        self._logger = logger
        self._client = google.genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self._tokenizer = GoogleEstimatingTokenizer(
            client=self._client, model_name=self.model_name
        )

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
            retry(ServerError, max_exceptions=2, wait_times=(1.0, 5.0)),
        ]
    )
    @override
    async def embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        gemini_api_arguments = {k: v for k, v in hints.items() if k in self.supported_hints}

        try:
            with self._logger.operation("Embedding text with gemini"):
                response = await self._client.aio.models.embed_content(  # type: ignore
                    model=self.model_name,
                    contents=texts,  # type: ignore
                    config=cast(google.genai.types.EmbedContentConfigDict, gemini_api_arguments),
                )
        except TooManyRequests:
            self._logger.error(
                (
                    "Google API rate limit exceeded. Possible reasons:\n"
                    "1. Your account may have insufficient API credits.\n"
                    "2. You may be using a free-tier account with limited request capacity.\n"
                    "3. You might have exceeded the requests-per-minute limit for your account.\n\n"
                    "Recommended actions:\n"
                    "- Check your Google API account balance and billing status.\n"
                    "- Review your API usage limits in Google's dashboard.\n"
                    "- For more details on rate limits and usage tiers, visit:\n"
                    "  https://cloud.google.com/docs/quota-and-billing/quotas/quotas-overview"
                ),
            )
            raise

        vectors = [
            data_point.values for data_point in response.embeddings or [] if data_point.values
        ]
        return EmbeddingResult(vectors=vectors)


class GeminiTextEmbedding_004(GoogleEmbedder):
    def __init__(self, logger: Logger) -> None:
        super().__init__(model_name="text-embedding-004", logger=logger)

    @property
    @override
    def max_tokens(self) -> int:
        return 8000

    @property
    def dimensions(self) -> int:
        return 768


class GeminiService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""

        if not os.environ.get("GEMINI_API_KEY"):
            return """\
You're using the GEMINI NLP service, but GEMINI_API_KEY is not set.
Please set GEMINI_API_KEY in your environment before running Parlant.
"""

        return None

    def __init__(self, logger: Logger, model_name: Optional[Union[List[str], str]] = None) -> None:
        self._model_name = model_name
        self._logger = logger
        self._logger.info("Initialized GeminiService")

    @override
    async def get_schematic_generator(self, t: type[T]) -> GeminiSchematicGenerator[T]:
        selected_model = self._model_name

        if isinstance(selected_model, list):
            model_classes: List[SchematicGenerator][T] = []
            for model_name in selected_model:
                model_cls = self._resolve_model_class(model_name)
                if model_cls:
                    model_classes.append(model_cls[t](self._logger))

            if not model_classes:
                self._logger.warning("No valid models found in list, falling back to defaults.")
                model_classes = [
                    Gemini_2_5_Flash[t](self._logger),
                    Gemini_2_5_Pro[t](self._logger),
                ]

            return FallbackSchematicGenerator[t](*model_classes, logger=self._logger)

        if isinstance(selected_model, str):
            model_cls = self._resolve_model_class(selected_model)
            if model_cls:
                return model_cls[t](self._logger)

            self._logger.warning(
                f"Unrecognized model name: '{selected_model}', falling back to defaults."
            )

        return FallbackSchematicGenerator[t](
            Gemini_2_5_Flash[t](self._logger),
            Gemini_2_5_Pro[t](self._logger),
            logger=self._logger,
        )

    async def _resolve_model_class(self, model_name: str) -> Optional[type[GeminiSchematicGenerator[T]]]:
        """Maps model names to their respective Gemini implementations."""
        mapping = {
            "gemini-2.5-pro": Gemini_2_5_Pro,
            "gemini-2.5-flash": Gemini_2_5_Flash,
            "gemini-2.0-flash": Gemini_2_0_Flash,
            "gemini-2.0-flash-lite": Gemini_2_0_Flash_Lite,
            "gemini-1.5-pro": Gemini_1_5_Pro,
            "gemini-1.5-flash": Gemini_1_5_Flash,
        }
        return mapping.get(model_name)


    @override
    async def get_embedder(self) -> Embedder:
        return GeminiTextEmbedding_004(self._logger)

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()


def convert_type_annotation_to_gemini_compatible_schema(annotation: Any) -> Any:
    origin = get_origin(annotation)

    # If not a generic type, check if it's a BaseModel or Enum
    if origin is None:
        # If it's an Enum class, convert to Literal of its values
        if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
            enum_values = tuple(member.value for member in annotation)
            if len(enum_values) == 1:
                return Literal[enum_values[0]]
            return Literal.__getitem__(enum_values)

        # If it's a BaseModel class, recursively convert it
        if inspect.isclass(annotation) and issubclass(annotation, DefaultBaseModel):
            return convert_model_to_gemini_compatible_schema(annotation)

        return annotation

    # Get the type arguments
    args = get_args(annotation)

    # Convert nested types recursively
    converted_args = tuple(convert_type_annotation_to_gemini_compatible_schema(arg) for arg in args)

    # Check if origin is Mapping or Sequence
    if origin is Mapping or origin is MappingABC:
        return dict[converted_args] if converted_args else dict  # type: ignore

    if origin is Sequence or origin is SequenceABC:
        return list[converted_args] if converted_args else list  # type: ignore

    # Handle UnionType (X | Y syntax) - not subscriptable!
    if origin is types.UnionType:
        return Union[converted_args]

    # For other generic types, preserve the origin with converted args
    if converted_args:
        return origin[converted_args]

    return annotation


def convert_model_to_gemini_compatible_schema(model_cls: type[DefaultBaseModel]) -> type[BaseModel]:
    """
    Create a new BaseModel class with converted annotations.
    Returns a new class without modifying the original.
    """
    # Avoid infinite recursion - check if already converted
    if hasattr(model_cls, "_conversion_cache"):
        return cast(type[BaseModel], model_cls._conversion_cache)

    # Build new annotations
    new_annotations = {}
    new_fields = {}

    for field_name, field_info in model_cls.model_fields.items():
        # Convert the annotation
        converted_annotation = convert_type_annotation_to_gemini_compatible_schema(
            field_info.annotation
        )
        new_annotations[field_name] = converted_annotation

        # Preserve field metadata (default, description, etc.)
        # We need to recreate the field with the new annotation
        field_kwargs = {}

        if field_info.default is not None and field_info.default is not FieldInfo:
            field_kwargs["default"] = field_info.default
        elif field_info.default_factory is not None:
            field_kwargs["default_factory"] = field_info.default_factory

        if field_info.description is not None:
            field_kwargs["description"] = field_info.description

        if field_info.title is not None:
            field_kwargs["title"] = field_info.title

        if field_info.examples is not None:
            field_kwargs["examples"] = field_info.examples

        # Add other field properties as needed
        if field_kwargs:
            new_fields[field_name] = Field(**field_kwargs)

    # Create new model class
    new_model_attrs = {"__annotations__": new_annotations, **new_fields}

    # Preserve model config if present
    if hasattr(model_cls, "model_config"):
        new_model_attrs["model_config"] = model_cls.model_config

    # Create the new class
    converted_model = type(f"{model_cls.__name__}Converted", (DefaultBaseModel,), new_model_attrs)

    # Cache the conversion to avoid infinite recursion
    setattr(model_cls, "_conversion_cache", converted_model)

    return converted_model


def convert_type_annotation_to_gemini_compatible_schema(annotation: Any) -> Any:
    origin = get_origin(annotation)

    # If not a generic type, check if it's a BaseModel or Enum
    if origin is None:
        # If it's an Enum class, convert to Literal of its values
        if inspect.isclass(annotation) and issubclass(annotation, enum.Enum):
            enum_values = tuple(member.value for member in annotation)
            if len(enum_values) == 1:
                return Literal[enum_values[0]]
            return Literal.__getitem__(enum_values)

        # If it's a BaseModel class, recursively convert it
        if inspect.isclass(annotation) and issubclass(annotation, DefaultBaseModel):
            return convert_model_to_gemini_compatible_schema(annotation)

        return annotation

    # Get the type arguments
    args = get_args(annotation)

    # Convert nested types recursively
    converted_args = tuple(convert_type_annotation_to_gemini_compatible_schema(arg) for arg in args)

    # Check if origin is Mapping or Sequence
    if origin is Mapping or origin is MappingABC:
        return dict[converted_args] if converted_args else dict  # type: ignore

    if origin is Sequence or origin is SequenceABC:
        return list[converted_args] if converted_args else list  # type: ignore

    # Handle UnionType (X | Y syntax) - not subscriptable!
    if origin is types.UnionType:
        return Union[converted_args]

    # For other generic types, preserve the origin with converted args
    if converted_args:
        return origin[converted_args]

    return annotation


def convert_model_to_gemini_compatible_schema(model_cls: type[DefaultBaseModel]) -> type[BaseModel]:
    """
    Create a new BaseModel class with converted annotations.
    Returns a new class without modifying the original.
    """
    # Avoid infinite recursion - check if already converted
    if hasattr(model_cls, "_conversion_cache"):
        return cast(type[BaseModel], model_cls._conversion_cache)

    # Build new annotations
    new_annotations = {}
    new_fields = {}

    for field_name, field_info in model_cls.model_fields.items():
        # Convert the annotation
        converted_annotation = convert_type_annotation_to_gemini_compatible_schema(
            field_info.annotation
        )
        new_annotations[field_name] = converted_annotation

        # Preserve field metadata (default, description, etc.)
        # We need to recreate the field with the new annotation
        field_kwargs = {}

        if field_info.default is not None and field_info.default is not FieldInfo:
            field_kwargs["default"] = field_info.default
        elif field_info.default_factory is not None:
            field_kwargs["default_factory"] = field_info.default_factory

        if field_info.description is not None:
            field_kwargs["description"] = field_info.description

        if field_info.title is not None:
            field_kwargs["title"] = field_info.title

        if field_info.examples is not None:
            field_kwargs["examples"] = field_info.examples

        # Add other field properties as needed
        if field_kwargs:
            new_fields[field_name] = Field(**field_kwargs)

    # Create new model class
    new_model_attrs = {"__annotations__": new_annotations, **new_fields}

    # Preserve model config if present
    if hasattr(model_cls, "model_config"):
        new_model_attrs["model_config"] = model_cls.model_config

    # Create the new class
    converted_model = type(f"{model_cls.__name__}Converted", (DefaultBaseModel,), new_model_attrs)

    # Cache the conversion to avoid infinite recursion
    setattr(model_cls, "_conversion_cache", converted_model)

    return converted_model
