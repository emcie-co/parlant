import time
from anthropic import AsyncAnthropicBedrock
from pydantic import ValidationError
from typing import Any, Mapping, override
import jsonfinder  # type: ignore
import os
import tiktoken

from parlant.adapters.nlp.common import normalize_json_output
from parlant.adapters.nlp.hugging_face import JinaAIEmbedder
from parlant.core.nlp.embedding import Embedder
from parlant.core.nlp.generation import (
    T,
    BaseSchematicGenerator,
    GenerationInfo,
    SchematicGenerationResult,
    UsageInfo,
)
from parlant.core.logging import Logger
from parlant.core.nlp.moderation import ModerationService, NoModeration
from parlant.core.nlp.service import NLPService
from parlant.core.nlp.tokenization import EstimatingTokenizer


class AnthropicBedrockEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self) -> None:
        self.encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return int(len(tokens) * 1.15)


class AnthropicBedrockAISchematicGenerator(BaseSchematicGenerator[T]):
    supported_hints = ["temperature"]

    def __init__(
        self,
        model_name: str,
        logger: Logger,
    ) -> None:
        self.model_name = model_name
        self._logger = logger

        self._client = AsyncAnthropicBedrock(
            aws_access_key=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            aws_region=os.environ["AWS_REGION"],
        )

        self._estimating_tokenizer = AnthropicBedrockEstimatingTokenizer()

    @property
    @override
    def id(self) -> str:
        return f"bedrock/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> AnthropicBedrockEstimatingTokenizer:
        return self._estimating_tokenizer

    @override
    async def generate(
        self,
        prompt: str,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        anthropic_api_arguments = {k: v for k, v in hints.items() if k in self.supported_hints}

        t_start = time.time()
        response = await self._client.messages.create(
            messages=[{"role": "user", "content": prompt}],
            model=self.model_name,
            max_tokens=4096,
            **anthropic_api_arguments,
        )
        t_end = time.time()

        raw_content = response.content[0].text

        try:
            json_content = normalize_json_output(raw_content)
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
                        input_tokens=response.usage.input_tokens,
                        output_tokens=response.usage.output_tokens,
                    ),
                ),
            )
        except ValidationError:
            self._logger.error(
                f"JSON content returned by {self.model_name} does not match expected schema:\n{raw_content}"
            )
            raise


class Claude_Sonnet_3_5(AnthropicBedrockAISchematicGenerator[T]):
    def __init__(self, logger: Logger) -> None:
        super().__init__(
            model_name="anthropic.claude-3-5-sonnet-20240620-v1:0",
            logger=logger,
        )

    @override
    @property
    def max_tokens(self) -> int:
        return 200 * 1024


class BedrockService(NLPService):
    def __init__(self, logger: Logger) -> None:
        self._logger = logger

    @override
    async def get_schematic_generator(self, t: type[T]) -> AnthropicBedrockAISchematicGenerator[T]:
        return Claude_Sonnet_3_5[t](self._logger)  # type: ignore

    @override
    async def get_embedder(self) -> Embedder:
        return JinaAIEmbedder()

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()