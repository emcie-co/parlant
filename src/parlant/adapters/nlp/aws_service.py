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

import time
from anthropic import (
    AsyncAnthropicBedrock,
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)  # type: ignore
from pydantic import ValidationError
from typing import Any, Mapping
from typing_extensions import override
import jsonfinder  # type: ignore
import os
import tiktoken

from parlant.adapters.nlp.common import normalize_json_output
from parlant.adapters.nlp.hugging_face import JinaAIEmbedder
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.nlp.embedding import Embedder
from parlant.core.nlp.generation import (
    T,
    SchematicGenerator,
    SchematicGenerationResult,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.loggers import Logger
from parlant.core.nlp.moderation import ModerationService, NoModeration
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.service import NLPService
from parlant.core.nlp.tokenization import EstimatingTokenizer


class AnthropicBedrockEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self) -> None:
        self.encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return int(len(tokens) * 1.15)


class AnthropicBedrockAISchematicGenerator(SchematicGenerator[T]):
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
            aws_session_token=os.environ.get("AWS_SESSION_TOKEN", None),
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

    @policy(
        [
            retry(
                exceptions=(
                    APIConnectionError,
                    APITimeoutError,
                    RateLimitError,
                    APIResponseValidationError,
                )
            ),
            retry(InternalServerError, max_exceptions=2, wait_times=(1.0, 5.0)),
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

        anthropic_api_arguments = {k: v for k, v in hints.items() if k in self.supported_hints}

        t_start = time.time()
        try:
            response = await self._client.messages.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                max_tokens=4096,
                **anthropic_api_arguments,
            )
        except RateLimitError:
            self._logger.error(
                "AWS Bedrock API rate limit exceeded. Possible reasons:\n"
                "1. Your account may have insufficient API credits.\n"
                "2. You may be using a free-tier account with limited request capacity.\n"
                "3. You might have exceeded the requests-per-minute limit for your account.\n\n"
                "Recommended actions:\n"
                "- Check your AWS Bedrock account balance and billing status.\n"
                "- Review your API usage limits in AWS Bedrock's dashboard.\n"
                "- For more details on rate limits and usage tiers, visit:\n"
                "  https://us-east-1.console.aws.amazon.com/servicequotas/home/services/bedrock/quotas",
            )
            raise

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
    @staticmethod
    def verify_environment() -> str | None:
        """Returns an error message if the environment is not set up correctly."""

        if not os.environ.get("ANTHROPIC_API_KEY"):
            return """\
You're using the AWS Bedrock NLP service, but some environment variables are missing.
Please consider seting the following your environment before running Parlant.

- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_REGION
- AWS_SESSION_TOKEN
"""
        return None

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
