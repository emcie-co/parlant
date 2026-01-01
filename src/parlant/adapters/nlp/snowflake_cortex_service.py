# Copyright 2026 Emcie Co Ltd.
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

# Maintainer: Tao Tang <ttan@habitus.dk>

from __future__ import annotations

import os
import time
import json
from typing import Any, Mapping, Optional, Type, cast

import httpx
import tiktoken
from typing_extensions import override
from pydantic import ValidationError

from parlant.adapters.nlp.common import normalize_json_output, record_llm_metrics
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.service import EmbedderHints, NLPService, SchematicGeneratorHints
from parlant.core.nlp.embedding import BaseEmbedder, Embedder, EmbeddingResult
from parlant.core.nlp.generation import (
    T,
    BaseSchematicGenerator,
    SchematicGenerator,
    SchematicGenerationResult,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.nlp.moderation import ModerationService, NoModeration
from parlant.core.tracer import Tracer

HTTPX_TIMEOUT = httpx.Timeout(timeout=60.0, connect=5.0, read=60.0, write=60.0)


class CortexEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or "cl100k_base"
        try:
            self.encoding = tiktoken.encoding_for_model(self.model_name)
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        return int(len(self.encoding.encode(prompt)) * 1.05)


class CortexSchematicGenerator(BaseSchematicGenerator[T]):
    """
    Snowflake Cortex chat generator via REST:
        POST {BASE}/api/v2/cortex/inference:complete
    """

    _provider_params = ["temperature", "top_p", "top_k", "max_tokens", "stop"]
    supported_hints = _provider_params + ["strict"]

    def __init__(self, *, schema: type[T], logger: Logger, tracer: Tracer, meter: Meter) -> None:
        self.schema = schema
        self._logger = logger
        self._tracer = tracer
        self._meter = meter
        self._base_url = os.environ["SNOWFLAKE_CORTEX_BASE_URL"].rstrip("/")
        self._token = os.environ["SNOWFLAKE_AUTH_TOKEN"]
        self.model_name = os.environ["SNOWFLAKE_CORTEX_CHAT_MODEL"]

        self._tokenizer = CortexEstimatingTokenizer(self.model_name)
        self._client = httpx.AsyncClient(timeout=HTTPX_TIMEOUT)
        self._max_tokens_hint = int(os.environ.get("SNOWFLAKE_CORTEX_MAX_TOKENS", "8192"))

    @property
    @override
    def id(self) -> str:
        return f"snowflake-cortex/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> EstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        return self._max_tokens_hint

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    @policy(
        [
            retry(
                exceptions=(
                    httpx.ReadTimeout,
                    httpx.ConnectTimeout,
                    httpx.RemoteProtocolError,
                ),
                max_exceptions=3,
                wait_times=(1.0, 2.0, 4.0),
            ),
            retry(httpx.HTTPStatusError, max_exceptions=2, wait_times=(1.0, 5.0)),
        ]
    )
    @override
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        with self._logger.scope(f"Cortex LLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        schema: Type[T] = self.schema

        messages = [{"role": "user", "content": prompt}]
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
        }

        for k in self._provider_params:
            if k in hints:
                payload[k] = hints[k]

        # Strict path: provider-enforced JSON schema
        if hints.get("strict", False):
            try:
                payload["response_format"] = {
                    "type": "json",
                    "schema": schema.model_json_schema(),
                }
            except Exception as e:
                # If schema export fails, fall back to local validation
                self._logger.debug(f"Strict schema export failed, falling back: {e}")

        url = f"{self._base_url}/api/v2/cortex/inference:complete"

        t0 = time.time()
        resp = await self._client.post(url, headers=self._headers(), json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            self._logger.error(f"Cortex COMPLETE error {e.response.status_code}: {e.response.text}")
            raise
        t1 = time.time()

        data = resp.json()
        msg = (data.get("choices") or [{}])[0].get("message", {})
        raw = msg.get("content")
        if raw is None:
            cl = msg.get("content_list") or []
            if cl and isinstance(cl[0], dict):
                raw = cl[0].get("text")
        if raw is None:
            raw = msg if msg else data

        # Parse JSON
        try:
            if isinstance(raw, str):
                normalized = normalize_json_output(raw)
                parsed = cast(dict[str, Any], json.loads(normalized))
            elif isinstance(raw, dict):
                parsed = raw
            else:
                parsed = json.loads(str(raw))
        except Exception:
            try:
                normalized = normalize_json_output(str(raw))
                parsed = cast(dict[str, Any], json.loads(normalized))
            except Exception as ex:
                self._logger.error(f"Failed to parse structured output: {ex}\nRaw: {raw}")
                raise

        # Validate against the schema model
        try:
            content = schema.model_validate(parsed)
        except ValidationError as ve:
            self._logger.error(
                f"Structured output validation failed:\n{ve.json(indent=2)}\nRaw: {raw}"
            )
            raise

        usage_block = data.get("usage") or {}

        await record_llm_metrics(
            self._meter,
            self.model_name,
            schema_name=schema.__name__,
            input_tokens=usage_block.get("prompt_tokens", 0),
            output_tokens=usage_block.get("completion_tokens", 0),
        )

        return SchematicGenerationResult(
            content=content,
            info=GenerationInfo(
                schema_name=schema.__name__,
                model=self.id,
                duration=(t1 - t0),
                usage=UsageInfo(
                    input_tokens=usage_block.get("prompt_tokens", 0),
                    output_tokens=usage_block.get("completion_tokens", 0),
                    extra={},
                ),
            ),
        )


class CortexEmbedder(BaseEmbedder):
    """Embeddings via Snowflake Cortex.

    Endpoint:
        POST {BASE}/api/v2/cortex/inference:embed
    """

    supported_arguments = ["dimensions"]

    def __init__(self, *, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        model_name = os.environ["SNOWFLAKE_CORTEX_EMBED_MODEL"]
        super().__init__(logger=logger, tracer=tracer, meter=meter, model_name=model_name)

        self._base_url = os.environ["SNOWFLAKE_CORTEX_BASE_URL"].rstrip("/")
        self._token = os.environ["SNOWFLAKE_AUTH_TOKEN"]

        self._client = httpx.AsyncClient(timeout=HTTPX_TIMEOUT)
        self._tokenizer = CortexEstimatingTokenizer(self.model_name)
        self._dims = self._infer_dims(self.model_name)

    @property
    @override
    def id(self) -> str:
        return f"snowflake-cortex/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> EstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def dimensions(self) -> int:
        return self._dims

    @staticmethod
    def _infer_dims(model_name: str) -> int:
        n = model_name.lower()
        if "e5-base" in n:
            return 768
        if "snowflake-arctic-embed-m" in n:
            return 768
        if "snowflake-arctic-embed-l" in n:
            return 1024
        return 768

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    @policy(
        [
            retry(
                exceptions=(
                    httpx.ReadTimeout,
                    httpx.ConnectTimeout,
                    httpx.RemoteProtocolError,
                ),
                max_exceptions=3,
                wait_times=(1.0, 2.0, 4.0),
            ),
            retry(httpx.HTTPStatusError, max_exceptions=2, wait_times=(1.0, 5.0)),
        ]
    )
    @override
    async def do_embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        payload: dict[str, Any] = {"model": self.model_name, "text": texts}
        if "dimensions" in hints:
            payload["dimensions"] = hints["dimensions"]

        url = f"{self._base_url}/api/v2/cortex/inference:embed"
        resp = await self._client.post(url, headers=self._headers(), json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            self.logger.error(f"Cortex EMBED error {e.response.status_code}: {e.response.text}")
            raise

        data = resp.json()

        vectors: list[list[float]] = []
        for row in data.get("data", []):
            emb = row.get("embedding")
            if isinstance(emb, list) and emb and isinstance(emb[0], list):
                emb = emb[0]
            vectors.append(emb)

        return EmbeddingResult(vectors=vectors)

    @property
    @override
    def max_tokens(self) -> int:
        return 8192


class SnowflakeCortexService(NLPService):
    """Parlant adapter for Snowflake Cortex (chat + embeddings).

    Environment Variables:
        SNOWFLAKE_CORTEX_BASE_URL: Base account URL (e.g. https://<account>.snowflakecomputing.com)
        SNOWFLAKE_AUTH_TOKEN: OAuth/Keypair JWT/PAT token
        SNOWFLAKE_CORTEX_CHAT_MODEL: Chat model name
        SNOWFLAKE_CORTEX_EMBED_MODEL: Embedding model name
        SNOWFLAKE_CORTEX_MAX_TOKENS: Optional max token hint
    """

    @staticmethod
    def verify_environment() -> str | None:
        missing = []
        if not os.environ.get("SNOWFLAKE_CORTEX_BASE_URL"):
            missing.append(
                "SNOWFLAKE_CORTEX_BASE_URL (e.g. https://<account>.snowflakecomputing.com)"
            )
        if not os.environ.get("SNOWFLAKE_AUTH_TOKEN"):
            missing.append("SNOWFLAKE_AUTH_TOKEN (OAuth/Keypair JWT/PAT)")
        if not os.environ.get("SNOWFLAKE_CORTEX_CHAT_MODEL"):
            missing.append("SNOWFLAKE_CORTEX_CHAT_MODEL")
        if not os.environ.get("SNOWFLAKE_CORTEX_EMBED_MODEL"):
            missing.append("SNOWFLAKE_CORTEX_EMBED_MODEL")
        if missing:
            return "Missing Snowflake Cortex settings:\n  - " + "\n  - ".join(missing)
        return None

    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        self._logger = logger
        self._tracer = tracer
        self._meter = meter

        self._base_url = os.environ["SNOWFLAKE_CORTEX_BASE_URL"].rstrip("/")
        self._token = os.environ["SNOWFLAKE_AUTH_TOKEN"]
        self._chat_model = os.environ["SNOWFLAKE_CORTEX_CHAT_MODEL"]
        self._embed_model = os.environ["SNOWFLAKE_CORTEX_EMBED_MODEL"]

        self._logger.info(
            f"SnowflakeCortexService: chat={self._chat_model} | embed={self._embed_model} @ {self._base_url}"
        )

    @override
    async def get_schematic_generator(
        self, t: type[T], hints: SchematicGeneratorHints = {}
    ) -> SchematicGenerator[T]:
        return CortexSchematicGenerator[t](  # type: ignore[valid-type,misc]
            schema=t, logger=self._logger, tracer=self._tracer, meter=self._meter
        )

    @override
    async def get_embedder(self, hints: EmbedderHints = {}) -> Embedder:
        return CortexEmbedder(logger=self._logger, tracer=self._tracer, meter=self._meter)

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
