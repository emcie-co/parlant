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

import os
import time
import json
from typing import Any, Mapping, Optional, Type, cast

import httpx
import tiktoken
from typing_extensions import override
from pydantic import BaseModel, ValidationError

from parlant.adapters.nlp.common import normalize_json_output
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.loggers import Logger
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.nlp.service import NLPService
from parlant.core.nlp.embedding import Embedder, EmbeddingResult
from parlant.core.nlp.generation import T, SchematicGenerator, SchematicGenerationResult
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.nlp.moderation import ModerationService, NoModeration


class CortexEstimatingTokenizer(EstimatingTokenizer):
    """
    Token estimator. Cortex doesn't expose a tokenizer; use tiktoken heuristics.
    Default to cl100k_base if the specific model encoding is unknown.
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or "cl100k_base"
        try:
            self.encoding = tiktoken.encoding_for_model(self.model_name)
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        return int(len(self.encoding.encode(prompt)) * 1.05)


class CortexSchematicGenerator(SchematicGenerator[T]):
    """
    Snowflake Cortex chat generator via REST:
        POST {BASE}/api/v2/cortex/inference:complete

    Request (non-streaming, structured output):
        {
            "model": "<chat-model>",
            "messages": [{"role": "system", "content": "<prompt>"}],
            "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "<SchemaName>", "schema": <pydantic JSON Schema>}
            },
            "stream": false,
            ...hints
        }

    Response:
        {
            "choices": [
            {
                "message": {
                "content": "<JSON string or object>",
                "content_list": [{"type": "text", "text": "<same>"}]
                }
            }
            ],
            "usage": {"prompt_tokens": ..., "completion_tokens": ...}
        }
    """

    # Pass-through knobs supported by Cortex
    supported_hints = ["temperature", "top_p", "top_k", "max_tokens", "stop"]

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        token_type: Optional[str],
        model: str,
        logger: Logger,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._token_type = token_type
        self._model = model
        self._logger = logger
        self._tokenizer = CortexEstimatingTokenizer(self._model)
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=60.0))

        # Output upper bound
        self._max_tokens_hint = int(os.environ.get("SNOWFLAKE_CORTEX_MAX_TOKENS", "8192"))

    @property
    @override
    def id(self) -> str:
        return f"snowflake-cortex/{self._model}"

    @property
    @override
    def tokenizer(self) -> EstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        return self._max_tokens_hint

    def _headers(self) -> dict[str, str]:
        h = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self._token_type:
            h["X-Snowflake-Authorization-Token-Type"] = self._token_type
        return h

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
    async def generate(  # type: ignore[override]
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ):
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        messages = [{"role": "system", "content": prompt}]

        # Build a JSON Schema from the target Pydantic model
        schema_model: Type[BaseModel] = self.schema  # type: ignore[assignment]
        json_schema = cast(dict[str, Any], schema_model.model_json_schema())
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema_model.__name__, "schema": json_schema},
        }

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "response_format": response_format,
            "stream": False,  # ensure non-streaming
        }

        for k in self.supported_hints:
            if k in hints:
                payload[k] = hints[k]

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

        msg = (data.get("choices") or [{}])[0].get("message", {})  # type: ignore[assignment]
        raw = msg.get("content")
        if raw is None:
            # Fallback to content_list[0].text if present
            cl = msg.get("content_list") or []
            if cl and isinstance(cl[0], dict):
                raw = cl[0].get("text")

        # If still None, last-ditch: whole message object
        if raw is None:
            raw = msg if msg else data

        # Normalize & parse to a dict
        try:
            if isinstance(raw, str):
                normalized = normalize_json_output(raw)
                parsed = cast(dict[str, Any], json.loads(normalized))
            elif isinstance(raw, dict):
                parsed = raw
            else:
                # Try to coerce anything else via json
                parsed = json.loads(str(raw))
        except Exception:
            # If the provider returned free text, try extracting embedded JSON
            try:
                normalized = normalize_json_output(str(raw))
                parsed = cast(dict[str, Any], json.loads(normalized))
            except Exception as ex:
                self._logger.error(f"Failed to parse structured output: {ex}\nRaw: {raw}")
                raise

        try:
            content = schema_model.model_validate(parsed)  # type: ignore[attr-defined]
        except ValidationError as ve:
            # Log full validation diff + raw for debugging
            self._logger.error(
                f"Structured output validation failed:\n{ve.json(indent=2)}\nRaw: {raw}"
            )
            raise

        usage_block = data.get("usage") or {}
        return SchematicGenerationResult(
            content=content,
            info=GenerationInfo(
                schema_name=schema_model.__name__,
                model=self.id,
                duration=(t1 - t0),
                usage=UsageInfo(
                    input_tokens=usage_block.get("prompt_tokens", 0),
                    output_tokens=usage_block.get("completion_tokens", 0),
                    extra={},
                ),
            ),
        )


class CortexEmbedder(Embedder):
    """
    Snowflake Cortex embeddings via REST:
        POST {BASE}/api/v2/cortex/inference:embed

    Request:
        { "model": "<embed-model>", "text": ["...","..."], "dimensions": <optional> }

    Response (example):
        {
            "object": "list",
            "data": [{"object":"embedding","embedding":[[ ... floats ... ]],"index":0}],
            "model": "...",
            "usage": {"total_tokens": ...}
        }
    """

    supported_arguments = ["dimensions"]

    def __init__(
        self,
        *,
        base_url: str,
        token: str,
        token_type: Optional[str],
        model: str,
        logger: Logger,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._token_type = token_type
        self._model = model
        self._logger = logger
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(connect=5.0, read=60.0))
        self._tokenizer = CortexEstimatingTokenizer(self._model)
        self._dims = self._infer_dims(self._model)

    @property
    @override
    def id(self) -> str:
        return f"snowflake-cortex/{self._model}"

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
        h = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if self._token_type:
            h["X-Snowflake-Authorization-Token-Type"] = self._token_type
        return h

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
    async def embed(  # type: ignore[override]
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ):
        payload: dict[str, Any] = {"model": self._model, "text": texts}
        if "dimensions" in hints:
            payload["dimensions"] = hints["dimensions"]

        url = f"{self._base_url}/api/v2/cortex/inference:embed"
        resp = await self._client.post(url, headers=self._headers(), json=payload)
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            self._logger.error(f"Cortex EMBED error {e.response.status_code}: {e.response.text}")
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
        return 8192  # heuristic upper bound


class SnowflakeCortexService(NLPService):
    """
    Parlant adapter for Snowflake Cortex (chat + embeddings) via REST.

    Required env:
        - SNOWFLAKE_CORTEX_BASE_URL=https://<account>.snowflakecomputing.com
        - SNOWFLAKE_AUTH_TOKEN=<OAuth access token or Keypair JWT or PAT>
        - SNOWFLAKE_CORTEX_CHAT_MODEL=<chat model, e.g. 'mistral-large'>
        - SNOWFLAKE_CORTEX_EMBED_MODEL=<embed model, e.g. 'e5-base-v2'>

    Optional:
        - SNOWFLAKE_AUTH_TOKEN_TYPE=OAUTH | KEYPAIR_JWT | PAT
        - SNOWFLAKE_CORTEX_MAX_TOKENS=<int>
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

    def __init__(self, logger: Logger) -> None:
        self._logger = logger
        self._base_url = os.environ["SNOWFLAKE_CORTEX_BASE_URL"].rstrip("/")
        self._token = os.environ["SNOWFLAKE_AUTH_TOKEN"]
        self._token_type = os.environ.get("SNOWFLAKE_AUTH_TOKEN_TYPE")  # optional
        self._chat_model = os.environ["SNOWFLAKE_CORTEX_CHAT_MODEL"]
        self._embed_model = os.environ["SNOWFLAKE_CORTEX_EMBED_MODEL"]

        self._logger.info(
            f"SnowflakeCortexService: chat={self._chat_model} | embed={self._embed_model} @ {self._base_url}"
        )

    @override
    async def get_schematic_generator(self, t: type[T]) -> SchematicGenerator[T]:
        return CortexSchematicGenerator[T](
            base_url=self._base_url,
            token=self._token,
            token_type=self._token_type,
            model=self._chat_model,
            logger=self._logger,
        )

    @override
    async def get_embedder(self) -> Embedder:
        return CortexEmbedder(
            base_url=self._base_url,
            token=self._token,
            token_type=self._token_type,
            model=self._embed_model,
            logger=self._logger,
        )

    @override
    async def get_moderation_service(self) -> ModerationService:
        # No dedicated moderation route; keep a no-op for now.
        return NoModeration()
