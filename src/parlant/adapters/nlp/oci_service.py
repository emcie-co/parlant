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

from __future__ import annotations

import base64
import email.utils
import hashlib
import json
import os
import time
from copy import deepcopy
from typing import Any, Mapping
from urllib.parse import urlparse

import httpx
import jsonfinder  # type: ignore
import tiktoken
from pydantic import ValidationError
from typing_extensions import override

import oci  # type: ignore[import-untyped]
from oci.signer import Signer  # type: ignore[import-untyped]
from oci.generative_ai_inference.models import (  # type: ignore[import-untyped]  # type: ignore[import-untyped]
    ChatDetails,
    CohereChatRequest,
    CohereResponseJsonFormat,
    EmbedTextDetails,
    FunctionDefinition,
    GenericChatRequest,
    OnDemandServingMode,
    TextContent,
    ToolChoiceRequired,
    UserMessage,
)

from parlant.adapters.nlp.common import normalize_json_output, record_llm_metrics
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.nlp.embedding import BaseEmbedder, Embedder, EmbeddingResult
from parlant.core.nlp.generation import (
    T,
    BaseSchematicGenerator,
    SchematicGenerationResult,
)
from parlant.core.nlp.generation_info import GenerationInfo, UsageInfo
from parlant.core.nlp.moderation import ModerationService, NoModeration
from parlant.core.nlp.policies import policy, retry
from parlant.core.nlp.service import EmbedderHints, NLPService, SchematicGeneratorHints
from parlant.core.nlp.tokenization import EstimatingTokenizer
from parlant.core.tracer import Tracer


RATE_LIMIT_ERROR_MESSAGE = """\
OCI Generative AI rate limit exceeded. Possible reasons:
1. Your OCI account may have insufficient credits.
2. You might have exceeded the requests-per-minute limit.
3. The model may have service-side throttling.

Recommended actions:
- Check your OCI account limits and usage.
- Reduce request volume or add backoff.
- Verify the model is enabled in your tenancy.
"""


class OCIRetryableError(Exception):
    pass


def _calculate_content_sha256(body: bytes) -> str:
    """Calculate SHA256 hash of body, base64 encoded (required for OCI request signing)."""
    m = hashlib.sha256()
    m.update(body)
    return base64.b64encode(m.digest()).decode("utf-8")


def _build_service_endpoint(region: str) -> str:
    """Build the OCI Generative AI service endpoint for a region."""
    return f"https://inference.generativeai.{region}.oci.oraclecloud.com"


def _serialize_oci_model(obj: Any) -> Any:
    """
    Serialize OCI model to JSON-compatible dict using camelCase keys.

    OCI API expects camelCase field names, but Python models use snake_case.
    The SDK stores the mapping in `attribute_map` on each model class.
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, list):
        return [_serialize_oci_model(item) for item in obj if item is not None]

    if isinstance(obj, dict):
        return {k: _serialize_oci_model(v) for k, v in obj.items() if v is not None}

    # OCI model objects have swagger_types and attribute_map
    if hasattr(obj, "swagger_types") and hasattr(obj, "attribute_map"):
        result: dict[str, Any] = {}
        for attr in obj.swagger_types:
            value = getattr(obj, attr, None)
            if value is not None:
                # Use attribute_map to get the camelCase key name
                key = obj.attribute_map.get(attr, attr)
                result[key] = _serialize_oci_model(value)
        return result

    # Fallback for unknown types
    return obj


class OCIAsyncHttpClient:
    """
    Async HTTP client for OCI Generative AI API.

    Uses httpx.AsyncClient for native async I/O instead of blocking threads.
    Reuses OCI SDK's Signer for request authentication.
    """

    BASE_PATH = "/20231130"

    def __init__(self, config: dict[str, Any]) -> None:
        self._region = config.get("region", "us-chicago-1")
        self._endpoint = _build_service_endpoint(self._region)
        self._signer = Signer(
            tenancy=config["tenancy"],
            user=config["user"],
            fingerprint=config["fingerprint"],
            private_key_file_location=config.get("key_file"),
            pass_phrase=config.get("pass_phrase"),
        )
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or lazily create the async HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    def _sign_request(
        self,
        method: str,
        url: str,
        body: bytes | None = None,
    ) -> dict[str, str]:
        """
        Sign a request using OCI's Signer and return headers.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: Full URL
            body: Request body as bytes (for POST/PUT/PATCH)

        Returns:
            Dictionary of signed headers including Authorization
        """
        parsed = urlparse(url)
        host = parsed.netloc
        path = parsed.path
        if parsed.query:
            path = f"{path}?{parsed.query}"

        headers: dict[str, str] = {
            "date": email.utils.formatdate(usegmt=True),
            "host": host,
            "content-type": "application/json",
        }

        if body and method.upper() in ("POST", "PUT", "PATCH"):
            headers["content-length"] = str(len(body))
            headers["x-content-sha256"] = _calculate_content_sha256(body)

            signed = self._signer._body_signer.sign(
                headers,
                host=host,
                method=method.upper(),
                path=path,
            )
        else:
            signed = self._signer._basic_signer.sign(
                headers,
                host=host,
                method=method.upper(),
                path=path,
            )

        return dict(signed)

    async def request(
        self,
        method: str,
        path: str,
        body: Any | None = None,
    ) -> dict[str, Any]:
        """
        Make a signed async request to OCI.

        Args:
            method: HTTP method
            path: API path (e.g., "/actions/chat")
            body: Request body (will be serialized with oci.util.to_dict if OCI model)

        Returns:
            Response JSON as dictionary

        Raises:
            OCIRetryableError: For rate limit (429) or server errors (5xx)
            httpx.HTTPStatusError: For other HTTP errors
        """
        url = f"{self._endpoint}{self.BASE_PATH}{path}"

        if body is not None:
            if hasattr(body, "swagger_types"):
                body_dict = _serialize_oci_model(body)
            else:
                body_dict = body
            body_bytes = json.dumps(body_dict).encode("utf-8")
        else:
            body_bytes = None

        headers = self._sign_request(method, url, body_bytes)
        headers["accept"] = "application/json"

        client = await self._get_client()

        response = await client.request(
            method=method,
            url=url,
            headers=headers,
            content=body_bytes,
        )

        if response.status_code == 429:
            raise OCIRetryableError(f"Rate limit exceeded: {response.text}")
        if 500 <= response.status_code < 600:
            raise OCIRetryableError(f"Server error {response.status_code}: {response.text}")

        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    async def chat(self, chat_details: Any) -> dict[str, Any]:
        """Call the chat endpoint."""
        return await self.request("POST", "/actions/chat", chat_details)

    async def embed_text(self, embed_details: Any) -> dict[str, Any]:
        """Call the embed text endpoint."""
        return await self.request("POST", "/actions/embedText", embed_details)


def _expand_path(path: str) -> str:
    return os.path.expanduser(path)


def load_oci_config() -> dict[str, Any]:
    required_env_keys = [
        "OCI_USER",
        "OCI_TENANCY",
        "OCI_FINGERPRINT",
        "OCI_KEY_FILE",
        "OCI_REGION",
    ]

    if all(os.environ.get(k) for k in required_env_keys):
        config: dict[str, Any] = {
            "user": os.environ["OCI_USER"],
            "tenancy": os.environ["OCI_TENANCY"],
            "fingerprint": os.environ["OCI_FINGERPRINT"],
            "key_file": _expand_path(os.environ["OCI_KEY_FILE"]),
            "region": os.environ["OCI_REGION"],
        }
        if os.environ.get("OCI_PASSPHRASE"):
            config["pass_phrase"] = os.environ["OCI_PASSPHRASE"]
        return config

    config_file = _expand_path(os.environ.get("OCI_CONFIG_FILE", "~/.oci/config"))
    profile_name = os.environ.get("OCI_CONFIG_PROFILE", "DEFAULT")
    result: dict[str, Any] = oci.config.from_file(
        file_location=config_file, profile_name=profile_name
    )
    return result


def _fix_double_encoded_fields(data: Any, schema: dict[str, Any]) -> Any:
    """Fix for models (e.g. Meta Llama) that serialize array/object fields as JSON strings in function calls."""
    if not isinstance(data, dict):
        return data

    properties = schema.get("properties", {})
    for key, value in data.items():
        if key not in properties:
            continue
        expected_type = properties[key].get("type")
        if expected_type in ("array", "object") and isinstance(value, str):
            try:
                data[key] = json.loads(value)
            except json.JSONDecodeError:
                pass

    return data


def _flatten_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    # OCI rejects JSON Schema references ($ref/$defs); inline them for compatibility.
    resolved = deepcopy(schema)
    defs = resolved.get("$defs", {}) or {}
    definitions = resolved.get("definitions", {}) or {}

    def resolve_ref(ref: str) -> Any:
        if ref.startswith("#/$defs/"):
            key = ref.split("/", 2)[2]
            return defs.get(key)
        if ref.startswith("#/definitions/"):
            key = ref.split("/", 2)[2]
            return definitions.get(key)
        return None

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                target = resolve_ref(node["$ref"])
                merged = {k: v for k, v in node.items() if k != "$ref"}
                if target is None:
                    return {k: walk(v) for k, v in merged.items()}
                resolved_target = walk(target)
                if isinstance(resolved_target, dict):
                    return {**resolved_target, **merged}
                return merged
            return {k: walk(v) for k, v in node.items() if k not in ("$defs", "definitions")}
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    flattened: dict[str, Any] = walk(resolved)
    flattened.pop("$defs", None)
    flattened.pop("definitions", None)
    return flattened


class OCIEstimatingTokenizer(EstimatingTokenizer):
    def __init__(self) -> None:
        self.encoding = tiktoken.encoding_for_model("gpt-4o-2024-08-06")

    @override
    async def estimate_token_count(self, prompt: str) -> int:
        tokens = self.encoding.encode(prompt)
        return int(len(tokens) * 1.15)


class OCISchematicGenerator(BaseSchematicGenerator[T]):
    supported_hints = ["temperature", "max_tokens", "top_p", "top_k", "stop"]

    def __init__(
        self,
        model_id: str,
        compartment_id: str,
        config: dict[str, Any],
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        default_temperature: float | None = None,
        default_max_tokens: int | None = None,
        max_context_tokens: int = 8192,
    ) -> None:
        super().__init__(logger=logger, tracer=tracer, meter=meter, model_name=model_id)

        self._compartment_id = compartment_id
        self._client = OCIAsyncHttpClient(config)
        self._is_cohere = model_id.startswith("cohere.")
        self._tokenizer = OCIEstimatingTokenizer()
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens
        self._max_context_tokens = max_context_tokens

    @property
    @override
    def id(self) -> str:
        return f"oci/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> OCIEstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        return self._max_context_tokens

    def _list_arguments(self, hints: Mapping[str, Any]) -> dict[str, Any]:
        args = {k: v for k, v in hints.items() if k in self.supported_hints}

        if "temperature" not in args and self._default_temperature is not None:
            args["temperature"] = self._default_temperature
        if "max_tokens" not in args and self._default_max_tokens is not None:
            args["max_tokens"] = self._default_max_tokens

        return args

    def _build_generic_request(
        self,
        prompt: str,
        hints: Mapping[str, Any],
    ) -> GenericChatRequest:
        args = self._list_arguments(hints)
        if "stop" in args:
            stop_value = args.pop("stop")
            if isinstance(stop_value, str):
                stop_value = [stop_value]
            args["stop"] = stop_value

        # Use function calling instead of response_format for better reliability
        tool_definition = FunctionDefinition(
            type="FUNCTION",
            name=self.schema.__name__,
            description=f"Generate a response following the {self.schema.__name__} schema",
            parameters=_flatten_json_schema(self.schema.model_json_schema()),
        )

        return GenericChatRequest(
            api_format="GENERIC",
            messages=[
                UserMessage(
                    content=[
                        TextContent(
                            text=prompt,
                        )
                    ]
                )
            ],
            tools=[tool_definition],
            tool_choice=ToolChoiceRequired(),
            **args,
        )

    def _build_cohere_request(
        self,
        prompt: str,
        hints: Mapping[str, Any],
    ) -> CohereChatRequest:
        args = self._list_arguments(hints)
        if "stop" in args:
            stop_value = args.pop("stop")
            if isinstance(stop_value, str):
                stop_value = [stop_value]
            args["stop_sequences"] = stop_value

        # Cohere models have a max of 4096 tokens
        if "max_tokens" in args and args["max_tokens"] > 4096:
            args["max_tokens"] = 4096

        # Cohere only supports JSON_OBJECT, not JSON_SCHEMA
        # The schema constraint is in the prompt itself
        response_format = CohereResponseJsonFormat()

        return CohereChatRequest(
            api_format="COHERE",
            message=prompt,
            response_format=response_format,
            **args,
        )

    def _build_chat_details(
        self,
        prompt: str,
        hints: Mapping[str, Any],
    ) -> ChatDetails:
        request = (
            self._build_cohere_request(prompt, hints)
            if self._is_cohere
            else self._build_generic_request(prompt, hints)
        )

        return ChatDetails(
            compartment_id=self._compartment_id,
            serving_mode=OnDemandServingMode(model_id=self.model_name),
            chat_request=request,
        )

    def _get_nested(self, data: Any, *keys: str, default: Any = None) -> Any:
        """Safely get nested values from dict or object."""
        current = data
        for key in keys:
            if current is None:
                return default
            if isinstance(current, dict):
                current = current.get(key)
            else:
                current = getattr(current, key, None)
        return current if current is not None else default

    def _extract_tool_call_arguments(self, response: dict[str, Any]) -> str | None:
        """Extract arguments from function call response (Generic API only)."""
        # OCI API returns camelCase keys
        chat_response = self._get_nested(response, "chatResponse")
        if not chat_response:
            return None

        choices = self._get_nested(chat_response, "choices")
        if not choices:
            return None

        message = self._get_nested(choices[0], "message") if choices else None
        if not message:
            return None

        tool_calls = self._get_nested(message, "toolCalls")
        if not tool_calls:
            return None

        tool_call = tool_calls[0]
        tool_type = self._get_nested(tool_call, "type")
        if tool_type != "FUNCTION":
            return None

        arguments = self._get_nested(tool_call, "arguments")
        if arguments is None:
            return None

        if isinstance(arguments, str):
            return arguments
        if isinstance(arguments, dict):
            return json.dumps(arguments)

        return None

    def _extract_text(self, response: dict[str, Any]) -> str:
        # OCI API returns camelCase keys
        chat_response = self._get_nested(response, "chatResponse")
        if not chat_response:
            return ""

        if self._is_cohere:
            text = self._get_nested(chat_response, "text")
            if text:
                return str(text)
            message = self._get_nested(chat_response, "message")
            return str(message) if message else ""

        choices = self._get_nested(chat_response, "choices")
        if choices:
            message = self._get_nested(choices[0], "message") if choices else None
            if message:
                content = self._get_nested(message, "content")
                if isinstance(content, list):
                    parts = []
                    for part in content:
                        text = self._get_nested(part, "text")
                        if text:
                            parts.append(text)
                    if parts:
                        return "".join(parts)
                if isinstance(content, str):
                    return content

        text = self._get_nested(chat_response, "text")
        if text:
            return str(text)
        message = self._get_nested(chat_response, "message")
        if isinstance(message, str):
            return message

        return ""

    def _extract_usage(self, response: dict[str, Any]) -> tuple[int, int]:
        # OCI API returns camelCase keys
        chat_response = self._get_nested(response, "chatResponse")
        usage = self._get_nested(chat_response, "usage") if chat_response else None

        prompt_tokens = self._get_nested(usage, "promptTokens", default=0) if usage else 0
        completion_tokens = self._get_nested(usage, "completionTokens", default=0) if usage else 0
        return prompt_tokens, completion_tokens

    @policy(
        [
            retry(
                exceptions=(
                    OCIRetryableError,
                    httpx.TimeoutException,
                    httpx.ConnectError,
                ),
                max_exceptions=3,
                wait_times=(1.0, 2.0, 4.0),
            )
        ]
    )
    @override
    async def do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        with self.logger.scope(f"OCI LLM Request ({self.schema.__name__})"):
            return await self._do_generate(prompt, hints)

    async def _do_generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[T]:
        if isinstance(prompt, PromptBuilder):
            prompt = prompt.build()

        chat_details = self._build_chat_details(prompt, hints)

        t_start = time.time()
        try:
            response = await self._client.chat(chat_details)
        except OCIRetryableError:
            self.logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise
        except httpx.HTTPStatusError as e:
            raise OCIRetryableError(str(e)) from e

        t_end = time.time()

        # For Generic API, try to extract from tool call first
        raw_content = None
        if not self._is_cohere:
            raw_content = self._extract_tool_call_arguments(response)

        # Fallback to text extraction
        if not raw_content:
            raw_content = self._extract_text(response) or "{}"

        try:
            json_content = json.loads(normalize_json_output(raw_content))
        except json.JSONDecodeError:
            self.logger.warning(f"Invalid JSON returned by {self.model_name}:\n{raw_content}")
            json_content = jsonfinder.only_json(raw_content)[2]
            self.logger.warning("Found JSON content within model response; continuing...")

        json_content = _fix_double_encoded_fields(json_content, self.schema.model_json_schema())

        try:
            content = self.schema.model_validate(json_content)

            prompt_tokens, completion_tokens = self._extract_usage(response)

            await record_llm_metrics(
                self.meter,
                self.model_name,
                schema_name=self.schema.__name__,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
            )

            return SchematicGenerationResult(
                content=content,
                info=GenerationInfo(
                    schema_name=self.schema.__name__,
                    model=self.id,
                    duration=(t_end - t_start),
                    usage=UsageInfo(
                        input_tokens=prompt_tokens,
                        output_tokens=completion_tokens,
                    ),
                ),
            )
        except ValidationError as e:
            self.logger.error(
                "JSON content returned by "
                f"{self.model_name} does not match expected schema:\n"
                f"{raw_content}\n{e.json(indent=2)}"
            )
            raise


class OCIEmbedder(BaseEmbedder):
    supported_hints = ["input_type", "truncate", "is_echo"]

    def __init__(
        self,
        model_id: str,
        compartment_id: str,
        config: dict[str, Any],
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        dimensions_override: int | None = None,
    ) -> None:
        super().__init__(logger, tracer, meter, model_id)

        self._compartment_id = compartment_id
        self._client = OCIAsyncHttpClient(config)
        self._tokenizer = OCIEstimatingTokenizer()
        self._dimensions_override = dimensions_override
        self._cached_dimensions: int | None = None

    @property
    @override
    def id(self) -> str:
        return f"oci/{self.model_name}"

    @property
    @override
    def tokenizer(self) -> OCIEstimatingTokenizer:
        return self._tokenizer

    @property
    @override
    def max_tokens(self) -> int:
        return 512

    @property
    @override
    def dimensions(self) -> int:
        if self._dimensions_override is not None:
            return self._dimensions_override
        if self._cached_dimensions is not None:
            return self._cached_dimensions
        return 1024

    def _get_nested(self, data: Any, *keys: str, default: Any = None) -> Any:
        """Safely get nested values from dict or object."""
        current = data
        for key in keys:
            if current is None:
                return default
            if isinstance(current, dict):
                current = current.get(key)
            else:
                current = getattr(current, key, None)
        return current if current is not None else default

    @policy(
        [
            retry(
                exceptions=(
                    OCIRetryableError,
                    httpx.TimeoutException,
                    httpx.ConnectError,
                ),
                max_exceptions=3,
                wait_times=(1.0, 2.0, 4.0),
            )
        ]
    )
    @override
    async def do_embed(
        self,
        texts: list[str],
        hints: Mapping[str, Any] = {},
    ) -> EmbeddingResult:
        embed_hints = {k: v for k, v in hints.items() if k in self.supported_hints}

        details = EmbedTextDetails(
            compartment_id=self._compartment_id,
            inputs=texts,
            serving_mode=OnDemandServingMode(model_id=self.model_name),
            **embed_hints,
        )

        try:
            response = await self._client.embed_text(details)
        except OCIRetryableError:
            self.logger.error(RATE_LIMIT_ERROR_MESSAGE)
            raise
        except httpx.HTTPStatusError as e:
            raise OCIRetryableError(str(e)) from e

        embeddings = self._get_nested(response, "embeddings", default=[]) or []

        vectors: list[list[float]] = []
        for item in embeddings:
            if isinstance(item, list):
                vectors.append(item)
                continue
            vector = self._get_nested(item, "embedding")
            if vector is None:
                vector = self._get_nested(item, "values")
            if vector is not None:
                vectors.append(list(vector))

        if self._cached_dimensions is None and vectors:
            self._cached_dimensions = len(vectors[0])
            self.logger.debug(
                f"Detected embedding dimensions for '{self.model_name}': {self._cached_dimensions}"
            )

        return EmbeddingResult(vectors=vectors)


class OCIService(NLPService):
    @staticmethod
    def verify_environment() -> str | None:
        if not os.environ.get("OCI_COMPARTMENT_ID"):
            return """\
            You're using the OCI NLP service, but OCI_COMPARTMENT_ID is not set.
            Please set OCI_COMPARTMENT_ID in your environment before running Parlant.
            """

        required_env_keys = [
            "OCI_USER",
            "OCI_TENANCY",
            "OCI_FINGERPRINT",
            "OCI_KEY_FILE",
            "OCI_REGION",
        ]

        if all(os.environ.get(k) for k in required_env_keys):
            key_path = _expand_path(os.environ["OCI_KEY_FILE"])
            if not os.path.exists(key_path):
                return """\
                OCI_KEY_FILE was provided but the file does not exist.
                Please check the path in OCI_KEY_FILE.
                """
            return None

        config_file = _expand_path(os.environ.get("OCI_CONFIG_FILE", "~/.oci/config"))
        if not os.path.exists(config_file):
            return f"""\
                OCI config file not found at: {config_file}
                Please create a config file or set OCI_CONFIG_FILE to a valid path.
                """

        return None

    def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
        self._logger = logger
        self._tracer = tracer
        self._meter = meter

        self._config = load_oci_config()
        self._compartment_id = os.environ["OCI_COMPARTMENT_ID"]
        self._model_id = os.environ.get("OCI_MODEL_ID", "openai.gpt-oss-120b")
        self._embedding_model_id = os.environ.get(
            "OCI_EMBEDDING_MODEL_ID",
            "cohere.embed-multilingual-v3.0",
        )

        self._default_temperature = 0.2
        if os.environ.get("OCI_TEMPERATURE"):
            self._default_temperature = float(os.environ["OCI_TEMPERATURE"])

        # Default to 4096 tokens if not specified - prevents truncation issues
        self._default_max_tokens = 10000
        if os.environ.get("OCI_MAX_TOKENS"):
            self._default_max_tokens = int(os.environ["OCI_MAX_TOKENS"])

        self._max_context_tokens = int(os.environ.get("OCI_MAX_CONTEXT_TOKENS", "8192"))

        self._embedding_dims_override = None
        if os.environ.get("OCI_EMBEDDING_DIMS"):
            self._embedding_dims_override = int(os.environ["OCI_EMBEDDING_DIMS"])

        embedder_model_id = self._embedding_model_id
        compartment_id = self._compartment_id
        config = self._config
        dims_override = self._embedding_dims_override

        class DynamicOCIEmbedder(OCIEmbedder):
            def __init__(self, logger: Logger, tracer: Tracer, meter: Meter) -> None:
                super().__init__(
                    model_id=embedder_model_id,
                    compartment_id=compartment_id,
                    config=config,
                    logger=logger,
                    tracer=tracer,
                    meter=meter,
                    dimensions_override=dims_override,
                )

        self._dynamic_embedder_class = DynamicOCIEmbedder

        self._logger.info(
            f"Initialized OCIService with model '{self._model_id}' and embedder '{self._embedding_model_id}'"
        )

    @override
    async def get_schematic_generator(
        self, t: type[T], hints: SchematicGeneratorHints = {}
    ) -> OCISchematicGenerator[T]:
        return OCISchematicGenerator[t](  # type: ignore
            model_id=self._model_id,
            compartment_id=self._compartment_id,
            config=self._config,
            logger=self._logger,
            tracer=self._tracer,
            meter=self._meter,
            default_temperature=self._default_temperature,
            default_max_tokens=self._default_max_tokens,
            max_context_tokens=self._max_context_tokens,
        )

    @override
    async def get_embedder(self, hints: EmbedderHints = {}) -> Embedder:
        return self._dynamic_embedder_class(
            logger=self._logger, tracer=self._tracer, meter=self._meter
        )

    @override
    async def get_moderation_service(self) -> ModerationService:
        return NoModeration()
