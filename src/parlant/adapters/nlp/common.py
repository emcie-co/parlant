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


from collections.abc import Mapping
from difflib import get_close_matches
import os
from typing import Any, Protocol, TypeVar, cast

from parlant.core.meter import Counter, Meter
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.nlp.model_routing import (
    InvalidModelArgumentsError,
    MissingProviderCredentialsError,
)
from parlant.core.nlp.react import ServiceTier, ToolResultPart
from parlant.core.persistence.data_collection import DataCollectingSchematicGenerator
from parlant.core.tracer import Tracer


# Staged retrievers and replay fallbacks put results in history without the
# call that produced them. Providers that pair every result with its call drop
# or reject such a result, so their adapters send it as the output of a
# synthetic call under this name, the way they send system updates. The name is
# fixed because tool names like "built-in:rule-retriever-..." aren't valid
# function names; the original name travels in the output.
RESULT_ONLY_TOOL_NAME = "retrieve_context"


def result_only_tool_output(part: ToolResultPart) -> dict[str, Any]:
    """What a synthetic call's output carries for a result that answers no call."""
    return {"tool": part.name, "result": part.content, "is_error": part.is_error}


def require_api_key(provider: str, env_var: str, api_key: str | None = None) -> str:
    """The provider's API key, or an error saying exactly what to set.

    Called from each provider's constructors, so a missing key fails when the
    provider is built rather than as a 401 from the first request — which
    surfaces deep inside an embedding or generation call, far from the cause.
    """
    if key := (api_key or os.environ.get(env_var)):
        return key

    raise MissingProviderCredentialsError(provider, env_var)


def normalize_json_output(raw_output: str) -> str:
    json_start = raw_output.find("```json")

    if json_start != -1:
        json_start = json_start + 7
    else:
        json_start = 0

    json_end = raw_output[json_start:].rfind("```")

    if json_end == -1:
        json_end = len(raw_output[json_start:])

    return raw_output[json_start : json_start + json_end].strip()


_INPUT_TOKENS_COUNTER: Counter
_OUTPUT_TOKENS_COUNTER: Counter
_CACHED_TOKENS_COUNTER: Counter
_COUNTERS_INITIALIZED = False


async def record_llm_metrics(
    meter: Meter,
    model_name: str,
    schema_name: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> None:
    global _COUNTERS_INITIALIZED
    global _INPUT_TOKENS_COUNTER
    global _OUTPUT_TOKENS_COUNTER
    global _CACHED_TOKENS_COUNTER

    if not _COUNTERS_INITIALIZED:
        _INPUT_TOKENS_COUNTER = meter.create_counter(
            name="input_tokens",
            description="Number of input tokens sent to a LLM model",
        )
        _OUTPUT_TOKENS_COUNTER = meter.create_counter(
            name="output_tokens",
            description="Number of output tokens received from a LLM model",
        )
        _CACHED_TOKENS_COUNTER = meter.create_counter(
            name="cached_input_tokens",
            description="Number of input tokens served from cache for a LLM model",
        )

        _COUNTERS_INITIALIZED = True

    await _INPUT_TOKENS_COUNTER.increment(
        input_tokens,
        {"model_name": model_name, "schema_name": schema_name},
    )

    await _OUTPUT_TOKENS_COUNTER.increment(
        output_tokens,
        {"model_name": model_name, "schema_name": schema_name},
    )

    await _CACHED_TOKENS_COUNTER.increment(
        cached_input_tokens,
        {"model_name": model_name, "schema_name": schema_name},
    )


def validate_model_arguments(
    provider: str,
    args: Mapping[str, Any],
    allowed: set[str],
) -> dict[str, Any]:
    unknown = set(args) - allowed
    if not unknown:
        return dict(args)

    rendered = ", ".join(sorted(unknown))
    reserved = unknown & {"logger", "tracer", "meter", "usage_reporter"}
    suggestions = tuple(
        suggestion
        for argument in sorted(unknown - reserved)
        if (
            suggestion := next(
                iter(get_close_matches(argument, sorted(allowed), n=1, cutoff=0.7)),
                None,
            )
        )
    )
    details = (
        f" Framework-owned argument(s) cannot be overridden: {', '.join(sorted(reserved))}."
        if reserved
        else ""
    )
    suggestion_text = (
        f" Did you mean {', '.join(repr(value) for value in suggestions)}?" if suggestions else ""
    )
    raise InvalidModelArgumentsError(
        f"Unsupported {provider} model argument(s): {rendered}.{details}{suggestion_text}",
        provider=provider,
        suggestions=suggestions,
    )


_VALID_SERVICE_TIERS = ("standard", "flex", "priority")


def validate_service_tier(provider: str, value: object) -> ServiceTier:
    """Validate a route-level ``service_tier`` argument value.

    The seam's vocabulary is provider-neutral (``standard`` / ``flex`` /
    ``priority``); each adapter maps it onto its provider's own names.
    """
    if value in _VALID_SERVICE_TIERS:
        return cast(ServiceTier, value)

    hint = (
        ' (OpenAI calls this tier "fast"; select it with "priority" and Parlant maps the name.)'
        if value == "fast"
        else ""
    )
    raise InvalidModelArgumentsError(
        f"Unsupported {provider} service_tier {value!r}; "
        f"valid values: {', '.join(_VALID_SERVICE_TIERS)}.{hint}",
        provider=provider,
        suggestions=(),
    )


def decorate_schematic_generator(
    generator: SchematicGenerator[Any],
    tracer: Tracer,
) -> SchematicGenerator[Any]:
    if os.environ.get("PARLANT_DATA_COLLECTION", "false").lower() in {"false", "no", "0"}:
        return generator
    return DataCollectingSchematicGenerator(generator, tracer)


# Marks a request as originating from Parlant. The value carries no version and no
# deployment detail — its only job is to let a provider count how much of their
# traffic arrives through Parlant.
#
# Note that a custom header is only counted by a provider who has been asked to
# count it. Where a provider offers its own attribution mechanism, prefer that:
# OpenRouter's X-Title, for instance, already feeds their public app rankings.
CLIENT_IDENTITY_HEADER = "X-Parlant-Client"
CLIENT_IDENTITY = "parlant"

_CLIENT_IDENTITY_DISABLED = ("0", "false", "no", "off")


# Structural type for the Stainless clients: they expose their own User-Agent and
# can copy themselves with extra default headers.
class _StainlessClient(Protocol):
    @property
    def user_agent(self) -> str: ...

    def with_options(self, *, default_headers: Mapping[str, str]) -> Any: ...


_StainlessClientT = TypeVar("_StainlessClientT", bound=_StainlessClient)


def client_identity_enabled() -> bool:
    """Whether this deployment identifies its requests as Parlant's."""
    setting = os.environ.get("PARLANT_CLIENT_IDENTITY", "").strip().lower()
    return setting not in _CLIENT_IDENTITY_DISABLED


def parlant_client_headers() -> dict[str, str]:
    """The header that identifies a request as Parlant's.

    Empty when ``PARLANT_CLIENT_IDENTITY`` is set to a falsey value, for
    deployments that send nothing identifying to a third party."""
    return {CLIENT_IDENTITY_HEADER: CLIENT_IDENTITY} if client_identity_enabled() else {}


def parlant_user_agent_token() -> str:
    """Parlant's User-Agent token, or empty when identification is switched off.

    A User-Agent token is the one identification a provider counts without being
    asked, since every provider logs the header already. It is appended to the
    client library's own User-Agent, never substituted for it — the library's
    identity is useful to the provider too."""
    return CLIENT_IDENTITY if client_identity_enabled() else ""


def append_parlant_user_agent(user_agent: str) -> str:
    """Append Parlant's token to a User-Agent, idempotently."""
    token = parlant_user_agent_token()

    if not token or token in user_agent.split():
        return user_agent

    return f"{user_agent} {token}" if user_agent else token


def with_client_identity(client: _StainlessClientT) -> _StainlessClientT:
    """The same SDK client, with Parlant appended to its User-Agent.

    For the Stainless-generated clients (OpenAI, Anthropic, and OpenRouter which
    rides the OpenAI one), which expose their own ``user_agent`` and copy
    themselves through ``with_options``."""
    if not client_identity_enabled():
        return client

    user_agent = client.user_agent

    # Only a real agent string can be appended to. A client that reports something
    # else (a test double, a future SDK that computes it lazily) keeps its own
    # headers untouched rather than having a mangled agent forced onto it.
    if not isinstance(user_agent, str):
        return client

    return cast(
        _StainlessClientT,
        client.with_options(default_headers={"User-Agent": append_parlant_user_agent(user_agent)}),
    )
