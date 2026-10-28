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

from types import SimpleNamespace
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, cast

import pytest

from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.core.agents import AgentId
from parlant.core.common import DefaultBaseModel, IdGenerator
from parlant.core.engines.compass.prompt_builder import PromptBuilder
from parlant.core.engines.engine_context import EngineContext
from parlant.core.nlp.embedding import NullEmbedder
from datetime import timedelta

from parlant.core.application_context import ApplicationContext
from parlant.core.health import (
    NLP_EMBED_KIND,
    NLP_REQUEST_KIND,
    NLP_REQUESTS_COUNTER,
    NLP_TOKENS_COUNTER,
    HealthReporter,
    NLPHealthView,
    ReportRetention,
)
from parlant.core.nlp.common import UsageInfo
from parlant.core.nlp.generation import (
    SchematicGenerationResult,
    SchematicGenerator,
)
from parlant.core.nlp.generation_info import GenerationInfo
from parlant.core.agents import Effort
from parlant.core.nlp.decision import DecisionMaker
from parlant.core.nlp.model_routing import (
    InvalidModelArgumentsError,
    InvalidModelRouteError,
    MissingModelRouteError,
    ModelCapability,
    ModelConfigurationService,
    ModelConfig,
    ModelImplementationRegistry,
    ModelProviderExecutionError,
    ModelProviderUnavailableError,
    ModelRouteBinding,
    ModelRouter,
    ModelRoutingError,
    ModelScope,
    MappingSecretLoader,
    RuntimeModelRouteSource,
    SchematicSchemaMismatchError,
    SecretLoader,
    SecretRef,
    SecretValue,
    TransientModelRouteStore,
    UnsupportedModelCapabilityError,
    UnsupportedRateLimitCapabilityError,
    validate_model_route,
)
from parlant.core.nlp.model_route_store import ModelRouteDocumentStore
from parlant.core.nlp.rate_limits import (
    RateLimit,
    RateLimiter,
    RateLimitPermit,
    TransientRateLimiter,
)
from parlant.core.nlp.react import (
    Message,
    ReactGenerator,
    ReactGeneratorHints,
    ReasoningConfig,
    StreamEvent,
    ToolChoice,
    ToolSpec,
    TurnBuilder,
)
from parlant.core.nlp.tokenization import EstimatingTokenizer, ZeroEstimatingTokenizer


class _Schema(DefaultBaseModel):
    value: str


class _OtherSchema(DefaultBaseModel):
    other_value: str


class _StubDecisionMaker(DecisionMaker):
    def __init__(self, name: str) -> None:
        self.name = name

    async def decide(
        self,
        state: Any,
        questions: Mapping[str, Any],
        hints: Mapping[str, Any] = {},
    ) -> Any:
        raise NotImplementedError

    @property
    def id(self) -> str:
        return self.name

    @property
    def max_tokens(self) -> int:
        return 1


class _Generator(SchematicGenerator[_Schema]):
    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def schema(self) -> type[_Schema]:
        return _Schema

    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[_Schema]:
        raise NotImplementedError

    @property
    def id(self) -> str:
        return self.name

    @property
    def max_tokens(self) -> int:
        return 1

    @property
    def tokenizer(self) -> EstimatingTokenizer:
        return ZeroEstimatingTokenizer()


class _OtherGenerator(SchematicGenerator[_OtherSchema]):
    @property
    def schema(self) -> type[_OtherSchema]:
        return _OtherSchema

    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[_OtherSchema]:
        raise NotImplementedError

    @property
    def id(self) -> str:
        return "other-generator"

    @property
    def max_tokens(self) -> int:
        return 1

    @property
    def tokenizer(self) -> EstimatingTokenizer:
        return ZeroEstimatingTokenizer()


class _CountingSecretLoader(SecretLoader):
    def __init__(self, values: Mapping[str, str]) -> None:
        self._values = values
        self.calls = 0

    async def load(self, ref: SecretRef) -> str:
        self.calls += 1
        return self._values[ref.key]


class _RecordingPermit(RateLimitPermit):
    def __init__(self, limiter: _RecordingRateLimiter) -> None:
        self._limiter = limiter
        self._released = False

    async def release(self) -> None:
        if not self._released:
            self._released = True
            self._limiter.releases += 1


class _RecordingRateLimiter(RateLimiter):
    def __init__(self) -> None:
        self.acquisitions: list[tuple[str, RateLimit]] = []
        self.releases = 0

    async def acquire(self, *, bucket: str, limit: RateLimit) -> RateLimitPermit:
        self.acquisitions.append((bucket, limit))
        return _RecordingPermit(self)


class _ReactGenerator(ReactGenerator):
    def __init__(self) -> None:
        super().__init__(model="react-test")

    @property
    def provider_name(self) -> str:
        return "test"

    def _encode(
        self,
        history: Sequence[Message],
        tools: Sequence[ToolSpec],
        tool_choice: ToolChoice,
        *,
        reasoning: ReasoningConfig,
        hints: ReactGeneratorHints = {},
    ) -> object:
        return object()

    async def _raw_stream(self, request: object) -> AsyncIterator[object]:
        if False:
            yield request

    def _decode(self, raw_event: object, builder: TurnBuilder) -> Sequence[StreamEvent]:
        return []


def _routing(
    rate_limiter: RateLimiter | None = None,
) -> tuple[
    ModelConfigurationService,
    ModelRouter,
    ModelImplementationRegistry,
]:
    store = TransientModelRouteStore()
    runtime = RuntimeModelRouteSource()
    secret_loader = MappingSecretLoader({})
    implementations = ModelImplementationRegistry(secret_loader)
    configuration = ModelConfigurationService(
        store,
        runtime,
        implementations,
    )
    return (
        configuration,
        ModelRouter(
            store,
            runtime,
            implementations,
            rate_limiter or TransientRateLimiter(),
        ),
        implementations,
    )


def _context(agent_id: str = "agent-1") -> EngineContext[Any, Any]:
    return cast(
        EngineContext[Any, Any],
        SimpleNamespace(agent=SimpleNamespace(id=AgentId(agent_id))),
    )


def _provider(
    generator: SchematicGenerator[_Schema],
    reasoning: Any = None,
) -> ModelConfig:
    async def provide(context: EngineContext[Any, Any] | None) -> SchematicGenerator[_Schema]:
        return generator

    return ModelConfig(provide, reasoning=reasoning)


@pytest.mark.asyncio
async def test_that_an_exact_route_beats_a_namespace_wildcard_in_the_same_scope() -> None:
    configuration, router, _ = _routing()
    wildcard = _Generator("wildcard")
    exact = _Generator("exact")

    await configuration.configure(
        ModelScope.agent("agent-1"),
        {
            "review.*": _provider(wildcard),
            "review.tools.low": _provider(exact),
        },
    )

    resolved = await router.get_schematic_generator(
        context=_context(),
        route="review.tools.low",
        schema=_Schema,
    )

    assert resolved is exact


@pytest.mark.asyncio
async def test_that_indexing_does_not_fall_back_to_an_evaluation_route() -> None:
    configuration, router, _ = _routing()

    await configuration.configure(
        ModelScope.agent("agent-1"),
        {"evaluation": _provider(_Generator("evaluation"))},
    )

    with pytest.raises(MissingModelRouteError):
        await router.get_schematic_generator(
            context=_context(),
            route="indexing",
            schema=_Schema,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selector", "matching_effort", "nonmatching_effort"),
    [
        ("review.tools[effort=min]", "min", "low"),
        ("review.tools[effort!=min]", "low", "min"),
        ("review.tools[effort<medium]", "low", "medium"),
        ("review.tools[effort<=medium]", "medium", "high"),
        ("review.tools[effort>medium]", "high", "medium"),
        ("review.tools[effort>=medium]", "medium", "low"),
    ],
)
async def test_that_effort_qualifier_comparisons_select_matching_routes(
    selector: str,
    matching_effort: str,
    nonmatching_effort: str,
) -> None:
    configuration, router, _ = _routing()
    qualified = _Generator("qualified")
    fallback = _Generator("fallback")

    await configuration.configure(
        ModelScope.agent("agent-1"),
        {
            "review.tools": _provider(fallback),
            selector: _provider(qualified),
        },
    )

    matching = await router.get_schematic_generator(
        context=_context(),
        route="review.tools",
        qualifiers={"effort": matching_effort},
        schema=_Schema,
    )
    nonmatching = await router.get_schematic_generator(
        context=_context(),
        route="review.tools",
        qualifiers={"effort": nonmatching_effort},
        schema=_Schema,
    )

    assert matching is qualified
    assert nonmatching is fallback


@pytest.mark.asyncio
async def test_that_multiple_effort_constraints_form_a_conjunctive_range() -> None:
    configuration, router, _ = _routing()
    range_match = _Generator("range")
    fallback = _Generator("fallback")

    await configuration.configure(
        ModelScope.agent("agent-1"),
        {
            "review.tools": _provider(fallback),
            "review.tools[effort>=low, effort<high]": _provider(range_match),
        },
    )

    for effort in ("low", "medium"):
        assert (
            await router.get_schematic_generator(
                context=_context(),
                route="review.tools",
                qualifiers={"effort": effort},
                schema=_Schema,
            )
            is range_match
        )

    for effort in ("min", "high", "max"):
        assert (
            await router.get_schematic_generator(
                context=_context(),
                route="review.tools",
                qualifiers={"effort": effort},
                schema=_Schema,
            )
            is fallback
        )


@pytest.mark.asyncio
async def test_that_route_qualifiers_are_canonicalized_before_storage() -> None:
    configuration, _, _ = _routing()
    scope = ModelScope.agent("agent-1")

    await configuration.configure(
        scope,
        {"review.tools[ effort >= low, effort < high ]": _provider(_Generator("range"))},
    )

    assert set(await configuration.read(scope)) == {"review.tools[effort<high,effort>=low]"}


@pytest.mark.asyncio
async def test_that_a_narrower_effort_constraint_beats_a_broader_one() -> None:
    configuration, router, _ = _routing()
    broad = _Generator("broad")
    narrow = _Generator("narrow")

    await configuration.configure(
        ModelScope.agent("agent-1"),
        {
            "review.tools[effort>=medium]": _provider(broad),
            "review.tools[effort=high]": _provider(narrow),
        },
    )

    resolved = await router.get_schematic_generator(
        context=_context(),
        route="review.tools",
        qualifiers={"effort": "high"},
        schema=_Schema,
    )

    assert resolved is narrow


@pytest.mark.asyncio
async def test_that_path_specificity_beats_a_namespace_effort_qualifier() -> None:
    configuration, router, _ = _routing()
    exact = _Generator("exact")
    qualified_namespace = _Generator("qualified-namespace")

    await configuration.configure(
        ModelScope.agent("agent-1"),
        {
            "matching.distill": _provider(exact),
            "matching.*[effort=high]": _provider(qualified_namespace),
        },
    )

    resolved = await router.get_schematic_generator(
        context=_context(),
        route="matching.distill",
        qualifiers={"effort": "high"},
        schema=_Schema,
    )

    assert resolved is exact


@pytest.mark.asyncio
async def test_that_a_qualified_global_route_selects_a_provider_by_effort() -> None:
    configuration, router, implementations = _routing()
    regular = ZeroEstimatingTokenizer()
    high_effort = ZeroEstimatingTokenizer()

    async def construct_regular(
        model: str,
        args: Mapping[str, Any],
    ) -> EstimatingTokenizer:
        return regular

    async def construct_high_effort(
        model: str,
        args: Mapping[str, Any],
    ) -> EstimatingTokenizer:
        return high_effort

    implementations.register(
        "regular-provider",
        ModelCapability.TOKENIZER,
        construct_regular,
        default="regular-tokenizer",
    )
    implementations.register(
        "high-provider",
        ModelCapability.TOKENIZER,
        construct_high_effort,
        default="high-tokenizer",
    )
    await configuration.configure(
        ModelScope.server(),
        {
            "*": "regular-provider",
            "*[effort>=high]": "high-provider",
        },
    )

    assert (
        await router.get_estimating_tokenizer(
            context=None,
            qualifiers={"effort": "medium"},
        )
        is regular
    )
    assert (
        await router.get_estimating_tokenizer(
            context=None,
            qualifiers={"effort": "high"},
        )
        is high_effort
    )


@pytest.mark.asyncio
async def test_that_equally_specific_overlapping_effort_routes_are_rejected() -> None:
    configuration, _, _ = _routing()

    with pytest.raises(InvalidModelRouteError, match="ambiguous"):
        await configuration.configure(
            ModelScope.agent("agent-1"),
            {
                "review.tools[effort>=medium]": _provider(_Generator("upper")),
                "review.tools[effort<=medium]": _provider(_Generator("lower")),
            },
        )


@pytest.mark.parametrize(
    ("route", "message"),
    [
        ("review.tools[priority=high]", "Unsupported model route qualifier 'priority'"),
        ("review.tools[effort==high]", "Invalid effort qualifier"),
        ("review.tools[effort=medum]", "Did you mean 'medium'"),
        ("review.tools[effort>max]", "does not match any effort level"),
    ],
)
def test_that_invalid_effort_qualifiers_are_rejected(route: str, message: str) -> None:
    with pytest.raises(InvalidModelRouteError, match=message):
        validate_model_route(route, ModelConfig("openai:gpt-4o"))


@pytest.mark.asyncio
async def test_that_embedding_is_the_default_for_task_specific_embedding_routes() -> None:
    configuration, router, _ = _routing()
    default = NullEmbedder()
    task_specific = NullEmbedder()

    async def provide_default(context: EngineContext[Any, Any] | None) -> NullEmbedder:
        return default

    async def provide_task_specific(context: EngineContext[Any, Any] | None) -> NullEmbedder:
        return task_specific

    await configuration.configure(
        ModelScope.server(),
        {
            "embedding": ModelConfig(provide_default),
            "embedding.rules": ModelConfig(provide_task_specific),
        },
    )

    assert await router.get_embedder(context=None, route="embedding.glossary") is default
    assert await router.get_embedder(context=None, route="embedding.rules") is task_specific


@pytest.mark.asyncio
async def test_that_an_agent_wildcard_beats_a_server_exact_route() -> None:
    configuration, router, _ = _routing()
    server_exact = _Generator("server-exact")
    agent_wildcard = _Generator("agent-wildcard")

    await configuration.configure(
        ModelScope.server(),
        {"review.tools.low": _provider(server_exact)},
    )
    await configuration.configure(
        ModelScope.agent("agent-1"),
        {"review.*": _provider(agent_wildcard)},
    )

    resolved = await router.get_schematic_generator(
        context=_context(),
        route="review.tools.low",
        schema=_Schema,
    )

    assert resolved is agent_wildcard


@pytest.mark.asyncio
async def test_that_a_provider_route_uses_the_capability_default() -> None:
    configuration, router, implementations = _routing()
    tokenizer = ZeroEstimatingTokenizer()

    async def construct_tokenizer(
        model: str,
        args: Mapping[str, Any],
    ) -> EstimatingTokenizer:
        assert model == "default-tokenizer"
        assert not args
        return tokenizer

    implementations.register(
        "test-provider",
        ModelCapability.TOKENIZER,
        construct_tokenizer,
        default="default-tokenizer",
    )
    await configuration.configure(ModelScope.server(), {"*": "test-provider"})

    resolved = await router.get_estimating_tokenizer(context=None)

    assert resolved is tokenizer


@pytest.mark.asyncio
async def test_that_a_provider_default_resolver_can_return_a_model_config_with_arguments() -> None:
    configuration, router, implementations = _routing()
    tokenizer = ZeroEstimatingTokenizer()
    received: tuple[str, Mapping[str, Any]] | None = None

    async def construct_tokenizer(
        model: str,
        args: Mapping[str, Any],
    ) -> EstimatingTokenizer:
        nonlocal received
        received = (model, args)
        return tokenizer

    def resolve_default(
        route: str,
        qualifiers: Mapping[str, str],
    ) -> ModelConfig:
        assert route == "tokenizer"
        assert qualifiers == {"effort": "high"}
        return ModelConfig(
            "test-provider:default-tokenizer",
            args={
                "provider": {
                    "preferred_min_throughput": {"p90": 50},
                }
            },
        )

    implementations.register(
        "test-provider",
        ModelCapability.TOKENIZER,
        construct_tokenizer,
        default=resolve_default,
    )
    await configuration.configure(ModelScope.server(), {"*": "test-provider"})

    resolved = await router.get_estimating_tokenizer(
        context=None,
        qualifiers={"effort": "high"},
    )

    assert resolved is tokenizer
    assert received == (
        "default-tokenizer",
        {
            "provider": {
                "preferred_min_throughput": {"p90": 50},
            }
        },
    )


def test_that_a_provider_default_model_config_cannot_select_another_provider() -> None:
    implementations = ModelImplementationRegistry(MappingSecretLoader({}))

    async def construct_tokenizer(
        model: str,
        args: Mapping[str, Any],
    ) -> EstimatingTokenizer:
        return ZeroEstimatingTokenizer()

    implementations.register(
        "test-provider",
        ModelCapability.TOKENIZER,
        construct_tokenizer,
        default=ModelConfig("other-provider:default-tokenizer"),
    )

    with pytest.raises(InvalidModelRouteError, match="owned by provider 'other-provider'"):
        implementations.default_config(
            "test-provider",
            ModelCapability.TOKENIZER,
            "tokenizer",
            {},
        )


@pytest.mark.asyncio
async def test_that_configuring_a_lazy_provider_loads_it_once() -> None:
    configuration, router, implementations = _routing()
    tokenizer = ZeroEstimatingTokenizer()
    load_count = 0

    def load_provider() -> None:
        nonlocal load_count
        load_count += 1

        async def construct_tokenizer(
            model: str,
            args: Mapping[str, Any],
        ) -> EstimatingTokenizer:
            assert model == "lazy-tokenizer"
            assert not args
            return tokenizer

        implementations.register(
            "lazy-provider",
            ModelCapability.TOKENIZER,
            construct_tokenizer,
            default="lazy-tokenizer",
        )

    implementations.register_provider_loader("lazy-provider", load_provider)

    assert "lazy-provider" in implementations.providers
    assert load_count == 0

    await configuration.configure(ModelScope.server(), {"*": "lazy-provider"})
    first = await router.get_estimating_tokenizer(context=None)
    second = await router.get_estimating_tokenizer(context=None)

    assert first is tokenizer
    assert second is tokenizer
    assert load_count == 1


@pytest.mark.asyncio
async def test_that_resolving_a_persisted_route_loads_its_provider() -> None:
    store = TransientModelRouteStore()
    runtime = RuntimeModelRouteSource()
    implementations = ModelImplementationRegistry(MappingSecretLoader({}))
    tokenizer = ZeroEstimatingTokenizer()
    load_count = 0

    def load_provider() -> None:
        nonlocal load_count
        load_count += 1

        async def construct_tokenizer(
            model: str,
            args: Mapping[str, Any],
        ) -> EstimatingTokenizer:
            return tokenizer

        implementations.register(
            "persisted-provider",
            ModelCapability.TOKENIZER,
            construct_tokenizer,
        )

    implementations.register_provider_loader("persisted-provider", load_provider)
    await store.update_routes(
        ModelScope.agent("agent-1"),
        {"tokenizer": ModelRouteBinding(ModelConfig("persisted-provider:tokenizer"))},
    )

    resolved = await ModelRouter(
        store,
        runtime,
        implementations,
        TransientRateLimiter(),
    ).get_estimating_tokenizer(context=_context())

    assert resolved is tokenizer
    assert load_count == 1


@pytest.mark.asyncio
async def test_that_a_lazy_provider_load_failure_has_route_context() -> None:
    configuration, _, implementations = _routing()

    def load_provider() -> None:
        raise ModelProviderUnavailableError(
            "The requested model provider is unavailable",
            provider="unavailable-provider",
        )

    implementations.register_provider_loader("unavailable-provider", load_provider)

    with pytest.raises(ModelProviderUnavailableError) as exc_info:
        await configuration.configure(
            ModelScope.server(),
            {"*": "unavailable-provider"},
        )

    assert exc_info.value.route == "*"
    assert exc_info.value.provider == "unavailable-provider"


@pytest.mark.asyncio
async def test_that_a_model_provider_receives_the_engine_context() -> None:
    configuration, router, _ = _routing()
    context = _context()
    generator = _Generator("callback")
    received_context: EngineContext[Any, Any] | None = None

    async def provide(
        model_context: EngineContext[Any, Any] | None,
    ) -> SchematicGenerator[_Schema]:
        nonlocal received_context
        received_context = model_context
        return generator

    await configuration.configure(
        ModelScope.agent("agent-1"),
        {"review.*": ModelConfig(provide)},
    )

    await router.get_schematic_generator(
        context=context,
        route="review.message.high",
        schema=_Schema,
    )

    assert received_context is context


@pytest.mark.asyncio
async def test_that_an_agent_route_update_is_observed_by_the_next_resolution() -> None:
    configuration, router, _ = _routing()
    first = _Generator("first")
    second = _Generator("second")
    scope = ModelScope.agent("agent-1")

    await configuration.configure(scope, {"review.*": _provider(first)})
    first_resolution = await router.get_schematic_generator(
        context=_context(),
        route="review.tools.high",
        schema=_Schema,
    )

    await configuration.configure(scope, {"review.*": _provider(second)})
    second_resolution = await router.get_schematic_generator(
        context=_context(),
        route="review.tools.high",
        schema=_Schema,
    )

    assert first_resolution is first
    assert second_resolution is second


@pytest.mark.asyncio
async def test_that_a_provider_returning_the_wrong_capability_is_rejected() -> None:
    configuration, router, _ = _routing()

    async def provide_tokenizer(
        context: EngineContext[Any, Any] | None,
    ) -> EstimatingTokenizer:
        return ZeroEstimatingTokenizer()

    await configuration.configure(
        ModelScope.agent("agent-1"),
        {"review.*": ModelConfig(cast(Any, provide_tokenizer))},
    )

    with pytest.raises(ModelRoutingError, match="expected SchematicGenerator"):
        await router.get_schematic_generator(
            context=_context(),
            route="review.tools.low",
            schema=_Schema,
        )


@pytest.mark.asyncio
async def test_that_a_provider_returning_the_wrong_schematic_schema_is_rejected() -> None:
    configuration, router, _ = _routing()

    async def provide_other_schema(
        context: EngineContext[Any, Any] | None,
    ) -> SchematicGenerator[_OtherSchema]:
        return _OtherGenerator()

    await configuration.configure(
        ModelScope.agent("agent-1"),
        {"review.tools.low": ModelConfig(cast(Any, provide_other_schema))},
    )

    with pytest.raises(SchematicSchemaMismatchError) as exc_info:
        await router.get_schematic_generator(
            context=_context(),
            route="review.tools.low",
            schema=_Schema,
        )

    assert exc_info.value.route == "review.tools.low"
    assert exc_info.value.scope == "agent:agent-1"
    assert "_OtherSchema" in str(exc_info.value)
    assert "_Schema" in str(exc_info.value)


@pytest.mark.asyncio
async def test_that_a_missing_route_reports_the_checked_scope_and_fallbacks() -> None:
    _, router, _ = _routing()

    with pytest.raises(MissingModelRouteError) as exc_info:
        await router.get_schematic_generator(
            context=_context(),
            route="review.tools.high",
            schema=_Schema,
        )

    error = exc_info.value
    assert error.route == "review.tools.high"
    assert error.scope == "agent:agent-1"
    assert error.suggestions == ("review.tools.*", "review.*", "*")


@pytest.mark.asyncio
async def test_that_a_provider_route_without_the_required_capability_is_rejected() -> None:
    configuration, router, implementations = _routing()

    async def construct_schematic(
        model: str,
        schema: type[Any],
        args: Mapping[str, Any],
    ) -> SchematicGenerator[Any]:
        return _Generator(model)

    implementations.register_schematic(
        "schematic-only",
        construct_schematic,
        default="local-schema-model",
    )
    await configuration.configure(ModelScope.agent("agent-1"), {"*": "schematic-only"})

    with pytest.raises(UnsupportedModelCapabilityError) as exc_info:
        await router.get_estimating_tokenizer(context=_context())

    error = exc_info.value
    assert error.route == "tokenizer"
    assert error.provider == "schematic-only"
    assert error.capability == ModelCapability.TOKENIZER.value


@pytest.mark.parametrize("route", ["*", "*[effort>=high]"])
@pytest.mark.parametrize(
    "target",
    [
        "openrouter:openai/gpt-4o",
        ModelConfig("openrouter:openai/gpt-4o"),
    ],
)
def test_that_provider_routes_reject_model_names(
    route: str,
    target: str | ModelConfig,
) -> None:
    with pytest.raises(ModelRoutingError, match="bare provider name"):
        validate_model_route(route, target)


@pytest.mark.asyncio
async def test_that_an_unknown_config_provider_suggests_a_registered_provider() -> None:
    configuration, _, implementations = _routing()

    async def construct_tokenizer(
        model: str,
        args: Mapping[str, Any],
    ) -> EstimatingTokenizer:
        return ZeroEstimatingTokenizer()

    implementations.register(
        "openai",
        ModelCapability.TOKENIZER,
        construct_tokenizer,
    )

    with pytest.raises(ModelRoutingError, match="Did you mean 'openai'"):
        await configuration.configure(
            ModelScope.agent("agent-1"),
            {"tokenizer": ModelConfig("openia:gpt-4o")},
        )


@pytest.mark.asyncio
async def test_that_a_config_with_a_secret_value_is_runtime_only() -> None:
    store = TransientModelRouteStore()
    runtime = RuntimeModelRouteSource()
    implementations = ModelImplementationRegistry(MappingSecretLoader({}))
    configuration = ModelConfigurationService(
        store,
        runtime,
        implementations,
    )
    router = ModelRouter(store, runtime, implementations, TransientRateLimiter())
    scope = ModelScope.agent("agent-1")
    received_args: Mapping[str, Any] | None = None

    async def construct_tokenizer(
        model: str,
        args: Mapping[str, Any],
    ) -> EstimatingTokenizer:
        nonlocal received_args
        received_args = args
        return ZeroEstimatingTokenizer()

    implementations.register(
        "openai",
        ModelCapability.TOKENIZER,
        construct_tokenizer,
    )

    await configuration.configure(
        scope,
        {
            "tokenizer": ModelConfig(
                "openai:local-tokenizer",
                {"auth": {"api_key": SecretValue("do-not-persist")}},
            )
        },
    )

    assert not await store.read_routes(scope)
    runtime_config = cast(
        ModelConfig,
        (await runtime.read_routes(scope))["tokenizer"].target,
    )
    auth = cast(Mapping[str, Any], runtime_config.args["auth"])
    assert isinstance(auth["api_key"], SecretValue)
    assert "do-not-persist" not in repr(runtime_config)

    await router.get_estimating_tokenizer(context=_context())

    assert received_args == {"auth": {"api_key": "do-not-persist"}}


@pytest.mark.asyncio
async def test_that_a_secret_reference_is_persisted_and_loaded_after_restart() -> None:
    store = TransientModelRouteStore()
    first_runtime = RuntimeModelRouteSource()
    loader = _CountingSecretLoader({"production/openai/api-key": "resolved-secret"})
    first_implementations = ModelImplementationRegistry(loader)
    configuration = ModelConfigurationService(
        store,
        first_runtime,
        first_implementations,
    )
    scope = ModelScope.agent("agent-1")

    async def first_construct_tokenizer(
        model: str,
        args: Mapping[str, Any],
    ) -> EstimatingTokenizer:
        return ZeroEstimatingTokenizer()

    first_implementations.register(
        "openai",
        ModelCapability.TOKENIZER,
        first_construct_tokenizer,
    )
    await configuration.configure(
        scope,
        {
            "tokenizer": ModelConfig(
                "openai:local-tokenizer",
                {"api_key": SecretRef("production/openai/api-key")},
            )
        },
    )

    stored_config = cast(
        ModelConfig,
        (await store.read_routes(scope))["tokenizer"].target,
    )
    assert stored_config.args["api_key"] == SecretRef("production/openai/api-key")

    received_args: Mapping[str, Any] | None = None
    restarted_implementations = ModelImplementationRegistry(loader)

    async def restarted_construct_tokenizer(
        model: str,
        args: Mapping[str, Any],
    ) -> EstimatingTokenizer:
        nonlocal received_args
        received_args = args
        return ZeroEstimatingTokenizer()

    restarted_implementations.register(
        "openai",
        ModelCapability.TOKENIZER,
        restarted_construct_tokenizer,
    )
    restarted_router = ModelRouter(
        store,
        RuntimeModelRouteSource(),
        restarted_implementations,
        TransientRateLimiter(),
    )

    await restarted_router.get_estimating_tokenizer(
        context=_context(),
        agent_id=AgentId("agent-1"),
    )
    await restarted_router.get_estimating_tokenizer(
        context=_context(),
        agent_id=AgentId("agent-1"),
    )

    assert received_args == {"api_key": "resolved-secret"}
    assert loader.calls == 1


@pytest.mark.asyncio
async def test_that_a_secret_reference_round_trips_through_document_storage() -> None:
    database = TransientDocumentDatabase()
    scope = ModelScope.agent("agent-1")
    config = ModelConfig(
        "openai:gpt-4o",
        {
            "auth": {
                "api_key": SecretRef("production/openai/api-key"),
            }
        },
        RateLimit(
            max_concurrency=2,
            requests_per_minute=30,
            bucket="shared-review-quota",
        ),
    )

    async with ModelRouteDocumentStore(IdGenerator(), database) as first_store:
        await first_store.update_routes(
            scope,
            {"review.*": ModelRouteBinding(config)},
        )

    async with ModelRouteDocumentStore(IdGenerator(), database) as restarted_store:
        restored = cast(
            ModelConfig,
            (await restarted_store.read_routes(scope))["review.*"].target,
        )

    assert restored == config


@pytest.mark.asyncio
async def test_that_a_route_reasoning_round_trips_through_document_storage() -> None:
    database = TransientDocumentDatabase()
    scope = ModelScope.agent("agent-1")
    routes = {
        "react": ModelConfig("gemini:gemini-3.8-flash", reasoning="low"),
        "react[effort>=high]": ModelConfig(
            "gemini:gemini-4-argon",
            reasoning=ReasoningConfig(effort="low", visibility="full"),
        ),
        "review.*": ModelConfig("gemini:gemini-3.8-flash"),
    }

    async with ModelRouteDocumentStore(IdGenerator(), database) as first_store:
        await first_store.update_routes(
            scope,
            {selector: ModelRouteBinding(config) for selector, config in routes.items()},
        )

    async with ModelRouteDocumentStore(IdGenerator(), database) as restarted_store:
        restored = await restarted_store.read_routes(scope)

    assert {selector: binding.target for selector, binding in restored.items()} == routes


@pytest.mark.asyncio
async def test_that_model_instances_are_cached_by_sanitized_execution_identity() -> None:
    configuration, router, implementations = _routing()
    construction_count = 0

    async def construct_tokenizer(
        model: str,
        args: Mapping[str, Any],
    ) -> EstimatingTokenizer:
        nonlocal construction_count
        construction_count += 1
        return ZeroEstimatingTokenizer()

    implementations.register(
        "openai",
        ModelCapability.TOKENIZER,
        construct_tokenizer,
    )
    shared_config = ModelConfig(
        "openai:local-tokenizer",
        {
            "endpoint": "https://models.example.com",
            "api_key": SecretValue("production-openai-api-key"),
        },
    )
    await configuration.configure(
        ModelScope.agent("agent-1"),
        {"tokenizer": shared_config},
    )
    await configuration.configure(
        ModelScope.agent("agent-2"),
        {"tokenizer": shared_config},
    )

    first = await router.get_estimating_tokenizer(context=_context("agent-1"))
    second = await router.get_estimating_tokenizer(context=_context("agent-2"))

    assert first is second
    assert construction_count == 1
    assert "production-openai-api-key" not in repr(implementations._cache)
    assert "https://models.example.com" not in repr(implementations._cache)

    await configuration.configure(
        ModelScope.agent("agent-2"),
        {
            "tokenizer": ModelConfig(
                "openai:local-tokenizer",
                {
                    "endpoint": "https://models.example.com",
                    "api_key": SecretValue("other-production-openai-api-key"),
                },
            )
        },
    )

    third = await router.get_estimating_tokenizer(context=_context("agent-2"))

    assert third is not first
    assert construction_count == 2


@pytest.mark.asyncio
async def test_that_model_provider_failures_are_wrapped_with_route_context() -> None:
    configuration, router, _ = _routing()

    async def failing_provider(
        context: EngineContext[Any, Any] | None,
    ) -> SchematicGenerator[_Schema]:
        raise RuntimeError("provider-internal-sensitive-detail")

    await configuration.configure(
        ModelScope.agent("agent-1"),
        {"review.tools.low": ModelConfig(failing_provider)},
    )

    with pytest.raises(ModelProviderExecutionError) as exc_info:
        await router.get_schematic_generator(
            context=_context(),
            route="review.tools.low",
            schema=_Schema,
        )

    error = exc_info.value
    assert error.route == "review.tools.low"
    assert error.scope == "agent:agent-1"
    assert error.capability == ModelCapability.SCHEMATIC.value
    assert error.__cause__ is not None
    assert "provider-internal-sensitive-detail" not in str(error)


@pytest.mark.asyncio
async def test_that_model_argument_errors_gain_context_without_exposing_values() -> None:
    configuration, router, implementations = _routing()

    async def reject_arguments(
        model: str,
        args: Mapping[str, Any],
    ) -> EstimatingTokenizer:
        raise InvalidModelArgumentsError(
            "Unsupported openai model argument 'api_keey'. Did you mean 'api_key'?",
            provider="openai",
            suggestions=("api_key",),
        )

    implementations.register(
        "openai",
        ModelCapability.TOKENIZER,
        reject_arguments,
    )
    await configuration.configure(
        ModelScope.agent("agent-1"),
        {
            "tokenizer": ModelConfig(
                "openai:local-tokenizer",
                {"api_keey": "do-not-expose-this-value"},
            )
        },
    )

    with pytest.raises(InvalidModelArgumentsError) as exc_info:
        await router.get_estimating_tokenizer(context=_context())

    error = exc_info.value
    assert error.route == "tokenizer"
    assert error.model == "openai:local-tokenizer"
    assert error.scope == "agent:agent-1"
    assert error.suggestions == ("api_key",)
    assert "do-not-expose-this-value" not in str(error)


def test_that_model_provider_callables_must_be_wrapped_in_model_config() -> None:
    async def provider(context: EngineContext[Any, Any] | None) -> SchematicGenerator[_Schema]:
        return _Generator("provider")

    with pytest.raises(InvalidModelRouteError, match="wrap provider callables"):
        validate_model_route("review.tools.low", cast(Any, provider))


def test_that_callable_model_configs_reject_constructor_arguments() -> None:
    async def provider(context: EngineContext[Any, Any] | None) -> SchematicGenerator[_Schema]:
        return _Generator("provider")

    with pytest.raises(InvalidModelArgumentsError, match="can only be used"):
        ModelConfig(provider, args={"api_key": "secret"})


@pytest.mark.asyncio
async def test_that_one_route_binding_shares_its_rate_limit_across_schematic_tasks() -> None:
    limiter = _RecordingRateLimiter()
    configuration, router, _ = _routing(limiter)
    provider = _provider(_Generator("limited")).model
    limit = RateLimit(max_concurrency=3)
    await configuration.configure(
        ModelScope.agent("agent-1"),
        {"review.*": ModelConfig(provider, rate_limit=limit)},
    )

    tools_generator = await router.get_schematic_generator(
        context=_context(),
        route="review.tools.low",
        schema=_Schema,
    )
    message_generator = await router.get_schematic_generator(
        context=_context(),
        route="review.message.low",
        schema=_Schema,
    )
    with pytest.raises(NotImplementedError):
        await tools_generator.generate("tools")
    with pytest.raises(NotImplementedError):
        await message_generator.generate("message")

    assert len(limiter.acquisitions) == 2
    assert limiter.acquisitions[0][0] == limiter.acquisitions[1][0]
    assert all(acquired_limit == limit for _, acquired_limit in limiter.acquisitions)
    assert limiter.releases == 2


@pytest.mark.asyncio
async def test_that_named_rate_limit_buckets_share_across_route_bindings() -> None:
    limiter = _RecordingRateLimiter()
    configuration, router, _ = _routing(limiter)
    provider = _provider(_Generator("limited")).model
    limit = RateLimit(max_concurrency=1, bucket="provider-quota")
    await configuration.configure(
        ModelScope.agent("agent-1"),
        {
            "review.tools": ModelConfig(provider, rate_limit=limit),
            "review.message": ModelConfig(provider, rate_limit=limit),
        },
    )

    for route in ("review.tools", "review.message"):
        generator = await router.get_schematic_generator(
            context=_context(),
            route=route,
            schema=_Schema,
        )
        with pytest.raises(NotImplementedError):
            await generator.generate(route)

    assert [bucket for bucket, _ in limiter.acquisitions] == [
        "provider-quota",
        "provider-quota",
    ]


@pytest.mark.asyncio
async def test_that_react_provider_requests_are_rate_limited() -> None:
    limiter = _RecordingRateLimiter()
    configuration, router, _ = _routing(limiter)

    async def provider(context: EngineContext[Any, Any] | None) -> ReactGenerator:
        return _ReactGenerator()

    limit = RateLimit(requests_per_minute=60)
    await configuration.configure(
        ModelScope.agent("agent-1"),
        {"react": ModelConfig(provider, rate_limit=limit)},
    )
    generator = await router.get_react_generator(
        context=_context(),
        route="react",
    )
    await generator.step([])

    assert len(limiter.acquisitions) == 1
    assert limiter.acquisitions[0][1] == limit
    assert limiter.releases == 1


@pytest.mark.asyncio
async def test_that_rate_limits_reject_unsupported_model_capabilities() -> None:
    configuration, router, _ = _routing()

    async def provider(context: EngineContext[Any, Any] | None) -> NullEmbedder:
        return NullEmbedder()

    await configuration.configure(
        ModelScope.agent("agent-1"),
        {"embedding": ModelConfig(provider, rate_limit=RateLimit(max_concurrency=1))},
    )

    with pytest.raises(UnsupportedRateLimitCapabilityError, match="schematic and React"):
        await router.get_embedder(context=_context(), route="embedding")


# ── ModelConfig.reasoning: control ReAct reasoning per route/effort ──────────


def test_that_model_config_reasoning_defaults_to_none() -> None:
    assert ModelConfig("openai:gpt-4o").reasoning is None


def test_that_model_config_normalizes_a_reasoning_effort_shorthand() -> None:
    config = ModelConfig("openai:gpt-4o", reasoning="high")

    assert config.reasoning == ReasoningConfig(effort="high")


def test_that_model_config_accepts_a_full_reasoning_config() -> None:
    reasoning = ReasoningConfig(effort="minimal", visibility="none")

    config = ModelConfig("openai:gpt-4o", reasoning=reasoning)

    assert config.reasoning == reasoning


def _react_provider(reasoning: Any = None) -> ModelConfig:
    async def provide(context: EngineContext[Any, Any] | None) -> ReactGenerator:
        return _ReactGenerator()

    return ModelConfig(provide, reasoning=reasoning)


@pytest.mark.asyncio
async def test_that_the_router_resolves_reasoning_from_the_matched_react_config() -> None:
    configuration, router, _ = _routing()

    await configuration.configure(
        ModelScope.agent("agent-1"),
        {"react": _react_provider(ReasoningConfig(effort="high", visibility="full"))},
    )

    reasoning = await router.resolve_react_reasoning(context=_context(), route="react")

    assert reasoning == ReasoningConfig(effort="high", visibility="full")


@pytest.mark.asyncio
async def test_that_react_reasoning_is_none_when_the_route_sets_no_reasoning() -> None:
    configuration, router, _ = _routing()

    await configuration.configure(
        ModelScope.agent("agent-1"),
        {"react": _react_provider()},
    )

    assert await router.resolve_react_reasoning(context=_context(), route="react") is None


@pytest.mark.asyncio
async def test_that_react_reasoning_follows_the_effort_qualifier() -> None:
    configuration, router, _ = _routing()

    await configuration.configure(
        ModelScope.agent("agent-1"),
        {
            "react": _react_provider("low"),
            "react[effort>=high]": _react_provider("high"),
        },
    )

    medium = await router.resolve_react_reasoning(
        context=_context(), route="react", qualifiers={"effort": "medium"}
    )
    high = await router.resolve_react_reasoning(
        context=_context(), route="react", qualifiers={"effort": "high"}
    )

    assert medium == ReasoningConfig(effort="low")
    assert high == ReasoningConfig(effort="high")


@pytest.mark.asyncio
async def test_that_the_router_resolves_reasoning_effort_from_the_matched_schematic_config() -> (
    None
):
    configuration, router, _ = _routing()

    await configuration.configure(
        ModelScope.agent("agent-1"),
        {
            "matching.rank": _provider(
                _Generator("ranker"),
                ReasoningConfig(effort="high", visibility="none"),
            )
        },
    )

    reasoning_effort = await router.resolve_schematic_reasoning_effort(
        context=_context(),
        route="matching.rank",
    )

    assert reasoning_effort == "high"


@pytest.mark.asyncio
async def test_that_schematic_reasoning_effort_follows_the_effort_qualifier() -> None:
    configuration, router, _ = _routing()

    await configuration.configure(
        ModelScope.agent("agent-1"),
        {
            "matching.rank": _provider(_Generator("default"), "minimal"),
            "matching.rank[effort>=high]": _provider(_Generator("high"), "medium"),
        },
    )

    low = await router.resolve_schematic_reasoning_effort(
        context=_context(),
        route="matching.rank",
        qualifiers={"effort": "low"},
    )
    high = await router.resolve_schematic_reasoning_effort(
        context=_context(),
        route="matching.rank",
        qualifiers={"effort": "high"},
    )

    assert low == "minimal"
    assert high == "medium"


@pytest.mark.asyncio
async def test_that_schematic_reasoning_effort_is_none_when_the_route_sets_none() -> None:
    configuration, router, _ = _routing()

    await configuration.configure(
        ModelScope.agent("agent-1"),
        {"matching.rank": _provider(_Generator("ranker"))},
    )

    assert (
        await router.resolve_schematic_reasoning_effort(
            context=_context(),
            route="matching.rank",
        )
        is None
    )


def test_that_moderation_routes_require_the_moderation_capability() -> None:
    router = ModelRouter.__new__(ModelRouter)

    for route in ("moderation", "moderation.customer", "moderation.agent"):
        router._validate_builtin_capability(route, ModelCapability.MODERATION)

    with pytest.raises(UnsupportedModelCapabilityError):
        router._validate_builtin_capability("moderation.customer", ModelCapability.REACT)


###############################################################################
# Health attribution
###############################################################################
#
# When the router is given a HealthReporter, every generator/embedder it
# resolves is wrapped so that requests report health attributed to the ROUTE
# and calling AGENT — the resolution layer is the only place both are known.
# The wrapped inner instance stays shared; inner models do not report.


class _SucceedingGenerator(SchematicGenerator[_Schema]):
    def __init__(self, name: str) -> None:
        self.name = name
        self.call_count = 0

    @property
    def schema(self) -> type[_Schema]:
        return _Schema

    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[_Schema]:
        self.call_count += 1
        return SchematicGenerationResult(
            content=_Schema(value="generated"),
            info=GenerationInfo(
                schema_name="_Schema",
                model=self.name,
                duration=0.01,
                usage=UsageInfo(input_tokens=3, output_tokens=2),
            ),
        )

    @property
    def id(self) -> str:
        return self.name

    @property
    def max_tokens(self) -> int:
        return 1

    @property
    def tokenizer(self) -> EstimatingTokenizer:
        return ZeroEstimatingTokenizer()


class _FailingGenerator(_SucceedingGenerator):
    async def generate(
        self,
        prompt: str | PromptBuilder,
        hints: Mapping[str, Any] = {},
    ) -> SchematicGenerationResult[_Schema]:
        self.call_count += 1
        raise RuntimeError("model exploded")


def _health_reporter() -> HealthReporter:
    reporter = HealthReporter(ApplicationContext(instance_id="test-instance"))
    reporter.configure_retention(
        NLP_REQUEST_KIND, ReportRetention(window=timedelta(minutes=5), max_count=100)
    )
    reporter.configure_retention(
        NLP_EMBED_KIND, ReportRetention(window=timedelta(minutes=5), max_count=100)
    )
    reporter.configure_counter(NLP_REQUESTS_COUNTER, retention=timedelta(minutes=5))
    reporter.configure_counter(NLP_TOKENS_COUNTER, retention=timedelta(minutes=5))
    return reporter


def _routing_with_health(
    reporter: HealthReporter,
) -> tuple[ModelConfigurationService, ModelRouter]:
    store = TransientModelRouteStore()
    runtime = RuntimeModelRouteSource()
    implementations = ModelImplementationRegistry(MappingSecretLoader({}))
    configuration = ModelConfigurationService(store, runtime, implementations)
    router = ModelRouter(
        store,
        runtime,
        implementations,
        TransientRateLimiter(),
        health_reporter=reporter,
    )
    return configuration, router


@pytest.mark.asyncio
async def test_that_routed_generations_report_health_attributed_to_route_and_agent() -> None:
    reporter = _health_reporter()
    configuration, router = _routing_with_health(reporter)
    inner = _SucceedingGenerator("m1")

    await configuration.configure(ModelScope.server(), {"matching.test": _provider(inner)})

    generator = await router.get_schematic_generator(
        context=_context("agent-7"),
        route="matching.test",
        schema=_Schema,
    )
    await generator.generate("hello")

    reports = reporter.read_reports(NLP_REQUEST_KIND)
    assert len(reports) == 1
    attributes = reports[0].attributes
    assert attributes[NLPHealthView.ATTR_ROUTE] == "matching.test"
    assert attributes[NLPHealthView.ATTR_AGENT] == "agent-7"
    assert attributes[NLPHealthView.ATTR_SUCCESS] is True
    assert attributes[NLPHealthView.ATTR_SCHEMA] == "_Schema"
    assert attributes[NLPHealthView.ATTR_LATENCY_MS] >= 0.0


@pytest.mark.asyncio
async def test_that_routes_sharing_a_model_share_the_instance_but_report_separately() -> None:
    reporter = _health_reporter()
    configuration, router = _routing_with_health(reporter)
    inner = _SucceedingGenerator("shared-model")

    await configuration.configure(
        ModelScope.server(),
        {
            "matching.first": _provider(inner),
            "matching.second": _provider(inner),
        },
    )

    for route in ("matching.first", "matching.second"):
        generator = await router.get_schematic_generator(
            context=_context(),
            route=route,
            schema=_Schema,
        )
        await generator.generate("hello")

    assert inner.call_count == 2  # both routes served by the same inner instance

    routes = [
        r.attributes[NLPHealthView.ATTR_ROUTE] for r in reporter.read_reports(NLP_REQUEST_KIND)
    ]
    assert sorted(routes) == ["matching.first", "matching.second"]


@pytest.mark.asyncio
async def test_that_server_scope_calls_report_without_agent_attribution() -> None:
    reporter = _health_reporter()
    configuration, router = _routing_with_health(reporter)

    await configuration.configure(
        ModelScope.server(), {"indexing": _provider(_SucceedingGenerator("m1"))}
    )

    generator = await router.get_schematic_generator(
        context=None,
        route="indexing",
        schema=_Schema,
    )
    await generator.generate("hello")

    (report,) = reporter.read_reports(NLP_REQUEST_KIND)
    assert report.attributes[NLPHealthView.ATTR_ROUTE] == "indexing"
    assert report.attributes[NLPHealthView.ATTR_AGENT] is None


@pytest.mark.asyncio
async def test_that_failed_generations_report_the_error_class_with_attribution() -> None:
    reporter = _health_reporter()
    configuration, router = _routing_with_health(reporter)

    await configuration.configure(
        ModelScope.server(), {"matching.test": _provider(_FailingGenerator("m1"))}
    )

    generator = await router.get_schematic_generator(
        context=_context("agent-9"),
        route="matching.test",
        schema=_Schema,
    )
    with pytest.raises(RuntimeError):
        await generator.generate("hello")

    (report,) = reporter.read_reports(NLP_REQUEST_KIND)
    assert report.attributes[NLPHealthView.ATTR_SUCCESS] is False
    assert report.attributes[NLPHealthView.ATTR_ERROR_CLASS] == "RuntimeError"
    assert report.attributes[NLPHealthView.ATTR_ROUTE] == "matching.test"
    assert report.attributes[NLPHealthView.ATTR_AGENT] == "agent-9"


@pytest.mark.asyncio
async def test_that_each_routed_generation_reports_exactly_once() -> None:
    # Guards against double counting: the router wrapper is the ONLY reporter;
    # inner/base generators no longer emit their own health reports.
    reporter = _health_reporter()
    configuration, router = _routing_with_health(reporter)

    await configuration.configure(
        ModelScope.server(), {"matching.test": _provider(_SucceedingGenerator("m1"))}
    )

    generator = await router.get_schematic_generator(
        context=_context(),
        route="matching.test",
        schema=_Schema,
    )
    await generator.generate("a")
    await generator.generate("b")

    assert len(reporter.read_reports(NLP_REQUEST_KIND)) == 2
    assert reporter.counter_sum(NLP_REQUESTS_COUNTER, timedelta(minutes=1)) == 2
    assert reporter.counter_sum(NLP_TOKENS_COUNTER, timedelta(minutes=1)) == 10


@pytest.mark.asyncio
async def test_that_routed_embeddings_report_health_attributed_to_route() -> None:
    reporter = _health_reporter()
    configuration, router = _routing_with_health(reporter)

    async def provide_embedder(context: EngineContext[Any, Any] | None) -> NullEmbedder:
        return NullEmbedder()

    await configuration.configure(
        ModelScope.server(), {"embedding.rules": ModelConfig(provide_embedder)}
    )

    embedder = await router.get_embedder(
        context=_context("agent-3"),
        route="embedding.rules",
    )
    await embedder.embed(["hello"])

    (report,) = reporter.read_reports(NLP_EMBED_KIND)
    assert report.attributes[NLPHealthView.ATTR_ROUTE] == "embedding.rules"
    assert report.attributes[NLPHealthView.ATTR_AGENT] == "agent-3"
    assert report.attributes[NLPHealthView.ATTR_SUCCESS] is True


###############################################################################
# Decision capability
###############################################################################
#
# A decision model answers typed questions rather than generating content, so a
# route can serve one alongside its schematic model. Capability is a selector
# clause: a bare selector keeps meaning the route's primary capability, and a
# non-primary one must be asked for explicitly — otherwise a DECISION request
# would happily build a decision client around a schematic model's name.


def test_that_a_capability_clause_parses_into_the_selector() -> None:
    selector = ModelRouter.parse_route_selector("matching.rank[capability=decision]")

    assert selector.route == "matching.rank"
    assert selector.capability is ModelCapability.DECISION
    assert selector.canonical == "matching.rank[capability=decision]"


def test_that_a_selector_without_a_capability_clause_carries_none() -> None:
    assert ModelRouter.parse_route_selector("matching.rank").capability is None


def test_that_an_unknown_capability_is_rejected() -> None:
    with pytest.raises(InvalidModelRouteError):
        ModelRouter.parse_route_selector("matching.rank[capability=telepathy]")


def test_that_a_capability_clause_combines_with_an_effort_clause() -> None:
    selector = ModelRouter.parse_route_selector("matching.rank[capability=decision,effort<=medium]")

    assert selector.capability is ModelCapability.DECISION
    assert selector.matches(Effort.LOW)
    assert not selector.matches(Effort.HIGH)


def test_that_matching_routes_permit_both_schematic_and_decision() -> None:
    router = ModelRouter.__new__(ModelRouter)

    router._validate_builtin_capability("matching.rank", ModelCapability.SCHEMATIC)
    router._validate_builtin_capability("matching.rank", ModelCapability.DECISION)

    # An unrelated capability on a matching route is still a configuration error.
    with pytest.raises(UnsupportedModelCapabilityError):
        router._validate_builtin_capability("matching.rank", ModelCapability.EMBEDDING)


def test_that_a_decision_capability_is_rejected_on_an_unrelated_route() -> None:
    router = ModelRouter.__new__(ModelRouter)

    with pytest.raises(UnsupportedModelCapabilityError):
        router._validate_builtin_capability("embedding.rules", ModelCapability.DECISION)


@pytest.mark.asyncio
async def test_that_an_unconfigured_decision_route_yields_no_decision_maker() -> None:
    # "Nobody configured a decision model" is a normal state, not an error: the
    # ranker asks, gets nothing, and keeps its schematic generator.
    store = TransientModelRouteStore()
    runtime = RuntimeModelRouteSource()
    implementations = ModelImplementationRegistry(MappingSecretLoader({}))
    router = ModelRouter(store, runtime, implementations, TransientRateLimiter())

    assert await router.get_decision_maker(context=None, route="matching.rank") is None


@pytest.mark.asyncio
async def test_that_a_capability_clause_routes_each_request_to_its_own_model() -> None:
    # One route name, two models: the request's capability picks which.
    store = TransientModelRouteStore()
    runtime = RuntimeModelRouteSource()
    implementations = ModelImplementationRegistry(MappingSecretLoader({}))
    configuration = ModelConfigurationService(store, runtime, implementations)
    router = ModelRouter(store, runtime, implementations, TransientRateLimiter())

    decision_models: list[str] = []

    async def construct_decision_maker(model: str, args: Mapping[str, Any]) -> Any:
        decision_models.append(model)
        return _StubDecisionMaker(model)

    implementations.register(
        "stub",
        ModelCapability.DECISION,
        construct_decision_maker,
        default="stub-decider",
    )

    async def construct_generator(model: str, schema: Any, args: Mapping[str, Any]) -> Any:
        return _Generator(model)

    implementations.register_schematic(
        "stub",
        construct_generator,
        default="stub-writer",
    )

    await configuration.configure(
        ModelScope.server(),
        {
            "matching.rank": ModelConfig("stub:stub-writer"),
            "matching.rank[capability=decision]": ModelConfig("stub:stub-decider"),
        },
    )

    decision_maker = await router.get_decision_maker(context=None, route="matching.rank")

    assert decision_maker is not None
    assert decision_models == ["stub-decider"]

    # The schematic request on the same route still resolves to the writer.
    generator = await router.get_schematic_generator(
        context=None,
        route="matching.rank",
        schema=_Schema,
    )
    assert "stub-writer" in generator.id


@pytest.mark.asyncio
async def test_that_a_bare_selector_never_serves_a_decision_request() -> None:
    # The hazard this rule exists for: a bare route points at a schematic model,
    # and must not be used to build a decision client around that model's name.
    store = TransientModelRouteStore()
    runtime = RuntimeModelRouteSource()
    implementations = ModelImplementationRegistry(MappingSecretLoader({}))
    configuration = ModelConfigurationService(store, runtime, implementations)
    router = ModelRouter(store, runtime, implementations, TransientRateLimiter())

    async def construct_decision_maker(model: str, args: Mapping[str, Any]) -> Any:
        return _StubDecisionMaker(model)

    async def construct_generator(model: str, schema: Any, args: Mapping[str, Any]) -> Any:
        return _Generator(model)

    implementations.register(
        "stub",
        ModelCapability.DECISION,
        construct_decision_maker,
        default="stub-decider",
    )
    implementations.register_schematic(
        "stub",
        construct_generator,
        default="stub-writer",
    )

    await configuration.configure(
        ModelScope.server(), {"matching.rank": ModelConfig("stub:stub-writer")}
    )

    assert await router.get_decision_maker(context=None, route="matching.rank") is None


@pytest.mark.asyncio
async def test_that_a_provider_wildcard_serves_its_decision_default_by_default() -> None:
    # A provider that registers a decision default serves decisions through the
    # catch-all "*" route, with no explicit [capability=decision] clause — so
    # "use this provider for everything" includes its decision model.
    store = TransientModelRouteStore()
    runtime = RuntimeModelRouteSource()
    implementations = ModelImplementationRegistry(MappingSecretLoader({}))
    configuration = ModelConfigurationService(store, runtime, implementations)
    router = ModelRouter(store, runtime, implementations, TransientRateLimiter())

    decision_models: list[str] = []

    async def construct_decision_maker(model: str, args: Mapping[str, Any]) -> Any:
        decision_models.append(model)
        return _StubDecisionMaker(model)

    async def construct_generator(model: str, schema: Any, args: Mapping[str, Any]) -> Any:
        return _Generator(model)

    implementations.register(
        "stub",
        ModelCapability.DECISION,
        construct_decision_maker,
        default="stub-decider",
    )
    implementations.register_schematic("stub", construct_generator, default="stub-writer")

    await configuration.configure(ModelScope.server(), {"*": "stub"})

    decision_maker = await router.get_decision_maker(context=None, route="matching.rank")

    assert decision_maker is not None
    assert decision_models == ["stub-decider"]


@pytest.mark.asyncio
async def test_that_a_provider_wildcard_without_a_decision_default_yields_no_decision_maker() -> (
    None
):
    # A provider that only does schematic work serves no decisions through "*": the
    # ranker gets None and keeps its schematic generator — never a hard error.
    store = TransientModelRouteStore()
    runtime = RuntimeModelRouteSource()
    implementations = ModelImplementationRegistry(MappingSecretLoader({}))
    configuration = ModelConfigurationService(store, runtime, implementations)
    router = ModelRouter(store, runtime, implementations, TransientRateLimiter())

    async def construct_generator(model: str, schema: Any, args: Mapping[str, Any]) -> Any:
        return _Generator(model)

    implementations.register_schematic("schematic-only", construct_generator, default="only-writer")

    await configuration.configure(ModelScope.server(), {"*": "schematic-only"})

    assert await router.get_decision_maker(context=None, route="matching.rank") is None
