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

"""SDK rule callback shims adapt handlers and canned-response field providers."""

from typing import Any, cast

import parlant.sdk as p
from parlant.core.engines.rule_evaluation import RuleEvaluation as _CoreRuleEvaluation
from parlant.core.rules import RuleId

from tests.core.stable.engines.compass.matching.utils import create_engine_context, create_rule


def _core_evaluation() -> _CoreRuleEvaluation:
    return _CoreRuleEvaluation(
        rule=create_rule("Customer asks about pricing"),
        reasoning="clearly on topic",
        is_relevant=True,
        score=0.9,
    )


async def test_that_a_two_parameter_handler_receives_context_and_evaluation() -> None:
    received: list[tuple[Any, Any]] = []

    async def handler(context: p.EngineContext, evaluation: p.RuleEvaluation) -> None:
        received.append((context, evaluation))

    context = create_engine_context(conversation=[])
    await p.Server._create_rule_handler_shim(
        handler, RuleId("rule-1"), cast(Any, context), _core_evaluation()
    )

    assert len(received) == 1
    assert received[0][0] is context
    assert received[0][1].id == "rule-1"
    assert received[0][1].reasoning == "clearly on topic"


async def test_that_a_context_only_handler_receives_just_the_context() -> None:
    received: list[Any] = []

    async def handler(context: p.EngineContext) -> None:
        received.append(context)

    context = create_engine_context(conversation=[])
    await p.Server._create_rule_handler_shim(
        cast(Any, handler), RuleId("rule-1"), cast(Any, context), _core_evaluation()
    )

    assert received == [context]


async def test_that_a_parameterless_handler_is_invoked_bare() -> None:
    calls: list[str] = []

    async def handler() -> None:
        calls.append("fired")

    context = create_engine_context(conversation=[])
    await p.Server._create_rule_handler_shim(
        cast(Any, handler), RuleId("rule-1"), cast(Any, context), _core_evaluation()
    )

    assert calls == ["fired"]


async def test_that_a_var_positional_handler_receives_both_arguments() -> None:
    received: list[tuple[Any, ...]] = []

    async def handler(*args: Any) -> None:
        received.append(args)

    context = create_engine_context(conversation=[])
    await p.Server._create_rule_handler_shim(
        cast(Any, handler), RuleId("rule-1"), cast(Any, context), _core_evaluation()
    )

    assert len(received) == 1
    assert len(received[0]) == 2


async def test_that_a_bound_method_handler_adapts_by_its_own_arity() -> None:
    class _Listener:
        def __init__(self) -> None:
            self.contexts: list[Any] = []

        async def on_matched(self, context: p.EngineContext) -> None:
            self.contexts.append(context)

    listener = _Listener()
    context = create_engine_context(conversation=[])
    await p.Server._create_rule_handler_shim(
        cast(Any, listener.on_matched), RuleId("rule-1"), cast(Any, context), _core_evaluation()
    )

    assert listener.contexts == [context]


async def test_that_a_parameterless_field_provider_is_invoked_bare() -> None:
    async def provider() -> dict[str, str]:
        return {"greeting": "Howdy partner!"}

    context = create_engine_context(conversation=[])
    await p.Server._create_field_provider_shim(
        provider,
        cast(Any, context),
        _core_evaluation(),
    )

    assert context.state.additional_canned_response_fields == {
        "greeting": "Howdy partner!",
    }


async def test_that_a_context_aware_field_provider_receives_the_context() -> None:
    received: list[Any] = []

    async def provider(context: p.EngineContext) -> dict[str, str]:
        received.append(context)
        return {"greeting": "Hello!"}

    context = create_engine_context(conversation=[])
    await p.Server._create_field_provider_shim(
        provider,
        cast(Any, context),
        _core_evaluation(),
    )

    assert received == [context]
    assert context.state.additional_canned_response_fields == {"greeting": "Hello!"}
