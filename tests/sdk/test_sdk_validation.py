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

import asyncio
import inspect
from types import SimpleNamespace
from typing import Callable, cast

import pytest

from parlant.core.async_utils import default_done_callback
from parlant.core.tools import ToolContext, ToolResult
from tests.sdk.utils import Context, SDKTest, get_message

from parlant import sdk as p


async def test_that_match_always_is_exported_at_module_level_only() -> None:
    assert callable(p.MATCH_ALWAYS)
    assert "MATCH_ALWAYS" in p.__all__
    assert not hasattr(p.Rule, "MATCH_ALWAYS")

    result = await p.MATCH_ALWAYS(
        cast(p.RuleEvaluationContext, object()),
        cast(p.Rule, SimpleNamespace(id="always-rule")),
    )

    assert result.id == "always-rule"
    assert result.is_relevant


class Test_that_agent_utter_defaults_to_the_current_session(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.utter_done = asyncio.Event()
        self.uttered_message: str | None = None

        self.agent = await server.create_agent(
            id="sdk-validation-agent-1",
            name="Utter Session Default Agent",
            instructions="Agent for testing that utter defaults to the current session",
        )

        @p.tool
        async def confirm_booking(context: ToolContext) -> ToolResult:
            async def utter_without_session() -> None:
                self.uttered_message = await p.Agent.current.utter(
                    rules=[{"content": "Tell the customer the booking is confirmed."}],
                )

                self.utter_done.set()

            asyncio.create_task(utter_without_session()).add_done_callback(default_done_callback())

            return ToolResult(data={"status": "confirming"})

        await self.agent.create_observation(
            condition="the customer asks to confirm their booking",
            tools=[confirm_booking],
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message_event(
            customer_message="Please confirm my booking",
            recipient=self.agent,
        )

        await asyncio.wait_for(self.utter_done.wait(), timeout=30)
        assert self.uttered_message

        session_events = await ctx.receive_message_events(min_offset=0)
        assert any(get_message(event) == self.uttered_message for event in session_events)


def test_that_observation_condition_is_required_in_sdk_signatures() -> None:
    assert (
        inspect.signature(p.Agent.create_observation).parameters["condition"].default
        is inspect.Parameter.empty
    )


@pytest.mark.parametrize(
    "method",
    [
        p.Agent.create_policy,
        p.Agent.create_observation,
    ],
)
def test_that_match_handler_sdk_signatures_use_only_on_matched(method: object) -> None:
    parameters = inspect.signature(cast(Callable[..., object], method)).parameters

    assert "on_matched" in parameters
    assert "on_selected" not in parameters
    assert "on_message" not in parameters


@pytest.mark.parametrize(
    "method",
    [
        p.Agent.create_policy,
        p.Agent.create_observation,
    ],
)
def test_that_rule_sdk_signatures_use_evaluation_and_discovery_terms(method: object) -> None:
    parameters = inspect.signature(cast(Callable[..., object], method)).parameters

    assert "evaluation_func" in parameters
    assert "evaluator" not in parameters
    assert "matcher" not in parameters
    assert "on_discovered" in parameters


class Test_that_create_observation_rejects_an_empty_condition(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="sdk-validation-agent-2",
            name="Observation validation agent",
            instructions="Agent for testing observation validation",
        )

    async def run(self, ctx: object) -> None:
        with pytest.raises(
            p.SDKError,
            match="create_observation requires 'condition' to be provided",
        ):
            await self.agent.create_observation(condition="")


def test_that_create_agent_requires_an_explicit_id() -> None:
    parameters = inspect.signature(p.Server.create_agent).parameters

    id_parameter = parameters["id"]
    assert id_parameter.default is inspect.Parameter.empty, "id must have no default"
    assert id_parameter.kind is inspect.Parameter.KEYWORD_ONLY, "id must be keyword-only"
