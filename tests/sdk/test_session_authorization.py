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

import time
from typing import Any

import jwt

from parlant.core.tools import ToolContext, ToolResult
import parlant.sdk as p

from tests.sdk.utils import Context, SDKTest


ADMIN_API_KEY = "test-admin-api-key"
CUSTOMER_JWT_SECRET = "integrator-shared-secret-0123456789abcdef"
CUSTOMER_ID = "cust_alpha"


def mint_customer_jwt(scope: str) -> str:
    now = int(time.time())
    return jwt.encode(
        {"sub": CUSTOMER_ID, "iat": now, "exp": now + 300, "scope": scope},
        CUSTOMER_JWT_SECRET,
        algorithm="HS256",
    )


class _SignedInCustomerTest(SDKTest):
    AUTH = p.AuthConfig(
        admin_api_key=ADMIN_API_KEY,
        customer_jwt_secret=CUSTOMER_JWT_SECRET,
    )

    async def setup(self, server: p.Server) -> None:
        await server.create_customer(name="Customer Alpha", id=CUSTOMER_ID)

        self.agent = await server.create_agent(
            id=f"authorization-agent-{type(self).__name__}",
            name="Test Agent",
            instructions="A travel agent",
        )


class _RecordsWhetherFlightsCanBeBooked(_SignedInCustomerTest):
    """Records, for every turn, whether the current session is authorized to
    book flights, from inside the engine."""

    async def configure_hooks(self, hooks: p.CompassHooks) -> p.CompassHooks:
        self.answers: list[bool] = []

        async def on_acknowledged(
            context: p.EngineContext, payload: Any, exception: Exception | None
        ) -> p.CompassHookResult:
            self.answers.append(p.Session.current.authorization.has_scope("flights:book"))
            return p.CompassHookResult.CALL_NEXT

        hooks.on_acknowledged.append(on_acknowledged)
        return hooks


class Test_that_the_session_is_authorized_for_the_scopes_of_the_current_message(
    _RecordsWhetherFlightsCanBeBooked
):
    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            "Hello",
            self.agent,
            customer_token=mint_customer_jwt("flights:book profile"),
        )

        assert self.answers == [True]


class Test_that_scopes_from_an_earlier_message_do_not_carry_to_a_later_one(
    _RecordsWhetherFlightsCanBeBooked
):
    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            "Hello",
            self.agent,
            customer_token=mint_customer_jwt("flights:book profile"),
        )
        await ctx.send_and_receive_message(
            "What's my name?",
            self.agent,
            reuse_session=True,
            customer_token=mint_customer_jwt("profile"),
        )

        assert self.answers == [True, False]


class Test_that_requiring_a_scope_the_message_was_not_sent_with_raises(_SignedInCustomerTest):
    async def configure_hooks(self, hooks: p.CompassHooks) -> p.CompassHooks:
        self.missing: list[list[str]] = []

        async def on_acknowledged(
            context: p.EngineContext, payload: Any, exception: Exception | None
        ) -> p.CompassHookResult:
            try:
                p.Session.current.authorization.require_scope("flights:book")
            except p.ScopeNotGrantedError as exc:
                self.missing.append(list(exc.scopes))
            return p.CompassHookResult.CALL_NEXT

        hooks.on_acknowledged.append(on_acknowledged)
        return hooks

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            "Hello",
            self.agent,
            customer_token=mint_customer_jwt("profile"),
        )

        assert self.missing == [["flights:book"]]


class Test_that_a_tool_sees_the_scopes_of_the_message_it_answers(_SignedInCustomerTest):
    async def setup(self, server: p.Server) -> None:
        await super().setup(server)

        self.seen: list[bool] = []

        @p.tool
        async def book_flight(context: ToolContext, destination: str) -> ToolResult:
            self.seen.append(p.Session.current.authorization.has_scope("flights:book"))
            return ToolResult(data={"destination": destination, "booked": True})

        await self.agent.create_observation(
            condition="the customer asks to book a flight",
            tools=[book_flight],
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            "Please book me a flight to Paris",
            self.agent,
            customer_token=mint_customer_jwt("flights:book"),
        )

        assert self.seen == [True]
