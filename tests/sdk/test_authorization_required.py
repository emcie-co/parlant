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

from typing import Any

from pytest import raises

from parlant.core.tools import ToolContext, ToolResult
import parlant.sdk as p

from tests.sdk.utils import Context, SDKTest


class _BooksFlightsOnlyForSignedInCustomers(SDKTest):
    EFFORT = p.Effort.MEDIUM
    MESSAGE: str | None = None

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id=f"authorization-required-agent-{type(self).__name__}",
            name="Travel Agent",
            instructions="A travel agency's assistant",
            effort=self.EFFORT,
        )

        message = self.MESSAGE

        @p.tool
        async def book_flight(context: ToolContext, destination: str) -> ToolResult:
            if not p.Session.current.authorization.has_scope("flights:book"):
                return ToolResult.authorization_required(
                    scopes=["flights:book"],
                    reason="book your flight",
                    message=message,
                )
            return ToolResult(data={"destination": destination, "booked": True})

        await self.agent.create_observation(
            condition="the customer asks to book a flight",
            tools=[book_flight],
        )

    async def custom_events(self, ctx: Context) -> list[dict[str, Any]]:
        session = await ctx.get_session()
        events = await ctx.client.sessions.list_events(
            session_id=session.id, kinds="custom", min_offset=0
        )
        return [e.model_dump() for e in events]


def _tells_the_customer_to_sign_in(response: str) -> bool:
    return any(phrase in response.lower() for phrase in ("sign in", "sign-in", "log in"))


class Test_that_a_tool_requiring_authorization_emits_an_authorization_required_event(
    _BooksFlightsOnlyForSignedInCustomers
):
    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message("Please book me a flight to Paris", self.agent)

        requests = [
            e
            for e in await self.custom_events(ctx)
            if e["data"].get("type") == "authorization_required"
        ]

        assert len(requests) == 1
        assert requests[0]["source"] == "system"
        assert requests[0]["data"] == {
            "type": "authorization_required",
            "scopes": ["flights:book"],
            "reason": "book your flight",
        }


class Test_that_the_agent_tells_the_customer_to_sign_in(_BooksFlightsOnlyForSignedInCustomers):
    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            "Please book me a flight to Paris", self.agent
        )

        assert _tells_the_customer_to_sign_in(response), response


class Test_that_the_agent_tells_the_customer_to_sign_in_at_low_effort(
    _BooksFlightsOnlyForSignedInCustomers
):
    EFFORT = p.Effort.LOW

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            "Please book me a flight to Paris", self.agent
        )

        assert _tells_the_customer_to_sign_in(response), response


class Test_that_the_tools_own_message_reaches_the_agent(_BooksFlightsOnlyForSignedInCustomers):
    MESSAGE = (
        "Booking needs the customer to sign in first. Their seat to Paris is held "
        "for 15 minutes, so let them know they have 15 minutes."
    )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            "Please book me a flight to Paris", self.agent
        )

        assert "15" in response, response


class Test_that_emitting_a_reserved_custom_event_type_is_rejected(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.server = server
        self.agent = await server.create_agent(
            id="authorization-required-reserved-agent",
            name="Travel Agent",
            instructions="A travel agency's assistant",
        )

    async def run(self, ctx: Context) -> None:
        created = await ctx.client.sessions.create(agent_id=self.agent.id)
        session = await self.server.get_session(id=created.id)

        for reserved in ("authorization_required", "customer.signed_in"):
            with raises(ValueError):
                await session.emit_custom({"type": reserved})

        events = await ctx.client.sessions.list_events(
            session_id=created.id, kinds="custom", min_offset=0
        )
        assert events == []


class Test_that_the_agent_tells_an_ai_agent_caller_that_the_person_it_represents_must_sign_in_without_writing_a_link_or_mentioning_a_button(
    _BooksFlightsOnlyForSignedInCustomers
):
    """An AI agent speaking for the customer has no Sign in button to press, and
    a link the model writes could be anything. Whoever runs the session (the
    platform's A2A gateway, for one) supplies the way to sign in."""

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            "I'm an AI assistant acting for my user. Please book them a flight to Paris.",
            self.agent,
            customer_represented_by_an_agent=True,
        )

        assert _tells_the_customer_to_sign_in(response), response
        assert "button" not in response.lower(), response
        assert "http" not in response.lower(), response
