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

"""Procedural policies: multi-step flows expressed as mermaid charts inside an
ordinary policy's content. This is the compass-era successor of journeys — the
engine reads the conversation against the chart and works out the current step
by itself; there is no graph machinery, only a policy."""

from parlant.core.relationships import RelationshipKind, RelationshipStore
from parlant.core.rules import RuleStore
from tests.sdk.utils import Context, SDKTest
from tests.test_utilities import nlp_test

from parlant import sdk as p


def procedure(description: str, chart_body: str) -> str:
    return f"{description}\n\n```mermaid\nflowchart TD\n{chart_body}\n```"


BOOKING_PROCEDURE = procedure(
    "When the customer wants to book a flight, follow this procedure step "
    "by step, tracking which step the conversation is currently at:",
    """\
    s0(["Start"])
    s1["ask for the destination city"]
    s2["ask for the travel dates"]
    s3["present matching flights"]
    s0 --> s1
    s1 -->|"the customer named a destination"| s2
    s2 -->|"the customer provided dates"| s3""",
)


class Test_that_a_procedural_policy_is_followed(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="procedural-policies-agent-1",
            name="Travel Agent",
            instructions="You help customers plan trips",
        )
        await self.agent.create_policy(
            title="Flight booking",
            content=BOOKING_PROCEDURE,
        )

    async def run(self, ctx: Context) -> None:
        response = await ctx.send_and_receive_message(
            "Hi, I'd like to book a flight please", recipient=self.agent
        )

        assert await nlp_test(
            response,
            "The agent asks the customer for their destination (where they want to fly to)",
        )


class Test_that_a_procedural_policy_advances_along_a_satisfied_edge(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="procedural-policies-agent-2",
            name="Travel Agent",
            instructions="You help customers plan trips",
        )
        await self.agent.create_policy(
            title="Flight booking",
            content=BOOKING_PROCEDURE,
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            "Hi, I'd like to book a flight please", recipient=self.agent
        )
        response = await ctx.send_and_receive_message(
            "I want to fly to Tokyo",
            recipient=self.agent,
            reuse_session=True,
        )

        assert await nlp_test(
            response,
            "The agent asks the customer about their travel dates (when they want to fly)",
        )


class Test_that_branch_conditions_steer_the_procedure(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="procedural-policies-agent-3",
            name="Claims Agent",
            instructions="You process insurance claims",
        )
        await self.agent.create_policy(
            title="Claim intake",
            content=procedure(
                "When the customer reports a claim, follow this procedure:",
                """\
    s0(["Start"])
    s1["ask for the claim amount"]
    s2{"branch on amount"}
    s3["tell the customer a specialist will contact them for review"]
    s4["approve the claim on the spot and confirm it is settled"]
    s0 --> s1
    s1 --> s2
    s2 -->|"the claim exceeds $10,000"| s3
    s2 -->|"the claim is $10,000 or less"| s4""",
            ),
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            "I need to file a claim for storm damage", recipient=self.agent
        )
        response = await ctx.send_and_receive_message(
            "The damage is about $50,000",
            recipient=self.agent,
            reuse_session=True,
        )

        assert await nlp_test(
            response,
            "The agent says a specialist (or a person for further review/escalation) "
            "will be involved or contact the customer, rather than approving on the spot",
        )


class Test_that_a_procedural_policys_on_matched_fires_when_activated(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.matches: list[str] = []

        async def on_matched(ctx: object, evaluation: object) -> None:
            self.matches.append("matched")

        self.agent = await server.create_agent(
            id="procedural-policies-agent-4",
            name="Travel Agent",
            instructions="You help customers plan trips",
        )
        await self.agent.create_policy(
            title="Flight booking",
            content=BOOKING_PROCEDURE,
            on_matched=on_matched,
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message(
            "Hi, I'd like to book a flight please", recipient=self.agent
        )

        assert self.matches


class Test_that_a_procedure_can_reference_a_sub_procedure(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="procedural-policies-agent-5",
            name="Banking Agent",
            instructions="You help customers close accounts",
        )
        await self.agent.create_policy(
            title="Account closure",
            content=procedure(
                "When the customer wants to close their account, follow this procedure:",
                """\
    s0(["Start"])
    s1["confirm the closure request"]
    s2["finalize the closure"]
    subgraph sg0["Sub-procedure: Identity verification"]
        v0(["Start"])
        v1["ask for a government id number"]
        v0 --> v1
    end
    s0 --> s1
    s1 -->|"the customer confirms"| v0
    v1 -->|"an id was provided"| s2""",
            ),
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message("I want to close my account", recipient=self.agent)
        response = await ctx.send_and_receive_message(
            "Yes, I'm sure — please close it",
            recipient=self.agent,
            reuse_session=True,
        )

        assert await nlp_test(
            response,
            "The agent asks the customer for identification "
            "(an id number, a government id, or identity verification)",
        )


class Test_that_a_scoped_observation_depends_on_its_procedure(SDKTest):
    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="procedural-policies-agent-6",
            name="Banking Agent",
            instructions="You help customers with transfers",
        )
        self.policy = await self.agent.create_policy(
            title="Wire transfer",
            content=procedure(
                "When the customer wants to wire money, follow this procedure:",
                """\
    s0(["Start"])
    s1["ask for the amount and recipient"]
    s2["confirm and execute the payment"]
    s0 --> s1
    s1 -->|"details provided"| s2""",
            ),
        )
        self.observation = await self.agent.create_observation(
            condition="the 'Wire transfer' procedure has reached the payment execution step",
            trigger_message_review=True,
            dependencies=[self.policy],
        )

    async def run(self, ctx: Context) -> None:
        # The dependency edge scopes the observation to the procedure and
        # renders the chart into its evaluation prompt (dependency context).
        relationships = await ctx.container[RelationshipStore].list_relationships(
            kind=RelationshipKind.DEPENDENCY,
            source_id=self.observation.id,
        )
        assert relationships

        observation = await ctx.container[RuleStore].read_rule(self.observation.id)
        assert observation.message_review is True


class Test_that_chained_sub_procedures_are_followed_in_order(SDKTest):
    """Three procedures deep (the old three-linked-journeys scenario): loan
    application → identity verification → credit check → approval."""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="procedural-policies-agent-7",
            name="Loan Agent",
            instructions="You process loan applications",
        )
        await self.agent.create_policy(
            title="Loan application",
            content=procedure(
                "When the customer wants to apply for a loan, follow this "
                "procedure step by step, in order:",
                """\
    s0(["Start"])
    subgraph sg0["Sub-procedure: Identity verification"]
        i0(["Start"])
        i1["ask the customer for their full name"]
        i0 --> i1
    end
    subgraph sg1["Sub-procedure: Credit check"]
        c0(["Start"])
        c1["ask the customer for their SSN"]
        c0 --> c1
    end
    s1["inform the customer their loan has been approved"]
    s0 --> i0
    i1 -->|"the customer provided their full name"| c0
    c1 -->|"the customer provided their SSN"| s1""",
            ),
        )

    async def run(self, ctx: Context) -> None:
        first = await ctx.send_and_receive_message(
            "Hi, I'd like to apply for a loan", recipient=self.agent
        )
        assert await nlp_test(first, "The agent asks the customer for their name")

        second = await ctx.send_and_receive_message(
            "My name is Dana Cohen", recipient=self.agent, reuse_session=True
        )
        assert await nlp_test(
            second, "The agent asks the customer for their SSN (social security number)"
        )

        third = await ctx.send_and_receive_message(
            "My SSN is 123-45-6789", recipient=self.agent, reuse_session=True
        )
        assert await nlp_test(third, "The agent informs the customer that their loan is approved")


class Test_that_the_procedure_routes_to_the_matching_sub_procedure(SDKTest):
    """Conditional routing (the old conditional-links scenario): support triage
    into a technical or a billing sub-procedure based on the customer's issue."""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="procedural-policies-agent-8",
            name="Support Agent",
            instructions="You triage customer support requests",
        )
        await self.agent.create_policy(
            title="Support triage",
            content=procedure(
                "When the customer needs support, follow this procedure:",
                """\
    s0(["Start"])
    s1["ask what kind of issue the customer is having"]
    subgraph sg0["Sub-procedure: Technical support"]
        t0(["Start"])
        t1["ask for the exact error message and the device being used"]
        t0 --> t1
    end
    subgraph sg1["Sub-procedure: Billing support"]
        b0(["Start"])
        b1["ask for the invoice number in question"]
        b0 --> b1
    end
    s0 --> s1
    s1 -->|"the issue is technical (bugs, errors, malfunctions)"| t0
    s1 -->|"the issue concerns billing or charges"| b0""",
            ),
        )

    async def run(self, ctx: Context) -> None:
        await ctx.send_and_receive_message("I need some help please", recipient=self.agent)
        response = await ctx.send_and_receive_message(
            "There's a wrong charge on my last invoice",
            recipient=self.agent,
            reuse_session=True,
        )

        assert await nlp_test(
            response,
            "The agent asks about the invoice (e.g. for the invoice number) "
            "rather than about technical details like error messages or devices",
        )


class Test_that_a_tool_step_is_executed_with_collected_data(SDKTest):
    """Tool steps (the old missing-data lookup scenario): the chart names the
    lookup tools at their step; the agent collects the data, runs the right
    tool, and surfaces its result."""

    async def setup(self, server: p.Server) -> None:
        from parlant.core.services.tools.plugins import tool
        from parlant.core.tools import ToolContext, ToolResult

        @tool
        def find_user_id_by_name(
            context: ToolContext, first_name: str, last_name: str
        ) -> ToolResult:
            return ToolResult(data={"user_id": f"{first_name.lower()}_{last_name.lower()}_8831"})

        @tool
        def find_user_id_by_email(context: ToolContext, email: str) -> ToolResult:
            return ToolResult(data={"user_id": f"{email}_8831"})

        self.agent = await server.create_agent(
            id="procedural-policies-agent-9",
            name="Retail Agent",
            instructions="You are a customer-service agent for an online retail store",
        )
        await self.agent.create_policy(
            title="Account lookup",
            content=procedure(
                "At the start of the conversation, look up the customer's "
                "account before helping them, following this procedure:",
                """\
    s0(["Start"])
    s1["greet the customer and ask for their full name or email address"]
    s2[["look up the account by name (tool: find_user_id_by_name)"]]
    s3[["look up the account by email (tool: find_user_id_by_email)"]]
    s4["tell the customer their exact user ID for reference"]
    s0 --> s1
    s1 -->|"the customer provided their full name"| s2
    s1 -->|"the customer provided their email"| s3
    s2 -->|"the account was found"| s4
    s3 -->|"the account was found"| s4""",
            ),
            tools=[find_user_id_by_name, find_user_id_by_email],
        )

    async def run(self, ctx: Context) -> None:
        first = await ctx.send_and_receive_message("Hello there!", recipient=self.agent)
        assert await nlp_test(first, "The agent asks for the customer's full name or email address")

        second = await ctx.send_and_receive_message(
            "Sure — I'm John Doe", recipient=self.agent, reuse_session=True
        )
        assert "john_doe_8831" in second


class Test_that_a_shared_verification_step_precedes_branching(SDKTest):
    """The old shared sub-journey scenario (insurance claims): one verification
    step gates two different outcomes, chosen by an answer collected earlier."""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="procedural-policies-agent-10",
            name="Insurance Agent",
            instructions="You file insurance claims",
        )
        await self.agent.create_policy(
            title="Claim filing",
            content=procedure(
                "When the customer wants to file an insurance claim, follow this procedure:",
                """\
    s0(["Start"])
    s1["ask whether this is a car insurance claim or a home insurance claim"]
    s2["ask for the customer's policy number to verify their identity"]
    s3{"branch on claim type from step 1"}
    s4["confirm the CAR insurance claim has been filed"]
    s5["confirm the HOME insurance claim has been filed"]
    s0 --> s1
    s1 -->|"the claim type is known"| s2
    s2 -->|"a policy number was provided"| s3
    s3 -->|"it is a car claim"| s4
    s3 -->|"it is a home claim"| s5""",
            ),
        )

    async def run(self, ctx: Context) -> None:
        first = await ctx.send_and_receive_message(
            "I need to file an insurance claim", recipient=self.agent
        )
        assert await nlp_test(
            first, "The agent asks whether the claim is for car or home insurance"
        )

        second = await ctx.send_and_receive_message(
            "It's for my car", recipient=self.agent, reuse_session=True
        )
        assert await nlp_test(second, "The agent asks for the customer's policy number")

        third = await ctx.send_and_receive_message(
            "My policy number is POL-99871", recipient=self.agent, reuse_session=True
        )
        assert await nlp_test(
            third,
            "The agent confirms a CAR insurance claim was filed (not a home claim)",
        )


class Test_that_a_large_nuanced_procedure_is_followed_precisely(SDKTest):
    """One big chart with the nuances that made journeys hard: a retention
    offer ladder (strict order, no skipping ahead, no repeating), reason-based
    branching, a loop-back edge for a changed reason, a conditional bypass
    inside the cancellation sub-procedure, and distinct terminal outcomes."""

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="procedural-policies-agent-11",
            name="ISP Retention Agent",
            instructions="You handle cancellation requests for an internet provider",
        )
        await self.agent.create_policy(
            title="Cancellation and retention",
            content=procedure(
                "When the customer wants to cancel their internet subscription, "
                "follow this procedure step by step. Never skip ahead in the "
                "retention ladder, never repeat an offer the customer already "
                "declined, and only reach cancellation after the ladder is "
                "exhausted:",
                """\
    s0(["Start"])
    s1["ask why the customer wants to cancel"]
    r{"branch on the reason"}
    m1["ask for the new address to check coverage"]
    m2{"is the new address covered?"}
    m3["offer to transfer the subscription to the new address for free"]
    p1["offer a 20% discount for the next 12 months"]
    p2["offer the cheaper Lite plan (half price, lower speed)"]
    t1["apologize, open a technical ticket, and offer one free month as credit"]
    subgraph sg0["Sub-procedure: Cancellation"]
        c0(["Start"])
        c1["verify the account holder: ask for the account PIN"]
        c2{"is the customer still within their 12-month contract period?"}
        c3["inform them of the $99 early-termination fee and ask them to confirm"]
        c4["confirm the cancellation and its effective date"]
        c0 -->|"identity not verified yet this conversation"| c1
        c0 -->|"identity was already verified earlier"| c2
        c1 --> c2
        c2 -->|"yes, within contract"| c3
        c3 -->|"the customer accepts the fee"| c4
        c2 -->|"no, out of contract"| c4
    end
    e1(["End: retained"])
    e2(["End: cancelled"])
    s0 --> s1
    s1 --> r
    r -->|"they are moving to a new home"| m1
    r -->|"it is too expensive"| p1
    r -->|"they are unhappy with the service quality"| t1
    m1 -->|"address provided"| m2
    m2 -->|"covered"| m3
    m2 -->|"not covered"| c0
    m3 -->|"customer accepts the transfer"| e1
    m3 -->|"customer declines the transfer"| c0
    p1 -->|"customer accepts the discount"| e1
    p1 -->|"customer declines the discount"| p2
    p2 -->|"customer accepts the Lite plan"| e1
    p2 -->|"customer declines the Lite plan"| c0
    t1 -->|"customer accepts the credit"| e1
    t1 -->|"customer still insists on cancelling"| c0
    r -->|"the customer changes or clarifies their reason at any point"| s1""",
            ),
        )

    async def run(self, ctx: Context) -> None:
        first = await ctx.send_and_receive_message(
            "Hi, I want to cancel my internet subscription", recipient=self.agent
        )
        assert await nlp_test(first, "The agent asks why the customer wants to cancel")

        # Reason: price → ladder step 1 (the discount), NOT the Lite plan,
        # NOT cancellation.
        second = await ctx.send_and_receive_message(
            "Honestly it's just gotten too expensive for me",
            recipient=self.agent,
            reuse_session=True,
        )
        assert await nlp_test(
            second,
            "The agent offers a discount (about 20% off) on the current plan, "
            "and does NOT mention a cheaper/lite plan and does NOT proceed to "
            "cancel",
        )

        # Declined once → ladder step 2 (the Lite plan), NOT repeating the
        # discount, NOT cancelling yet.
        third = await ctx.send_and_receive_message(
            "No thanks, I'd still like to cancel",
            recipient=self.agent,
            reuse_session=True,
        )
        assert await nlp_test(
            third,
            "The agent offers a cheaper or lite plan (a downgrade option), "
            "and does NOT merely repeat the earlier percentage discount and "
            "does NOT confirm cancellation yet",
        )

        # Ladder exhausted → cancellation sub-procedure begins with identity
        # verification (the PIN), since identity wasn't verified earlier.
        fourth = await ctx.send_and_receive_message(
            "No. Please just cancel it.",
            recipient=self.agent,
            reuse_session=True,
        )
        assert await nlp_test(
            fourth,
            "The agent proceeds toward cancellation by asking the customer to "
            "verify their identity or provide their account PIN (rather than "
            "making further retention offers)",
        )
