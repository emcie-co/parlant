# healthcare.py

import parlant.sdk as p
import asyncio
from datetime import datetime


@p.tool
async def get_insurance_providers(context: p.ToolContext) -> p.ToolResult:
    return p.ToolResult(["Mega Insurance", "Acme Insurance"])


@p.tool
async def get_upcoming_slots(context: p.ToolContext) -> p.ToolResult:
    # Simulate fetching available times from a database or API
    return p.ToolResult(data=["Monday 10 AM", "Tuesday 2 PM", "Wednesday 1 PM"])


@p.tool
async def get_later_slots(context: p.ToolContext) -> p.ToolResult:
    # Simulate fetching later available times
    return p.ToolResult(data=["November 3, 11:30 AM", "November 12, 3 PM"])


@p.tool
async def schedule_appointment(context: p.ToolContext, datetime: datetime) -> p.ToolResult:
    # Simulate scheduling the appointment
    return p.ToolResult(data=f"Appointment scheduled for {datetime}")


@p.tool
async def get_lab_results(context: p.ToolContext) -> p.ToolResult:
    # Simulate fetching lab results from a database or API,
    # using the customer ID from the context.
    lab_results = {
        "report": "All tests are within the valid range",
        "prognosis": "Patient is healthy as a horse!",
    }

    return p.ToolResult(
        data={
            "report": lab_results["report"],
            "prognosis": lab_results["prognosis"],
        }
    )


async def add_domain_glossary(agent: p.Agent) -> None:
    await agent.create_term(
        name="Office Phone Number",
        description="The phone number of our office, at +1-234-567-8900",
    )

    await agent.create_term(
        name="Office Hours",
        description="Office hours are Monday to Friday, 9 AM to 5 PM",
    )

    await agent.create_term(
        name="Charles Xavier",
        synonyms=["Professor X"],
        description="The doctor who specializes in neurology and is available on Mondays and Tuesdays.",
    )

    # Add other specific terms and definitions here, as needed...


# <<Add this function>>
async def create_scheduling_journey(server: p.Server, agent: p.Agent) -> p.Journey:
    # Create the journey
    journey = await agent.create_journey(
        title="Schedule an Appointment",
        description="Helps the patient find a time for their appointment.",
        conditions=["The patient wants to schedule an appointment"],
    )

    # First, determine the reason for the appointment
    t0 = await journey.initial_state.transition_to(chat_state="Determine the reason for the visit")

    # Load upcoming appointment slots into context
    t1 = await t0.target.transition_to(tool_state=get_upcoming_slots)

    # Ask which one works for them
    # We will transition conditionally from here based on the patient's response
    t2 = await t1.target.transition_to(
        chat_state="List available times and ask which ones works for them"
    )

    # We'll start with the happy path where the patient picks a time
    t3 = await t2.target.transition_to(
        chat_state="Confirm the details with the patient before scheduling",
        condition="The patient picks a time",
    )

    t4 = await t3.target.transition_to(
        tool_state=schedule_appointment,
        condition="The patient confirms the details",
    )
    t5 = await t4.target.transition_to(chat_state="Confirm the appointment has been scheduled")
    await t5.target.transition_to(state=p.END_JOURNEY)

    # Otherwise, if they say none of the times work, ask for later slots
    t6 = await t2.target.transition_to(
        tool_state=get_later_slots,
        condition="None of those times work for the patient",
    )
    t7 = await t6.target.transition_to(chat_state="List later times and ask if any of them works")

    # Transition back to our happy-path if they pick a time
    await t7.target.transition_to(state=t3.target, condition="The patient picks a time")

    # Otherwise, ask them to call the office
    t8 = await t7.target.transition_to(
        chat_state="Ask the patient to call the office to schedule an appointment",
        condition="None of those times work for the patient either",
    )
    await t8.target.transition_to(state=p.END_JOURNEY)

    # Handle edge-cases deliberately with guidelines

    await journey.create_guideline(
        condition="The patient says their visit is urgent",
        action="Tell them to call the office immediately",
    )

    return journey


async def create_lab_results_journey(server: p.Server, agent: p.Agent) -> p.Journey:
    # Create the journey
    journey = await agent.create_journey(
        title="Lab Results",
        description="Retrieves the patient's lab results and explains them.",
        conditions=["The patient wants to see their lab results"],
    )

    t0 = await journey.initial_state.transition_to(tool_state=get_lab_results)

    await t0.target.transition_to(
        chat_state="Tell the patient that the results are not available yet, and to try again later",
        condition="The lab results could not be found",
    )

    await t0.target.transition_to(
        chat_state="Explain the lab results to the patient - that they are normal",
        condition="The lab results are good - i.e., nothing to worry about",
    )

    await t0.target.transition_to(
        chat_state="Present the results and ask them to call the office "
        "for clarifications on the results as you are not a doctor",
        condition="The lab results are not good - i.e., there's an issue with the patient's health",
    )

    # Handle edge cases with guidelines...

    await agent.create_guideline(
        condition="The patient presses you for more conclusions about the lab results",
        action="Assertively tell them that you cannot help and they should call the office",
    )

    return journey


async def main() -> None:
    async with p.Server() as server:
        agent = await server.create_agent(
            name="Healthcare Agent",
            description="Is empathetic and calming to the patient.",
        )

        await add_domain_glossary(agent)
        scheduling_journey = await create_scheduling_journey(server, agent)
        lab_results_journey = await create_lab_results_journey(server, agent)

        status_inquiry = await agent.create_observation(
            "The patient asks to follow up on their visit, but it's not clear in which way",
        )

        # Use this observation to disambiguate between the two journeys
        await status_inquiry.disambiguate([scheduling_journey, lab_results_journey])

        await agent.create_guideline(
            condition="The patient asks about insurance",
            action="List the insurance providers we accept, and tell them to call the office for more details",
            tools=[get_insurance_providers],
        )

        await agent.create_guideline(
            condition="The patient asks to talk to a human agent",
            action="Ask them to call the office, providing the phone number",
        )

        await agent.create_guideline(
            condition="The patient inquires about something that has nothing to do with our healthcare",
            action="Kindly tell them you cannot assist with off-topic inquiries - do not engage with their request.",
        )


if __name__ == "__main__":
    asyncio.run(main())
