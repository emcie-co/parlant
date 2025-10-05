# travel_voice_agent.py

import parlant.sdk as p
import asyncio
from datetime import datetime


@p.tool
async def get_available_destinations(context: p.ToolContext) -> p.ToolResult:
    return p.ToolResult(
        [
            "Paris, France",
            "Tokyo, Japan",
            "Bali, Indonesia",
            "New York, USA",
        ]
    )


@p.tool
async def get_available_flights(context: p.ToolContext, destination: str) -> p.ToolResult:
    # Simulate fetching available flights from a booking system
    return p.ToolResult(
        data=[
            "Flight 123 - June 15, 9:00 AM, $850",
            "Flight 321 - June 16, 2:30 PM, $720",
            "Flight 987 - June 17, 6:45 PM, $680",
        ]
    )


@p.tool
async def get_alternative_flights(context: p.ToolContext, destination: str) -> p.ToolResult:
    # Simulate fetching alternative flights with different dates
    return p.ToolResult(
        data=[
            "Flight 485 - June 25, 11:00 AM, $920",
            "Flight 516 - July 2, 4:15 PM, $780",
        ]
    )


@p.tool
async def book_flight(context: p.ToolContext, flight_details: str) -> p.ToolResult:
    # Simulate booking the flight
    return p.ToolResult(
        data=f"Flight booked: {flight_details} for {p.Customer.current.name}. "
        f"Confirmation number: TRV-{datetime.now().strftime('%Y%m%d')}-001"
    )


@p.tool
async def get_booking_status(context: p.ToolContext, confirmation_number: str) -> p.ToolResult:
    # Simulate fetching booking status from a reservation system,
    # using the customer ID from the context.
    booking_info = {
        "status": "Confirmed",
        "details": "Flight to Paris on June 15, 9:00 AM. Seat 12A assigned.",
        "notes": "Check-in opens 24 hours before departure.",
    }

    return p.ToolResult(
        data={
            "status": booking_info["status"],
            "details": booking_info["details"],
            "notes": booking_info["notes"],
        }
    )


async def add_domain_glossary(agent: p.Agent) -> None:
    await agent.create_term(
        name="Office Phone Number",
        description="The phone number of our travel agency office, at +1-800-TRAVEL-1",
        synonyms=["contact number", "customer service number", "support line"],
    )

    await agent.create_term(
        name="Baggage Policy",
        description="This describes the rules and fees associated with checked and carry-on baggage.",
        synonyms=["luggage policy", "baggage rules", "carry-on policy"],
    )

    await agent.create_term(
        name="Cancellation Policy",
        description="This outlines the terms and conditions for cancelling a booking, including any fees or deadlines.",
        synonyms=["refund policy", "cancellation terms"],
    )

    await agent.create_term(
        name="Travel Insurance",
        description="An optional service that provides coverage for trip cancellations, medical emergencies, lost luggage, and other travel-related issues.",
        synonyms=["insurance", "trip protection", "travel protection"],
    )

    # Add other specific terms and definitions here, as needed...


async def create_flight_booking_journey(server: p.Server, agent: p.Agent) -> p.Journey:
    # Create the journey
    journey = await agent.create_journey(
        title="Book a Flight",
        description="Helps the customer find and book a flight to their desired destination.",
        conditions=["The customer wants to book a flight"],
    )

    # First, determine the destination
    t0 = await journey.initial_state.transition_to(chat_state="Ask about the destination")

    # Then ask about preferred travel dates
    t1 = await t0.target.transition_to(chat_state="Ask about preferred travel dates")

    # Load available flights into context
    t2 = await t1.target.transition_to(tool_state=get_available_flights)

    # Present flight options
    # We will transition conditionally from here based on the customer's response
    t3 = await t2.target.transition_to(
        chat_state="Present available flights and ask which one works for them"
    )

    # We'll start with the happy path where the customer picks a flight
    t4 = await t3.target.transition_to(
        chat_state="Collect passenger information and confirm booking details before proceeding",
        condition="The customer selects a flight",
    )

    t5 = await t4.target.transition_to(
        tool_state=book_flight,
        condition="The customer confirms the booking details",
    )
    t6 = await t5.target.transition_to(chat_state="Provide confirmation number and booking summary")
    await t6.target.transition_to(state=p.END_JOURNEY)

    # Otherwise, if none of the flights work, offer alternative dates
    t7 = await t3.target.transition_to(
        tool_state=get_alternative_flights,
        condition="None of the flights work for the customer",
    )
    t8 = await t7.target.transition_to(chat_state="Present alternative flights and ask if any work")

    # Transition back to our happy-path if they pick a flight
    await t8.target.transition_to(state=t4.target, condition="The customer selects a flight")

    # Otherwise, ask them to call the office or check our website
    t9 = await t8.target.transition_to(
        chat_state="Suggest calling our office or visiting our website for more options",
        condition="None of the alternative flights work either",
    )
    await t9.target.transition_to(state=p.END_JOURNEY)

    # Handle edge-cases deliberately with guidelines

    await journey.create_guideline(
        condition="The customer mentions they need to travel urgently or it's an emergency",
        action="Direct them to call our office immediately for priority booking assistance",
    )

    await journey.create_guideline(
        condition="The customer asks about visa requirements",
        action="Inform them that visa requirements vary by destination and nationality, and suggest they check with the embassy or consulate",
    )

    return journey


async def create_booking_status_journey(server: p.Server, agent: p.Agent) -> p.Journey:
    # Create the journey
    journey = await agent.create_journey(
        title="Check Booking Status",
        description="Retrieves the customer's booking status and provides relevant information.",
        conditions=["The customer wants to check their booking status"],
    )

    t0 = await journey.initial_state.transition_to(
        chat_state="Ask for the confirmation number or booking reference"
    )

    t1 = await t0.target.transition_to(tool_state=get_booking_status)

    await t1.target.transition_to(
        chat_state="Tell the customer that the booking could not be found and ask them to verify the confirmation number or call the office",
        condition="The booking could not be found",
    )

    await t1.target.transition_to(
        chat_state="Provide the booking details and confirm everything is in order",
        condition="The booking is confirmed and all details are correct",
    )

    await t1.target.transition_to(
        chat_state="Present the booking information and mention any issues or pending actions required",
        condition="The booking has issues or requires customer action",
    )

    # Handle edge cases with guidelines...

    await journey.create_guideline(
        condition="The customer wants to make changes to their booking",
        action="Explain the change policy and direct them to call our office for assistance with modifications",
    )

    await journey.create_guideline(
        condition="The customer is concerned about potential cancellation",
        action="Provide our cancellation policy and suggest they call the office to discuss their options",
    )

    return journey


async def configure_container(container: p.Container) -> p.Container:
    container[p.PerceivedPerformancePolicy] = p.VoiceOptimizedPerceivedPerformancePolicy()
    return container


async def main() -> None:
    async with p.Server(
        configure_container=configure_container,
    ) as server:
        agent = await server.create_agent(
            name="Walker",
            description="Is a knowledgeable travel agent who helps book flights, answer travel questions, and manage reservations.",
        )

        await add_domain_glossary(agent)

        await create_flight_booking_journey(server, agent)
        await create_booking_status_journey(server, agent)

        await agent.create_guideline(
            condition="The customer asks about travel insurance",
            action="Explain our travel insurance options, coverage details, and pricing, then offer to add it to their booking",
        )

        await agent.create_guideline(
            condition="The customer asks about hotel or car rental options",
            action="Inform them that we can help with complete travel packages and suggest they call our office or visit our website for hotel and car rental bookings",
        )

        await agent.create_guideline(
            condition="The customer asks to speak with a human agent",
            action="Provide the office phone number and office hours, and offer to help them with anything else in the meantime",
        )

        await agent.create_guideline(
            condition="The customer asks about destinations or activities unrelated to booking travel",
            action="Acknowledge their interest but explain that you specialize in travel bookings, and gently redirect to how you can help with their travel plans",
        )

        await agent.create_guideline(
            condition="The customer inquires about something that has nothing to do with travel",
            action="Kindly tell them you cannot assist with off-topic inquiries - do not engage with their request.",
        )


if __name__ == "__main__":
    asyncio.run(main())
