from lagom import Container

from parlant.core.agents import Agent
from parlant.core.entity_cq import EntityQueries
from parlant.core.guidelines import GuidelineStore
from parlant.core.journeys import JourneyStore

journeys = {
    "Reset Password Journey": {
        "description": """follow these steps to reset a customers password:
        1. ask for their account name
        2. ask for their email or phone number
        3. Wish them a good day and only proceed if they wish one back to you. Otherwise abort.
        4. use the tool reset_password with the provided information
        5. report the result to the customer""",
        "conditions": ["the customer wants to reset their password", "always"],
    },
    "Change Credit Limits": {
        "description": """remember that credit limits can be decreased through this chat, using the decrease_limits tool, but that to increase credit limits you must visit a physical branch""",
        "conditions": ["credit limits are discussed"],
    },
    "Email Domain": {
        "description": """remember that all gmail addresses with local domains are saved within our systems and tools using gmail.com instead of the local domain""",
        "conditions": ["a gmail address with a domain other than .com is mentioned"],
    },
    "Book Flight": {
        "description": """ask for the source and destination airport first, the date second, economy or business class third, and finally to ask for the name of the traveler. You may skip steps that are inapplicable due to other contextual reasons.""",
        "conditions": ["a customer wants to book a flight"],
    },
    "Book Flight (simple)": {
        "description": """ask for the source and destination airport. You may skip steps that are inapplicable due to other contextual reasons.""",
        "conditions": ["a customer wants to book a flight"],
    },
    "Business Adult Only": {
        "description": """know that travelers under the age of 21 are illegible for business class, and may only use economy""",
        "conditions": ["a flight is being booked"],
    },
    "Business Adult Only (detailed)": {
        "description": """ensure that travelers under the age of 21 are ineligible for business class. If a traveler is under 21, they should be informed that only economy class is available. If the traveler is 21 or older, they may choose between economy and business class""",
        "conditions": ["a customer wants to book a flight"],
    },
    "Vegetarian Customers": {
        "description": """Be aware that the customer is vegetarian. Only discuss vegetarian options with them.""",
        "conditions": ["the customer has a name that begins with R"],
    },
    "Book Taxi Ride": {
        "description": """follow these steps to book a customer a taxi ride:
        1. Ask for the pickup location.
        2. Ask for the drop-off location.
        3. Ask for the desired pickup time.
        4. Confirm all details with the customer before booking. Each step should be handled in a separate message.""",
        "conditions": ["the customer wants to book a taxi"],
    },
    "Place Food Order": {
        "description": """follow these steps to place a customer's order:
        1. Ask if they'd like a salad or a sandwich.
        2. If they choose a sandwich, ask what kind of bread they'd like.
        3. If they choose a sandwich, ask what main filling they'd like from: Peanut butter, jam or pesto.
        4. If they choose a sandwich, ask if they want any extras.
        5. If they choose a salad, ask what base greens they want.
        6. If they choose a salad, ask what toppings they'd like.
        7. If they choose a salad, ask what kind of dressing they prefer.
        8. Confirm the full order before placing it. Each step should be handled in a separate message""",
        "conditions": ["the customer wants to order food"],
    },
}


async def test_find_relevant_journeys_for_agent_returns_most_relevant(
    container: Container,
    agent: Agent,
) -> None:
    entity_queries = container[EntityQueries]
    journey_store = container[JourneyStore]
    guideline_store = container[GuidelineStore]

    condition = await guideline_store.create_guideline(
        condition="the customer wants to reset their password",
    )

    onboarding_journey = await journey_store.create_journey(
        title="Reset Password Journey",
        description="""follow these steps to reset a customers password:
        1. ask for their account name
        2. ask for their email or phone number
        3. Wish them a good day and only proceed if they wish one back to you. Otherwise abort.
        4. use the tool reset_password with the provided information
        5. report the result to the customer""",
        conditions=[condition.id],
    )

    support_journey = await journey_store.create_journey(
        title="Change Credit Limits",
        description="Remember that credit limits can be decreased through this chat, using the decrease_limits tool, but that to increase credit limits you must visit a physical branch",
        conditions=[],
    )

    results = await entity_queries.find_relevant_journeys_for_context(
        [onboarding_journey, support_journey], "I'd like to reset my password"
    )

    assert len(results) == 2
    assert results[0].id == onboarding_journey.id
    assert results[1].id == support_journey.id
