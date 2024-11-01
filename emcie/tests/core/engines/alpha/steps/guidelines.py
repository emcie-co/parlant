from pytest_bdd import given, parsers

from emcie.server.core.agents import AgentId
from emcie.server.core.engines.alpha.guideline_proposition import GuidelineProposition
from emcie.server.core.guideline_connections import ConnectionKind, GuidelineConnectionStore
from emcie.server.core.guidelines import Guideline, GuidelineStore
from tests.core.engines.alpha.utils import ContextOfTest, step


@step(given, parsers.parse("a guideline to {do_something} when {a_condition_holds}"))
def given_a_guideline_to_when(
    context: ContextOfTest,
    do_something: str,
    a_condition_holds: str,
    agent_id: AgentId,
) -> None:
    guideline_store = context.container[GuidelineStore]

    context.sync_await(
        guideline_store.create_guideline(
            guideline_set=agent_id,
            predicate=a_condition_holds,
            action=do_something,
        )
    )


@step(
    given,
    parsers.parse('a guideline "{guideline_name}", to {do_something} when {a_condition_holds}'),
)
def given_a_guideline_name_to_when(
    context: ContextOfTest,
    guideline_name: str,
    do_something: str,
    a_condition_holds: str,
    agent_id: AgentId,
) -> None:
    guideline_store = context.container[GuidelineStore]

    context.guidelines[guideline_name] = context.sync_await(
        guideline_store.create_guideline(
            guideline_set=agent_id,
            predicate=a_condition_holds,
            action=do_something,
        )
    )


@step(given, "50 other random guidelines")
def given_50_other_random_guidelines(
    context: ContextOfTest,
    agent_id: AgentId,
) -> list[Guideline]:
    guideline_store = context.container[GuidelineStore]

    async def create_guideline(predicate: str, action: str) -> Guideline:
        return await guideline_store.create_guideline(
            guideline_set=agent_id,
            predicate=predicate,
            action=action,
        )

    guidelines: list[Guideline] = []

    for guideline_params in [
        {
            "predicate": "The user mentions being hungry",
            "action": "Suggest our pizza specials to the user",
        },
        {
            "predicate": "The user asks about vegetarian options",
            "action": "list all vegetarian pizza options",
        },
        {
            "predicate": "The user inquires about delivery times",
            "action": "Provide the estimated delivery time based on their location",
        },
        {
            "predicate": "The user seems undecided",
            "action": "Recommend our top three most popular pizzas",
        },
        {
            "predicate": "The user asks for discount or promotions",
            "action": "Inform the user about current deals or coupons",
        },
        {
            "predicate": "The conversation starts",
            "action": "Greet the user and ask if they'd like to order a pizza",
        },
        {
            "predicate": "The user mentions a food allergy",
            "action": "Ask for specific allergies and recommend safe menu options",
        },
        {
            "predicate": "The user requests a custom pizza",
            "action": "Guide the user through choosing base, sauce, toppings, and cheese",
        },
        {
            "predicate": "The user wants to repeat a previous order",
            "action": "Retrieve the user’s last order details and confirm if they want the same",
        },
        {
            "predicate": "The user asks about portion sizes",
            "action": "Describe the different pizza sizes and how many they typically serve",
        },
        {
            "predicate": "The user requests a drink",
            "action": "list available beverages and suggest popular pairings with "
            "their pizza choice",
        },
        {
            "predicate": "The user asks for the price",
            "action": "Provide the price of the selected items and any additional costs",
        },
        {
            "predicate": "The user expresses concern about calories",
            "action": "Offer information on calorie content and suggest lighter "
            "options if desired",
        },
        {
            "predicate": "The user mentions a special occasion",
            "action": "Suggest our party meal deals and ask if they would like "
            "to include desserts",
        },
        {
            "predicate": "The user wants to know the waiting area",
            "action": "Inform about the waiting facilities at our location or "
            "suggest comfortable seating arrangements",
        },
        {
            "predicate": "The user is comparing pizza options",
            "action": "Highlight the unique features of different pizzas we offer",
        },
        {
            "predicate": "The user asks for recommendations",
            "action": "Suggest pizzas based on their previous orders or popular trends",
        },
        {
            "predicate": "The user is interested in combo deals",
            "action": "Explain the different combo offers and their benefits",
        },
        {
            "predicate": "The user asks if ingredients are fresh",
            "action": "Assure them of the freshness and quality of our ingredients",
        },
        {
            "predicate": "The user wants to modify an order",
            "action": "Assist in making the desired changes and confirm the new order details",
        },
        {
            "predicate": "The user has connectivity issues during ordering",
            "action": "Suggest completing the order via a different method (phone, app)",
        },
        {
            "predicate": "The user expresses dissatisfaction with a previous order",
            "action": "Apologize and offer a resolution (discount, replacement)",
        },
        {
            "predicate": "The user inquires about loyalty programs",
            "action": "Describe our loyalty program benefits and enrollment process",
        },
        {
            "predicate": "The user is about to end the conversation without ordering",
            "action": "Offer a quick summary of unique selling points or a one-time "
            "discount to encourage purchase",
        },
        {
            "predicate": "The user asks for gluten-free options",
            "action": "list our gluten-free pizza bases and toppings",
        },
        {
            "predicate": "The user is looking for side orders",
            "action": "Recommend complementary side dishes like garlic bread or salads",
        },
        {
            "predicate": "The user mentions children",
            "action": "Suggest our kids' menu or family-friendly options",
        },
        {
            "predicate": "The user is having trouble with the online payment",
            "action": "Offer assistance with the payment process or propose an "
            "alternative payment method",
        },
        {
            "predicate": "The user wants to know the origin of ingredients",
            "action": "Provide information about the source and quality assurance "
            "of our ingredients",
        },
        {
            "predicate": "The user asks for a faster delivery option",
            "action": "Explain express delivery options and any associated costs",
        },
        {
            "predicate": "The user seems interested in healthy eating",
            "action": "Highlight our health-conscious options like salads or "
            "pizzas with whole wheat bases",
        },
        {
            "predicate": "The user wants a contactless delivery",
            "action": "Confirm the address and explain the process for contactless delivery",
        },
        {
            "predicate": "The user is a returning customer",
            "action": "Welcome them back and ask if they would like to order their "
            "usual or try something new",
        },
        {
            "predicate": "The user inquires about our environmental impact",
            "action": "Share information about our sustainability practices and "
            "eco-friendly packaging",
        },
        {
            "predicate": "The user is planning a large event",
            "action": "Offer catering services and discuss bulk order discounts",
        },
        {
            "predicate": "The user seems in a rush",
            "action": "Suggest our quickest delivery option and process the order promptly",
        },
        {
            "predicate": "The user wants to pick up the order",
            "action": "Provide the pickup location and expected time until the order is ready",
        },
        {
            "predicate": "The user expresses interest in a specific topping",
            "action": "Offer additional information about that topping and suggest "
            "other complementary toppings",
        },
        {
            "predicate": "The user is making a business order",
            "action": "Propose our corporate deals and ask about potential regular "
            "orders for business meetings",
        },
        {
            "predicate": "The user asks for cooking instructions",
            "action": "Provide details on how our pizzas are made or instructions "
            "for reheating if applicable",
        },
        {
            "predicate": "The user inquires about the chefs",
            "action": "Share background information on our chefs’ expertise and experience",
        },
        {
            "predicate": "The user asks about non-dairy options",
            "action": "list our vegan cheese alternatives and other non-dairy products",
        },
        {
            "predicate": "The user expresses excitement about a new menu item",
            "action": "Provide more details about the item and suggest adding it to their order",
        },
        {
            "predicate": "The user wants a quiet place to eat",
            "action": "Describe the ambiance of our quieter dining areas or "
            "recommend off-peak times",
        },
        {
            "predicate": "The user asks about our app",
            "action": "Explain the features of our app and benefits of ordering through it",
        },
        {
            "predicate": "The user has difficulty deciding",
            "action": "Offer to make a selection based on their preferences or "
            "our chef’s recommendations",
        },
        {
            "predicate": "The user mentions they are in a specific location",
            "action": "Check if we deliver to that location and inform them about "
            "the nearest outlet",
        },
        {
            "predicate": "The user is concerned about food safety",
            "action": "Reassure them about our health and safety certifications and practices",
        },
        {
            "predicate": "The user is looking for a quiet place to eat",
            "action": "Describe the ambiance of our quieter dining areas or "
            "recommend off-peak times",
        },
        {
            "predicate": "The user shows interest in repeat orders",
            "action": "Introduce features like scheduled deliveries or subscription "
            "services to simplify their future orders",
        },
    ]:
        guidelines.append(context.sync_await(create_guideline(**guideline_params)))

    return guidelines


@step(given, parsers.parse('the guideline called "{guideline_id}"'))
def given_the_guideline_called(
    context: ContextOfTest,
    agent_id: AgentId,
    guideline_id: str,
) -> Guideline:
    guideline_store = context.container[GuidelineStore]

    async def create_guideline(predicate: str, action: str) -> Guideline:
        return await guideline_store.create_guideline(
            guideline_set=agent_id,
            predicate=predicate,
            action=action,
        )

    guidelines = {
        "check_drinks_in_stock": {
            "predicate": "a client asks for a drink",
            "action": "check if the drink is available in stock",
        },
        "check_toppings_in_stock": {
            "predicate": "a client asks about toppings or order pizza with toppings",
            "action": "check what toppings are available in stock",
        },
        "ask_expert_about_Spot": {
            "predicate": "a client asks for information about Spot",
            "action": "ask and get the answer from the expert",
        },
        "check_toppings_or_drinks_in_stock": {
            "predicate": "a client asks for toppings or drinks",
            "action": "check if they are available in stock",
        },
        "calculate_sum": {
            "predicate": "an equation involves adding numbers",
            "action": "calculate the sum",
        },
        "check_drinks_or_toppings_in_stock": {
            "predicate": "a client asks for a drink or toppings",
            "action": "check what drinks or toppings are available in stock",
        },
        "calculate_addition_or_multiplication": {
            "predicate": "an equation contains addition or multiplication",
            "action": "calculate it",
        },
        "retrieve_account_information": {
            "predicate": "asked for information about an account",
            "action": "answer by retrieving the information from the database",
        },
        "calculate_addition": {
            "predicate": "an equation contains an add function",
            "action": "get the result from the add tool",
        },
        "calculate_multiplication": {
            "predicate": "an equation contains a multiply function",
            "action": "get the result from the multiply tool",
        },
        "transfer_money_between_accounts": {
            "predicate": "asked to transfer money from one account to another",
            "action": "check if the account has enough balance to make the transfer"
            "and then proceed with the transfer",
        },
        "retrieve_Spot_information": {
            "predicate": "asked for information about Spot",
            "action": "answer by retrieving the information from the database",
        },
        "retrieve_account_balance": {
            "predicate": "asked for information about an account",
            "action": "answer by retrieving the information from the database",
        },
    }

    guideline = context.sync_await(create_guideline(**guidelines[guideline_id]))

    context.guidelines[guideline_id] = guideline

    return guideline


@step(
    given,
    parsers.parse(
        'that the "{guideline_name}" guideline is proposed with a priority of {score} because {rationale}'  # noqb
    ),
)
def given_a_guideline_proposition(
    context: ContextOfTest,
    guideline_name: str,
    score: int,
    rationale: str,
) -> None:
    guideline = context.guidelines[guideline_name]

    context.guideline_propositions[guideline_name] = GuidelineProposition(
        guideline=guideline,
        score=score,
        rationale=rationale,
    )


@step(
    given,
    parsers.parse(
        'a guideline connection whereby "{guideline_a}" {connection_kind} "{guideline_b}"'
    ),
)
def given_a_guideline_connection(
    context: ContextOfTest,
    guideline_a: str,
    connection_kind: str,
    guideline_b: str,
) -> None:
    store = context.container[GuidelineConnectionStore]

    context.sync_await(
        store.create_connection(
            source=context.guidelines[guideline_a].id,
            target=context.guidelines[guideline_b].id,
            kind={
                "entails": ConnectionKind.ENTAILS,
                "suggests": ConnectionKind.SUGGESTS,
            }[connection_kind],
        )
    )
