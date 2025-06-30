from typing import Any, Sequence
from lagom import Container
from matplotlib import pyplot as plt
import numpy as np

from parlant.core.agents import Agent
from parlant.core.async_utils import safe_gather
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.emissions import EmittedEvent
from parlant.core.engines.alpha.utils import context_variables_to_json
from parlant.core.glossary import Term
from parlant.core.guidelines import Guideline, GuidelineStore
from parlant.core.journeys import Journey, JourneyStore
from parlant.core.nlp.service import NLPService
from parlant.core.sessions import Event, EventSource
from tests.core.common.utils import create_event_message

JOURNEY_DICT = {
    "Reset Password": {
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


async def get_journeys_by_name(
    container: Container,
    journey_names: Sequence[str],
) -> Sequence[Journey]:
    journey_store = container[JourneyStore]
    guideline_store = container[GuidelineStore]

    journeys: list[Journey] = []
    for name in journey_names:
        conditions = [
            await guideline_store.create_guideline(
                condition=c,
            )
            for c in JOURNEY_DICT[name]["conditions"]
        ]

        journey = await journey_store.create_journey(
            title=name,
            description=str(JOURNEY_DICT[name]["description"]),
            conditions=[c.id for c in conditions],
        )
        journeys.append(journey)
    return journeys


async def journey_to_str(
    container: Container,
    journey: Journey,
) -> str:
    guideline_store = container[GuidelineStore]
    guidelines = [await guideline_store.read_guideline(c) for c in journey.conditions]

    return f"Journey(title: {journey.title}, description: {journey.description}, conditions: {[g.content.condition  for g in guidelines]})"


async def get_cosine_similarity(
    container: Container,
    query: str,
    journey: str,
) -> float:
    embedder = await container[NLPService].get_embedder()

    query_embeddings = list((await embedder.embed([query])).vectors)
    q_vector = np.array(query_embeddings[0], dtype=np.float32)

    journey_embeddings = list((await embedder.embed([journey])).vectors)
    j_vector = np.array(journey_embeddings[0], dtype=np.float32)

    cos_sim = np.dot(q_vector, j_vector) / (np.linalg.norm(q_vector) * np.linalg.norm(j_vector))

    return float(cos_sim)


async def get_journeys_similarity(
    container: Container,
    score_calculator: Any,
    interaction_history: Sequence[Event],
    journeys: Sequence[Journey],
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]] = [],
    guidelines: Sequence[Guideline] = [],
    terms: Sequence[Term] = [],
    staged_events: Sequence[EmittedEvent] = [],
) -> dict[str, dict[str, float]]:
    queries = score_calculator.get_query(
        interaction_history, context_variables, guidelines, terms, staged_events
    )
    similarities: dict[str, dict[str, float]] = {}
    for j in journeys:
        chunk_similarities = await score_calculator.get_queries_similarity(container, queries, j)
        similarities[j.title] = {}
        similarities[j.title]["chunks_score"] = chunk_similarities
        score = score_calculator.get_similarity_score(chunk_similarities)
        similarities[j.title]["score"] = float(score)
    return similarities


def _plot_similairty_vs_prefix(
    similarities: dict[str, list[float]],
    title: str,
) -> None:
    plt.figure(figsize=(12, 6))
    for journey, similarity in similarities.items():
        plt.plot(range(1, len(similarity) + 1), similarity, marker="o", label=journey)
    plt.title("Similarity vs Prefix for Journey")
    plt.xlabel("Prefix Length (messages)")
    plt.ylabel("Similarity Score")
    plt.grid(True)

    # Move legend outside the plot area on the right
    plt.legend(
        title="Journey",
        bbox_to_anchor=(1.05, 1),  # X=1.05 (just outside right edge), Y=1 (top)
        loc="upper left",
        borderaxespad=0.0,
    )

    plt.tight_layout()  # Adjust layout to fit the legend
    plt.savefig(f"similarity vs. interaction message {title}.png", bbox_inches="tight")
    plt.show()


async def get_journeys_similarity_for_prefix(
    container: Container,
    score_calculator: Any,
    interaction_history: Sequence[Event],
    journeys: Sequence[Journey],
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]] = [],
    guidelines: Sequence[Guideline] = [],
    terms: Sequence[Term] = [],
    staged_events: Sequence[EmittedEvent] = [],
    title: str = "",
) -> dict[str, list[float]]:
    similarities: dict[str, list[float]] = {}
    for journey in journeys:
        tasks = [
            score_calculator.get_queries_similarity(
                container,
                score_calculator.get_query(
                    interaction_history[:i], context_variables, guidelines, terms, staged_events
                ),
                journey,
            )
            for i in range(1, len(interaction_history) + 1)
        ]
        chunk_similarities = await safe_gather(*tasks)
        prefix_similarities: list[float] = []
        for c in chunk_similarities:
            score = score_calculator.get_similarity_score(c)
            prefix_similarities.append(float(score))
        similarities[journey.title] = prefix_similarities
    _plot_similairty_vs_prefix(similarities, title)
    return similarities


class BasicSimilarityScore:
    def get_query(
        self,
        interaction_history: Sequence[Event],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]] = [],
        guidelines: Sequence[Guideline] = [],
        terms: Sequence[Term] = [],
        staged_events: Sequence[EmittedEvent] = [],
    ) -> Sequence[str]:
        query = ""

        if context_variables:
            query += f"\n{context_variables_to_json(context_variables)}"

        if guidelines:
            query += str(
                [
                    f"When {g.content.condition}, then {g.content.action}"
                    if g.content.action
                    else f"When {g.content.condition}"
                    for g in guidelines
                ]
            )

        if staged_events:
            query += str([e.data for e in staged_events])

        if terms:
            query += str([t.name for t in terms])

        if interaction_history:
            query += str([e.data for e in interaction_history])

        # chunk_lists = await safe_gather(*[query_chunks(q, embedder) for q in queries])
        # all_chunks: list[str] = [chunk for sublist in chunk_lists for chunk in sublist]

        return [query]

    async def get_queries_similarity(
        self,
        container: Container,
        queries: Sequence[str],
        journey: Journey,
    ) -> Sequence[float]:
        journey_doc = await journey_to_str(container, journey)

        tasks = [
            get_cosine_similarity(
                container=container,
                query=q,
                journey=journey_doc,
            )
            for q in queries
        ]
        results = await safe_gather(*tasks)
        return results

    def get_similarity_score(
        self,
        chunk_similarities: Sequence[float],
    ) -> float:
        return float(np.mean(np.asarray(chunk_similarities)))


class LastMessageSimilarityScore:
    def get_query(
        self,
        interaction_history: Sequence[Event],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]] = [],
        guidelines: Sequence[Guideline] = [],
        terms: Sequence[Term] = [],
        staged_events: Sequence[EmittedEvent] = [],
    ) -> Sequence[str]:
        queries: list[str] = []

        query = ""

        if context_variables:
            query += f"\n{context_variables_to_json(context_variables)}"

        if guidelines:
            query += str(
                [
                    f"When {g.content.condition}, then {g.content.action}"
                    if g.content.action
                    else f"When {g.content.condition}"
                    for g in guidelines
                ]
            )

        if staged_events:
            query += str([e.data for e in staged_events])

        if terms:
            query += str([t.name for t in terms])

        if query:
            queries.append(query)

        query = ""

        if interaction_history:
            *previous, last = interaction_history
            if previous:
                query += str([e.data for e in previous]) + "\n\n"
                queries.append(query)

            query += str(last.data)
            queries.append(query)

        return queries

    async def get_queries_similarity(
        self,
        container: Container,
        queries: Sequence[str],
        journey: Journey,
    ) -> Sequence[float]:
        journey_doc = await journey_to_str(container, journey)

        tasks = [
            get_cosine_similarity(
                container=container,
                query=q,
                journey=journey_doc,
            )
            for q in queries
        ]
        results = await safe_gather(*tasks)
        return results

    def get_similarity_score(
        self,
        chunk_similarities: Sequence[float],
    ) -> float:
        if len(chunk_similarities) == 1:
            return chunk_similarities[0]
        return 0.3 * chunk_similarities[0] + 0.7 * chunk_similarities[1]


class JumpBoostSimilarityScore:
    def get_query(
        self,
        interaction_history: Sequence[Event],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]] = [],
        guidelines: Sequence[Guideline] = [],
        terms: Sequence[Term] = [],
        staged_events: Sequence[EmittedEvent] = [],
    ) -> Sequence[str]:
        queries: list[str] = []

        # Shared context part
        base_query = ""

        if context_variables:
            base_query += f"\n{context_variables_to_json(context_variables)}"

        if guidelines:
            base_query += str(
                [
                    f"When {g.content.condition}, then {g.content.action}"
                    if g.content.action
                    else f"When {g.content.condition}"
                    for g in guidelines
                ]
            )

        if staged_events:
            base_query += str([e.data for e in staged_events])

        if terms:
            base_query += str([t.name for t in terms])

        # Add a query for each prefix of the interaction history
        for i in range(1, len(interaction_history) + 1):
            prefix = interaction_history[:i]
            interaction_str = str([e.data for e in prefix])
            queries.append(interaction_str)

        return queries

    def _plot_scores(
        self,
        journey: Journey,
        scores: Sequence[float],
        new_score: Sequence[float],
    ) -> None:
        # Plot old vs new
        plt.figure(figsize=(10, 5))
        plt.plot(scores, label="Original Similarity", marker="o", linestyle="--")
        plt.plot(new_score, label="Dynamic Score", marker="o")
        plt.title("Similarity Scores: Original vs Dynamically Boosted")
        plt.xlabel("Chunk Index")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            f"similarity {journey.title} original score vs. score after decay and boost.png"
        )
        plt.show()

    def _compute_dynamic_score(
        self,
        chunk_similarities: Sequence[float],
        alpha: float = 0.6,
        jump_boost: float = 1.5,
    ) -> Sequence[float]:
        scores = []
        prev_score = chunk_similarities[0]
        for t, v in enumerate(chunk_similarities):
            jump = 0.0
            if t > 0 and v > chunk_similarities[t - 1] + 0.1:  # detect jump
                jump = v - chunk_similarities[t - 1]
                prev_score = v + jump_boost * jump
            else:
                prev_score = alpha * v + (1 - alpha) * prev_score
            scores.append(prev_score)
        return scores

    async def get_queries_similarity(
        self,
        container: Container,
        queries: Sequence[str],
        journey: Journey,
    ) -> Sequence[float]:
        journey_doc = await journey_to_str(container, journey)

        tasks = [
            get_cosine_similarity(
                container=container,
                query=q,
                journey=journey_doc,
            )
            for q in queries
        ]

        results = await safe_gather(*tasks)

        new_scores = self._compute_dynamic_score(results)

        # self._plot_scores(journey, results, new_scores)

        return new_scores

    def get_similarity_score(
        self,
        chunk_similarities: Sequence[float],
    ) -> float:
        return chunk_similarities[-1]


def get_queries_for_each_message(
    interaction_history: Sequence[Event],
    context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]] = [],
    guidelines: Sequence[Guideline] = [],
    terms: Sequence[Term] = [],
    staged_events: Sequence[EmittedEvent] = [],
) -> Sequence[str]:
    queries: list[str] = []

    query = ""

    if context_variables:
        query += f"\n{context_variables_to_json(context_variables)}"

    if guidelines:
        query += str(
            [
                f"When {g.content.condition}, then {g.content.action}"
                if g.content.action
                else f"When {g.content.condition}"
                for g in guidelines
            ]
        )

    if staged_events:
        query += str([e.data for e in staged_events])

    if terms:
        query += str([t.name for t in terms])

    if query:
        queries.append(query)

    if interaction_history:
        for h in interaction_history:
            queries.append(str(h.data))

    return queries


def plot_journey_similarities(similarity: dict[str, Sequence[float]]) -> None:
    """
    Plots similarity scores per chunk for each journey.

    Args:
        similarity: A dictionary mapping journey titles to lists of similarity scores.
    """
    plt.figure(figsize=(12, 6))

    for title, scores in similarity.items():
        plt.plot(range(len(scores)), scores, label=title, marker="o")

    plt.title("Similarity per Chunk")
    plt.xlabel("Chunk Index")
    plt.ylabel("Similarity Score")
    plt.legend(title="Journey")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"similarity last message {[s for s in similarity.keys()]}.png")
    plt.show()


async def test_find_relevant_journeys_for_agent_returns_most_relevant(
    container: Container,
    agent: Agent,
) -> None:
    conversation_context: list[tuple[EventSource, str]] = [
        (
            EventSource.CUSTOMER,
            "Hi, I forgot my password. Can you help me reset it?",
        ),
        (
            EventSource.AI_AGENT,
            "Of course, I'd be happy to help. Can you please provide your account name?",
        ),
        (
            EventSource.CUSTOMER,
            "Yes, it's jenny_the_cat89",
        ),
        (
            EventSource.AI_AGENT,
            "Thanks! Now, could you share the email address or phone number associated with your account?",
        ),
        (
            EventSource.CUSTOMER,
            "Sure, it's jenny@example.com",
        ),
        (
            EventSource.AI_AGENT,
            "Great. I hope you're having a lovely day!",
        ),
        (
            EventSource.CUSTOMER,
            "Thanks, you too!",
        ),
        (
            EventSource.AI_AGENT,
            "Thank you! Resetting your password now...",
        ),
        (
            EventSource.AI_AGENT,
            "Your password has been successfully reset. Please check your email for further instructions.",
        ),
        (
            EventSource.CUSTOMER,
            "Thanks! Also, I'd like to change my credit limit.",
        ),
        (
            EventSource.AI_AGENT,
            "I'd be glad to help with that. Could you tell me what you'd like your new credit limit to be?",
        ),
        (
            EventSource.CUSTOMER,
            "I'd like to increase it to $5,000.",
        ),
        (
            EventSource.AI_AGENT,
            "Understood. Let me process your request to change your credit limit to $5,000.",
        ),
        (
            EventSource.AI_AGENT,
            "Your credit limit has been successfully updated to $5,000. Let me know if there's anything else I can assist you with!",
        ),
        (
            EventSource.CUSTOMER,
            "Actually yes, I need to book a flight.",
        ),
        (
            EventSource.AI_AGENT,
            "Sure! Can you please tell me your departure and destination airports?",
        ),
        (
            EventSource.CUSTOMER,
            "Flying from JFK to LAX.",
        ),
        (
            EventSource.AI_AGENT,
            "Got it. What date would you like to travel?",
        ),
        (
            EventSource.CUSTOMER,
            "July 18th.",
        ),
        (
            EventSource.AI_AGENT,
            "And would you prefer economy or business class?",
        ),
        (
            EventSource.CUSTOMER,
            "Business class, please.",
        ),
        (
            EventSource.AI_AGENT,
            "Perfect. Lastly, can I have the name of the traveler?",
        ),
        (
            EventSource.CUSTOMER,
            "Jennifer Morales.",
        ),
        (
            EventSource.AI_AGENT,
            "Thanks, Jennifer. I’ll go ahead and book your business class flight from JFK to LAX on July 18th.",
        ),
        (
            EventSource.AI_AGENT,
            "Your flight has been booked! A confirmation has been sent to your email.",
        ),
        (
            EventSource.CUSTOMER,
            "Hi, I forgot my password. Can you help me reset it?",
        ),
        (
            EventSource.AI_AGENT,
            "Of course, I'd be happy to help. Can you please provide your account name?",
        ),
        (
            EventSource.CUSTOMER,
            "Yes, it's jenny_the_cat89",
        ),
    ]
    interaction_history = [
        create_event_message(
            offset=i,
            source=source,
            message=message,
        )
        for i, (source, message) in enumerate(conversation_context)
    ]

    journey_names = [
        "Reset Password",
        "Change Credit Limits",
        "Book Taxi Ride",
        "Business Adult Only (detailed)",
        "Vegetarian Customers",
        "Place Food Order",
        "Book Flight",
        "Book Flight (simple)",
    ]
    journeys = await get_journeys_by_name(container, journey_names)

    # await get_journeys_similarity_for_prefix(
    #     container=container,
    #     score_calculator=PrefixSimilarityScore(),
    #     interaction_history=interaction_history,
    #     journeys=journeys,
    # )

    await get_journeys_similarity_for_prefix(
        container=container,
        score_calculator=JumpBoostSimilarityScore(),
        interaction_history=interaction_history,
        journeys=journeys,
        title="jump boost",
    )

    # queries = get_basic_query(interaction_history)
    # queries = get_queries_for_each_message(interaction_history)

    # await find_relevant_journey(container, queries, journeys)
    # results = await entity_queries.find_relevant_journeys_for_context(
    #     [reset_journey, change_limit_journey], query
    # )

    # assert len(results) == 1
    # assert results[0].id == reset_journey.id
