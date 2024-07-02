from itertools import chain
import json
from typing import Iterable, Optional
from loguru import logger

from emcie.server.core.context_variables import ContextVariable, ContextVariableValue
from emcie.server.core.tools import Tool
from emcie.server.engines.alpha.guideline_filter import GuidelineProposition
from emcie.server.engines.alpha.tool_caller import ToolCaller, produced_tools_events_to_dict
from emcie.server.engines.alpha.utils import (
    context_variables_to_json,
    events_to_json,
    make_llm_client,
)
from emcie.server.engines.common import ProducedEvent
from emcie.server.core.guidelines import Guideline
from emcie.server.core.sessions import Event


class EventProducer:

    def __init__(self) -> None:
        self.tool_event_producer = ToolEventProducer()
        self.message_event_producer = MessageEventProducer()

    async def produce_events(
        self,
        context_variables: Iterable[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Iterable[Event],
        ordinary_guideline_propositions: Iterable[GuidelineProposition],
        tool_enabled_guideline_propositions: dict[GuidelineProposition, Iterable[Tool]],
    ) -> Iterable[ProducedEvent]:
        interaction_event_list = list(interaction_history)
        context_variable_list = list(context_variables)

        tool_events = await self.tool_event_producer.produce_events(
            context_variables=context_variable_list,
            interaction_history=interaction_event_list,
            ordinary_guidelines=[p.guideline for p in ordinary_guideline_propositions],
            tool_enabled_guidelines={
                p.guideline: tools for p, tools in tool_enabled_guideline_propositions.items()
            },
        )

        message_events = await self.message_event_producer.produce_events(
            context_variables=context_variable_list,
            interaction_history=interaction_event_list,
            ordinary_guideline_propositions=ordinary_guideline_propositions,
            tool_enabled_guidelines=tool_enabled_guideline_propositions,
            staged_events=tool_events,
        )

        return chain(tool_events, message_events)


class MessageEventProducer:
    def __init__(
        self,
    ) -> None:
        self._llm_client = make_llm_client("openai")

    async def produce_events(
        self,
        context_variables: list[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: list[Event],
        ordinary_guideline_propositions: Iterable[GuidelineProposition],
        tool_enabled_guidelines: dict[GuidelineProposition, Iterable[Tool]],
        staged_events: Iterable[ProducedEvent],
    ) -> Iterable[ProducedEvent]:
        interaction_event_list = list(interaction_history)

        if (
            not interaction_event_list
            and not ordinary_guideline_propositions
            and not tool_enabled_guidelines
        ):
            # No interaction and no guidelines that could trigger
            # a proactive start of the interaction
            return []

        prompt = self._format_prompt(
            context_variables=context_variables,
            interaction_history=interaction_history,
            guideline_propositions=ordinary_guideline_propositions,
            tool_enabled_guidelines=tool_enabled_guidelines,
            staged_events=staged_events,
        )

        if response_message := await self._generate_response_message(prompt):
            return [
                ProducedEvent(
                    source="server",
                    type=Event.MESSAGE_TYPE,
                    data={"message": response_message},
                )
            ]

        return []

    def _format_prompt(
        self,
        context_variables: list[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: list[Event],
        guideline_propositions: Iterable[GuidelineProposition],
        tool_enabled_guidelines: dict[GuidelineProposition, Iterable[Tool]],
        staged_events: Iterable[ProducedEvent],
    ) -> str:
        interaction_events_json = events_to_json(interaction_history)
        context_values = context_variables_to_json(context_variables)
        staged_events_as_dict = produced_tools_events_to_dict(staged_events)
        all_guideline_propositions = chain(guideline_propositions, tool_enabled_guidelines)

        rules = "\n".join(
            f"{i}) When {p.guideline.predicate}, then {p.guideline.content}"
            f"\n\t Priority (1-10): {p.score}, rationale: {p.rationale}"
            for i, p in enumerate(all_guideline_propositions, start=1)
        )

        prompt = ""

        if interaction_history:
            prompt += f"""
The following is a list of events describing a back-and-forth 
interaction between you, an AI assistant, and a user: ###
{interaction_events_json}
###
"""

        else:
            prompt += """
You, an AI assistant, are currently engaged at the start of an online session with a user.
The interaction has yet to be initiated by either party.

- Decision Criteria for Initiating Interaction:
A. If the rules below both apply to the context, as well as suggest that you should say something
to the user, then you should indeed initiate the interaction now.
B. Otherwise, if no reason is provided that suggests you should say something to the user,
then you should not initiate the interaction. Produce no response in this case.
###
"""
            prompt += """\
You, an AI assistant, are now present in an online session with a user.
An interaction may or may not now be initiated by you, addressing the user.

Here's how to decide whether to initiate the interaction:
A. If the rules below both apply to the context, as well as suggest that you should say something
to the user, then you should indeed initiate the interaction now.
B. Otherwise, if no reason is provided that suggests you should say something to the user,
then you should not initiate the interaction. Produce no response in this case.
"""
        if context_variables:
            prompt += f"""
The following is information that you're given about the user and context of the interaction: ###
{context_values}
###
"""

        if rules:
            prompt += f"""
In formulating your response, you are required to follow these rules,
which are applicable to the latest state of the interaction. 
Each rule is accompanied by a priority score indicating its significance, 
and a rationale explaining why it is applicable: ###
{rules}
###
"""
        prompt += """
You must generate your response message to the current
(latest) state of the interaction.
"""

        if staged_events_as_dict:
            prompt += f"""
For your information, here are some staged events that have just been produced,
to assist you with generating your response message while following the rules above: ###
{staged_events_as_dict}
###
"""
        prompt += f"""
Propose revisions to the message content, 
ensuring that your proposals adhere to each and every one of the provided rules based on the most recent state of interaction. 
Consider the priority scores assigned to each rule, acknowledging that in some cases, adherence to a higher-priority rule may necessitate deviation from another. 
Additionally, recognize that if a rule cannot be adhered to due to lack of necessary context or data, this must be clearly justified in your response.

Continuously critique each revision to refine the response. 
Ensure each critique is unique to prevent redundancy in the revision process.

Your final output should be a JSON object documenting the entire message development process. 
This document should detail how each rule was adhered to, 
instances where one rule was prioritized over another, 
situations where rules could not be followed due to lack of context or data, 
and the rationale for each decision made during the revision process.

Produce a valid JSON object in the format according to the following examples.

Example 1: When no response was deemed appropriate: ###
{{
    "produced_response": false,
    "rationale": "a few words to justify why a response was NOT produced here",
    "revisions": []
}}
###

Example 2: A response that took critique in a few revisions to get right: ###
{{
    "produced_response": true,
    "rationale": "a few words to justify why a response was produced here",
    "revisions": [
        {{
            "content": "some proposed message content",
            "rules_followed": [
                "#1; correctly did...",
                "#3; correctly said..."
            ],
            "rules_broken": [
                "#5; didn't do...",
                "#2; didn't say..."
            ],
            "followed_all_rules": false,
            "prioritized_some_rules_over_others": false,
            "rules_broken_due_to_prioritization": false
        }},
        ...,
        {{
            "content": "final verified message content",
            "rules_followed": [
                "#1; correctly did...",
                "#2; correctly said...",
                "#3; correctly said...",
                "#5; correctly did..."
            ],
            "rules_broken": [],
            "followed_all_rules": true,
            "prioritized_some_rules_over_others": false,
            "rules_broken_due_to_prioritization": false
        }},
    ]
}}

###

Example 3: A response where one rule was prioritized over another: ###
{{
    "produced_response": true,
    "rationale": "Ensuring food quality is paramount, thus it overrides the immediate provision of a burger with requested toppings.",
    "revisions": [
        {{
            "content": "I'd be happy to prepare your burger as soon as we restock the requested toppings.",
            "rules_followed": [
                "#2; upheld food quality and did not prepare the burger without the fresh toppings."
            ],
            "rules_broken": [
                "#1; did not provide the burger with requested toppings immediately due to the unavailability of fresh ingredients."
            ],
            "followed_all_rules": false,
            "prioritized_some_rules_over_others": true,
            "prioritization_rationale": "Given the higher priority score of Rule 2, maintaining food quality standards before serving the burger is prioritized over immediate service.",
            "rules_broken_due_to_prioritization": false
        }}
    ]
}}
###


Example 4: Non-Adherence Due to Missing Data: ###
{{
    "produced_response": true,
    "rationale": "No data of drinks menu is available, therefore informing the customer that we don't have this information at this time.",
    "revisions": [
        {{
            "content": "I'm sorry, I am unable to provide this information at this time.",
            "rules_followed": [
            ],
            "rules_broken": [
                "#1; Lacking menu data in the context prevented providing the client with drink information."
            ],
            "followed_all_rules": false,
            "rules_broken_due_to_prioritization": true
            "missing_data_details": "Menu data was missing",
            "prioritized_some_rules_over_others": false,
        }}
    ]
}}
###

"""  # noqa

        return prompt

    async def _generate_response_message(self, prompt: str) -> Optional[str]:
        response = await self._llm_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o",
            response_format={"type": "json_object"},
            temperature=0.5,
        )

        content = response.choices[0].message.content or ""

        json_content = json.loads(content)

        if not json_content["produced_response"]:
            return None

        final_revision = json_content["revisions"][-1]

        if not final_revision["followed_all_rules"]:
            logger.warning(f"PROBLEMATIC RESPONSE: {content}")

        return str(final_revision["content"])


class ToolEventProducer:
    def __init__(
        self,
    ) -> None:
        self._llm_client = make_llm_client("openai")
        self.tool_caller = ToolCaller()

    async def produce_events(
        self,
        context_variables: list[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: list[Event],
        ordinary_guidelines: Iterable[Guideline],
        tool_enabled_guidelines: dict[Guideline, Iterable[Tool]],
    ) -> Iterable[ProducedEvent]:
        if not tool_enabled_guidelines:
            return []

        produced_tool_events: list[ProducedEvent] = []

        tool_calls = await self.tool_caller.infer_tool_calls(
            context_variables,
            interaction_history,
            ordinary_guidelines,
            tool_enabled_guidelines,
            produced_tool_events,
        )

        tools = chain(*tool_enabled_guidelines.values())

        tool_results = await self.tool_caller.execute_tool_calls(
            tool_calls,
            tools,
        )

        if not tool_results:
            return []

        produced_tool_events.append(
            ProducedEvent(
                source="server",
                type=Event.TOOL_TYPE,
                data={"tools_result": tool_results},
            )
        )

        return produced_tool_events
