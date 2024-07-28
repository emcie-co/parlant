import json
from typing import Mapping, Optional, Sequence
from loguru import logger

from emcie.server.core.agents import Agent
from emcie.server.core.context_variables import ContextVariable, ContextVariableValue
from emcie.server.core.tools import Tool
from emcie.server.engines.alpha.guideline_proposition import GuidelineProposition
from emcie.server.engines.alpha.prompt_builder import BuiltInSection, PromptBuilder, SectionStatus
from emcie.server.engines.alpha.terminology import Term
from emcie.server.engines.alpha.utils import (
    make_llm_client,
)
from emcie.server.engines.common import ProducedEvent
from emcie.server.core.sessions import Event


class MessageEventProducer:
    def __init__(
        self,
    ) -> None:
        self._llm_client = make_llm_client("openai")

    async def produce_events(
        self,
        agents: Sequence[Agent],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_propositions: Sequence[GuidelineProposition],
        tool_enabled_guideline_propositions: Mapping[GuidelineProposition, Sequence[Tool]],
        staged_events: Sequence[ProducedEvent],
    ) -> Sequence[ProducedEvent]:
        assert len(agents) == 1

        if (
            not interaction_history
            and not ordinary_guideline_propositions
            and not tool_enabled_guideline_propositions
        ):
            # No interaction and no guidelines that could trigger
            # a proactive start of the interaction
            logger.debug("Skipping response; interaction is empty and there are no guidelines")
            return []

        prompt = self._format_prompt(
            agents=agents,
            context_variables=context_variables,
            interaction_history=interaction_history,
            terms=terms,
            ordinary_guideline_propositions=ordinary_guideline_propositions,
            tool_enabled_guideline_propositions=tool_enabled_guideline_propositions,
            staged_events=staged_events,
        )

        if response_message := await self._generate_response_message(prompt):
            logger.debug(f'Message production result: "{response_message}"')
            return [
                ProducedEvent(
                    source="server",
                    kind=Event.MESSAGE_KIND,
                    data={"message": response_message},
                )
            ]
        else:
            logger.debug("Skipping response; no response deemed necessary")

        return []

    def _format_prompt(
        self,
        agents: Sequence[Agent],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_propositions: Sequence[GuidelineProposition],
        tool_enabled_guideline_propositions: Mapping[GuidelineProposition, Sequence[Tool]],
        staged_events: Sequence[ProducedEvent],
    ) -> str:
        assert len(agents) == 1

        builder = PromptBuilder()

        builder.add_agent_identity(agents[0])
        builder.add_interaction_history(interaction_history)
        builder.add_context_variables(context_variables)
        builder.add_terminology(terms)
        builder.add_guideline_propositions(
            ordinary_guideline_propositions,
            tool_enabled_guideline_propositions,
        )

        builder.add_section(
            """
You must generate your response message to the current
(latest) state of the interaction.
"""
        )

        builder.add_staged_events(staged_events)

        if builder.section_status(BuiltInSection.GUIDELINE_PROPOSITIONS) != SectionStatus.ACTIVE:
            builder.add_section(
                """
Produce a valid JSON object in the following format: ###
{{
    "produced_response": true,
    "rationale": "<a few words to justify why you decided to respond in this way>",
    "revisions": [
        {
            "revision_number": <1 TO N>,
            "content": "<your message here>",
            "followed_all_rules": true
        }
    ]
}}
###
"""
            )
        else:
            builder.add_section(
                f"""
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

DO NOT PRODUCE MORE THAN 5 REVISIONS. IF YOU REACH the 5th REVISION, STOP THERE.

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
            "revision_number": 1,
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
            "rules_broken_due_to_missing_data": false,
            "rules_broken_due_to_prioritization": false
        }},
        ...,
        {{
            "revision_number": 2,
            "content": "final verified message content",
            "rules_followed": [
                "#1; correctly did...",
                "#2; correctly said...",
                "#3; correctly said...",
                "#5; correctly did..."
            ],
            "rules_broken": [],
            "followed_all_rules": true
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
            "revision_number": 1,
            "content": "I'd be happy to prepare your burger as soon as we restock the requested toppings.",
            "rules_followed": [
                "#2; upheld food quality and did not prepare the burger without the fresh toppings."
            ],
            "rules_broken": [
                "#1; did not provide the burger with requested toppings immediately due to the unavailability of fresh ingredients."
            ],
            "followed_all_rules": false,
            "rules_broken_due_to_prioritization": true,
            "prioritization_rationale": "Given the higher priority score of Rule 2, maintaining food quality standards before serving the burger is prioritized over immediate service.",
            "rules_broken_due_to_missing_data": false
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
            "revision_number": 1,
            "content": "I'm sorry, I am unable to provide this information at this time.",
            "rules_followed": [
            ],
            "rules_broken": [
                "#1; Lacking menu data in the context prevented providing the client with drink information."
            ],
            "followed_all_rules": false,
            "rules_broken_due_to_missing_data": true
            "missing_data_rationale": "Menu data was missing",
            "rules_broken_due_to_prioritization": false
        }}
    ]
}}
###
"""  # noqa
            )

        return builder.build()

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

        logger.debug(
            f'Message event producer response: {json.dumps(json_content["revisions"], indent=2)}'
        )

        final_revision = json_content["revisions"][-1]

        followed_all_rules = final_revision.get("followed_all_rules", False)
        rules_broken_due_to_prioritization = final_revision.get(
            "rules_broken_due_to_prioritization", False
        )

        if not followed_all_rules and not rules_broken_due_to_prioritization:
            logger.warning(f"PROBLEMATIC RESPONSE: {content}")

        return str(final_revision["content"])
