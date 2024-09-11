from itertools import chain
import json
from typing import Mapping, Optional, Sequence

from emcie.common.tools import Tool
from emcie.server.contextual_correlator import ContextualCorrelator
from emcie.server.core.agents import Agent
from emcie.server.core.context_variables import ContextVariable, ContextVariableValue
from emcie.server.engines.alpha.guideline_proposition import GuidelineProposition
from emcie.server.engines.alpha.message_event import MessageEventSchema
from emcie.server.engines.alpha.prompt_builder import BuiltInSection, PromptBuilder, SectionStatus
from emcie.server.core.terminology import Term
from emcie.server.engines.event_emitter import EmittedEvent
from emcie.server.core.sessions import Event
from emcie.server.llm.json_generators import JSONGenerator
from emcie.server.logger import Logger


class MessageEventProducer:
    def __init__(
        self,
        logger: Logger,
        correlator: ContextualCorrelator,
        message_event_generator: JSONGenerator[MessageEventSchema],
    ) -> None:
        self.logger = logger
        self.correlator = correlator
        self._message_event_generator = message_event_generator

    async def produce_events(
        self,
        agents: Sequence[Agent],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_propositions: Sequence[GuidelineProposition],
        tool_enabled_guideline_propositions: Mapping[GuidelineProposition, Sequence[Tool]],
        staged_events: Sequence[EmittedEvent],
    ) -> Sequence[EmittedEvent]:
        assert len(agents) == 1

        with self.logger.operation("Message production"):
            if (
                not interaction_history
                and not ordinary_guideline_propositions
                and not tool_enabled_guideline_propositions
            ):
                # No interaction and no guidelines that could trigger
                # a proactive start of the interaction
                self.logger.debug(
                    "Skipping response; interaction is empty and there are no guidelines"
                )
                return []

            self.logger.debug(
                f'Guidelines applied: {json.dumps([{
                    "predicate": p.guideline.content.predicate,
                    "action": p.guideline.content.action,
                    "rationale": p.rationale,
                    "score": p.score}
                for p in  chain(ordinary_guideline_propositions, tool_enabled_guideline_propositions.keys())], indent=2)}'
            )

            prompt = self._format_prompt(
                agents=agents,
                context_variables=context_variables,
                interaction_history=interaction_history,
                terms=terms,
                ordinary_guideline_propositions=ordinary_guideline_propositions,
                tool_enabled_guideline_propositions=tool_enabled_guideline_propositions,
                staged_events=staged_events,
            )

            self.logger.debug(f"Message generation prompt: \n{prompt}")

            if response_message := await self._generate_response_message(prompt):
                self.logger.debug(f'Message production result: "{response_message}"')
                return [
                    EmittedEvent(
                        source="server",
                        kind="message",
                        correlation_id=self.correlator.correlation_id,
                        data={"message": response_message},
                    )
                ]
            else:
                self.logger.debug("Skipping response; no response deemed necessary")

            return []

    def _format_prompt(
        self,
        agents: Sequence[Agent],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_propositions: Sequence[GuidelineProposition],
        tool_enabled_guideline_propositions: Mapping[GuidelineProposition, Sequence[Tool]],
        staged_events: Sequence[EmittedEvent],
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
You must generate your reply message to the current (latest) state of the interaction.
IMPORTANT: Strive to continue the interaction/conversation in the most natural way for a normal human conversation.
"""
        )

        builder.add_staged_events(staged_events)

        if builder.section_status(BuiltInSection.GUIDELINE_PROPOSITIONS) != SectionStatus.ACTIVE:
            builder.add_section(
                """
Produce a valid JSON object in the following format: ###
{{
    “last_message_of_user”: “<the user’s last message in the interaction>”,
    "produced_reply": true,
    "evaluations_for_each_of_the_provided_guidelines": [
        {
            "number": <GUIDELINE_NUMBER>,
            "instruction": <THE_INSTRUCTION_GIVEN_BY_THE_GUIDELINE>,
            "evaluation": <EXPLANATION_TO_WHAT_EXTENT_THIS_GUIDELINE_IS_RELEVANT_TO_YOUR_REPLY_NOW>,
            "adds_value": <EXPLANATION_OF_WHETHER_FOLLOWING_THIS_GUIDELINE_IN_THE_NEXT_MESSAGE_ADDS_NEW_VALUE_TO_THE_INTERACTION_THAT_WAS_NOT_ALREADY_COMMUNICATED>,
            "data_available": <EXPLANATION_OF_WHETHER_THE_REQUIRED_DATA_IS_AVAILABLE_OR_MISSING_IN_THE_PROVIDED_CONTEXT>
        },
        ...
    ],
    "rationale": "<a few words to justify why you decided to respond in this way>",
    "revisions": [
        {
            "revision_number": <1 TO N>,
            "content": "<your message here>",
            "followed_all_guidelines": true
        }
    ]
}}
###
"""
            )
        else:
            builder.add_section(
                f"""
Propose incremental revisions to your reply, ensuring that your proposals adhere
to each and every one of the provided guidelines based on the most recent state of interaction.

Mind the priority scores assigned to each guideline, acknowledging that in some cases,
adherence to a higher-priority guideline may necessitate deviation from another.
Use your best judgement in applying prioritization.
Note too that it is permissible for the final revision to break rules IF AND ONLY IF
all of the broken rules were broken due to conscious prioritization of guidelines,
and not due to overlooking or missing a detail in one of the guidelines.

Additionally, recognize that if a guideline cannot be adhered to due to lack of necessary
context or data, this must be clearly justified in your reply.

Continuously critique each revision to refine the reply.
Ensure each critique is unique to prevent redundancy in the revision process.

Your final output should be a JSON object documenting the entire message development process.
This document should detail how each guideline was adhered to,
instances where one guideline was prioritized over another,
situations where guidelines could not be followed due to lack of context or data,
and the rationale for each decision made during the revision process.

DO NOT PRODUCE MORE THAN 5 REVISIONS. IF YOU REACH the 5th REVISION, STOP THERE.

Produce a valid JSON object in the format according to the following examples.

Example 1: When no reply was deemed appropriate: ###
{{
    “last_message_of_user”: “<the user’s last message in the interaction>”,
    "produced_reply": false,
    "rationale": "<a few words to justify why a reply was NOT produced here>",
    "revisions": []
}}
###

Example 2: A reply that took critique in a few revisions to get right: ###
{{
    “last_message_of_user”: “<the user’s last message in the interaction>”,
    "produced_reply": true,
    "rationale": "<a few words to justify why you decided to respond to the user at all>",
    "evaluations_for_each_of_the_provided_guidelines": [
        {{
            "number": 1,
            "instruction": "Do this [...]",
            "evaluation": "in this situation, I am instructed to do [...]",
            "adds_value": "I didn't do it yet, so I should do it now"
            "data_available": "no particular data is needed for this"
        }},
        {{
            "number": 2,
            "instruction": "Say this [...]",
            "evaluation": "in this situation, I am instructed to say [...]",
            "adds_value": "I didn't say it yet, so I should say it now"
            "data_available": "no particular data is needed for this"
        }},
        {{
            "number": 3,
            "instruction": "Say that [...]",
            "evaluation": "in this situation, I am instructed to say [...]",
            "adds_value": "I didn't say it yet, so I should say it now"
            "data_available": "no particular data is needed for this"
        }},
        {{
            "number": 4,
            "instruction": "Do that [...]",
            "evaluation": "in this situation, I am instructed to do [...]",
            "adds_value": "I didn't do it yet, so I should do it now"
            "data_available": "no particular data is needed for this"
        }}
    ],
    "revisions": [
        {{
            "revision_number": 1,
            "content": "some proposed message content",
            "guidelines_followed": [
                "#1; correctly did...",
                "#3; correctly said..."
            ],
            "guidelines_broken": [
                "#2; didn't say..."
                "#4; didn't do...",
            ],
            "followed_all_guidelines": false,
            "guidelines_broken_due_to_missing_data": false,
            "guidelines_broken_only_due_to_prioritization": false
        }},
        ...,
        {{
            "revision_number": 2,
            "content": "final verified message content",
            "guidelines_followed": [
                "#1; correctly did...",
                "#2; correctly said...",
                "#3; correctly said...",
                "#5; correctly did..."
            ],
            "guidelines_broken": [],
            "followed_all_guidelines": true
        }},
    ]
}}

###

Example 3: A reply where one guideline was prioritized over another: ###
{{
    “last_message_of_user”: “<the user’s last message in the interaction>”,
    "produced_reply": true,
    "rationale": "<a few words to justify why you decided to respond to the user at all>",
    "evaluations_for_each_of_the_provided_guidelines": [
        {{
            "number": 1,
            "instruction": "When the user chooses and orders a burger, then provide it"
            "evaluation": "The user asked for a burger with cheese, so I need to provide it to him.",
            "adds_value": "I didn't provide the burger yet, so I should do so now."
            "data_available": "The burger choice is available in the interaction"
        }},
        {{
            "number": 2,
            "instruction": "When the user chooses specific ingredients on the burger, only provide those ingredients if we have them fresh in stock; otherwise, reject the order."
            "evaluation": "The user chose cheese on the burger, but all of the cheese we currently have is expired",
            "adds_value": "I must reject the order, otherwise the user might eat bad cheese"
            "data_available": "The relevant stock availability is given in the tool calls' data"
        }}
    ],
    "revisions": [
        {{
            "revision_number": 1,
            "content": "I'd be happy to prepare your burger as soon as we restock the requested toppings.",
            "guidelines_followed": [
                "#2; upheld food quality and did not go on to preparing the burger without fresh toppings."
            ],
            "guidelines_broken": [
                "#1; did not provide the burger with requested toppings immediately due to the unavailability of fresh ingredients."
            ],
            "followed_all_guidelines": false,
            "guidelines_broken_only_due_to_prioritization": true,
            "prioritization_rationale": "Given the higher priority score of guideline 2, maintaining food quality standards before serving the burger is prioritized over immediate service.",
            "guidelines_broken_due_to_missing_data": false
        }}
    ]
}}
###


Example 4: Non-Adherence Due to Missing Data: ###
{{
    “last_message_of_user”: “<the user’s last message in the interaction>”,
    "produced_reply": true,
    "rationale": "<a few words to justify why you decided to respond to the user at all>",
    "evaluations_for_each_of_the_provided_guidelines": [
        {{
            "number": 1,
            "instruction": "When the user asks for a drink, check the menu and offer what's on it"
            "evaluation": "The user did ask for a drink, so I should check the menu to see what's available.",
            "adds_value": "The user doesn't know what drinks we have yet, so I should tell him."
            "data_available": "No, I don't have the menu info in the interaction or tool calls"
        }}
    ],
    "revisions": [
        {{
            "revision_number": 1,
            "content": "I'm sorry, I am unable to provide this information at this time.",
            "guidelines_followed": [
            ],
            "guidelines_broken": [
                "#1; Lacking menu data in the context prevented me from providing the client with drink information."
            ],
            "followed_all_guidelines": false,
            "missing_data_rationale": "Menu data was missing",
            "guidelines_broken_due_to_missing_data": true
            "guidelines_broken_only_due_to_prioritization": false
        }}
    ]
}}
###
"""  # noqa
            )

        return builder.build()

    async def _generate_response_message(self, prompt: str) -> Optional[str]:
        message_event_response = await self._message_event_generator.generate(
            prompt=prompt,
            args={"temperature": 0.5},
        )

        if not message_event_response.content.produced_reply:
            return None

        self.logger.debug(
            f"Message event producer response: {json.dumps([r.model_dump() for r in message_event_response.content.revisions], indent=2)}"
        )

        if last_correct_revision := next(
            (
                r
                for r in reversed(message_event_response.content.revisions)
                if r.guidelines_broken_only_due_to_prioritization
            ),
            None,
        ):
            # Sometimes the LLM continues generating revisions even after
            # it generated a correct one. Those next revisions tend to be
            # faulty, as they do not handle prioritization well.
            # This is a workaround.
            final_revision = last_correct_revision
        else:
            final_revision = message_event_response.content.revisions[-1]

        if (
            not final_revision.followed_all_rules
            and not final_revision.rules_broken_due_to_prioritization
        ):
            self.logger.warning(f"PROBLEMATIC RESPONSE: {final_revision.content}")

        return str(final_revision.content)
