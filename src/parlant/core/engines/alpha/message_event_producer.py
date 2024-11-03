from itertools import chain
import json
import traceback
from typing import Mapping, Optional, Sequence

from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.agents import Agent
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.engines.alpha.guideline_proposition import GuidelineProposition
from parlant.core.engines.alpha.prompt_builder import (
    PromptBuilder,
)
from parlant.core.glossary import Term
from parlant.core.emissions import EmittedEvent, EventEmitter
from parlant.core.sessions import Event
from parlant.core.common import DefaultBaseModel
from parlant.core.logging import Logger
from parlant.core.tools import ToolId


class Revision(DefaultBaseModel):
    revision_number: int
    content: str
    guidelines_followed: Optional[list[str]] = []
    guidelines_broken: Optional[list[str]] = []
    followed_all_guidelines: Optional[bool] = False
    guidelines_broken_due_to_missing_data: Optional[bool] = False
    missing_data_rationale: Optional[str] = None
    guidelines_broken_only_due_to_prioritization: Optional[bool] = False
    prioritization_rationale: Optional[str] = None


class GuidelineEvaluation(DefaultBaseModel):
    number: int
    instruction: str
    evaluation: str
    adds_value: str
    data_available: str


class MessageGenerationError(Exception):
    def __init__(self, message: str = "Message generation failed") -> None:
        super().__init__(message)


class MessageEventSchema(DefaultBaseModel):
    last_message_of_user: str
    produced_reply: Optional[bool] = True
    rationale: str
    revisions: list[Revision]
    evaluations_for_each_of_the_provided_guidelines: Optional[list[GuidelineEvaluation]] = None


class MessageEventProducer:
    def __init__(
        self,
        logger: Logger,
        correlator: ContextualCorrelator,
        schematic_generator: SchematicGenerator[MessageEventSchema],
    ) -> None:
        self._logger = logger
        self._correlator = correlator
        self._schematic_generator = schematic_generator

    async def produce_events(
        self,
        event_emitter: EventEmitter,
        agents: Sequence[Agent],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_propositions: Sequence[GuidelineProposition],
        tool_enabled_guideline_propositions: Mapping[GuidelineProposition, Sequence[ToolId]],
        staged_events: Sequence[EmittedEvent],
    ) -> Sequence[EmittedEvent]:
        assert len(agents) == 1

        with self._logger.operation("Message production"):
            if (
                not interaction_history
                and not ordinary_guideline_propositions
                and not tool_enabled_guideline_propositions
            ):
                # No interaction and no guidelines that could trigger
                # a proactive start of the interaction
                self._logger.debug(
                    "Skipping response; interaction is empty and there are no guidelines"
                )
                return []

            self._logger.debug(
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

            self._logger.debug(f"Message production prompt:\n{prompt}")

            last_known_event_offset = interaction_history[-1].offset if interaction_history else -1

            await event_emitter.emit_status_event(
                correlation_id=self._correlator.correlation_id,
                data={
                    "acknowledged_offset": last_known_event_offset,
                    "status": "typing",
                    "data": {},
                },
            )

            generation_attempt_temperatures = {
                0: 0.5,
                1: 1,
                2: 0.1,
            }

            last_generation_exception: Exception | None = None

            for generation_attempt in range(3):
                try:
                    if response_message := await self._generate_response_message(
                        prompt,
                        temperature=generation_attempt_temperatures[generation_attempt],
                    ):
                        self._logger.debug(f'Message production result: "{response_message}"')

                        event = await event_emitter.emit_message_event(
                            correlation_id=self._correlator.correlation_id,
                            data=response_message,
                        )

                        return [event]
                    else:
                        self._logger.debug("Skipping response; no response deemed necessary")
                        return []
                except Exception as exc:
                    self._logger.warning(
                        f"Generation attempt {generation_attempt} failed: {traceback.format_exception(exc)}"
                    )
                    last_generation_exception = exc

            raise MessageGenerationError() from last_generation_exception

    def _format_prompt(
        self,
        agents: Sequence[Agent],
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        ordinary_guideline_propositions: Sequence[GuidelineProposition],
        tool_enabled_guideline_propositions: Mapping[GuidelineProposition, Sequence[ToolId]],
        staged_events: Sequence[EmittedEvent],
    ) -> str:
        assert len(agents) == 1
        builder = PromptBuilder()

        builder.add_section(
            """
You are an AI agent who is interacting with a user. The current state of this interaction will be provided to you later in this message.
You must generate your reply message to the current (latest) state of the interaction.
"""
        )

        builder.add_agent_identity(agents[0])
        builder.add_section(
            """
Task Description:
Continue the provided interaction in a natural and human-like manner. Your task is to produce a response to the latest state of the interaction.
Always do the following:
1. GENERAL BEHAVIOR: Make your response as human-like as possible. Be concise and avoid being overly polite when not necessary. 
2. AVOID REPEATING YOURSELF: When replying— try to avoid repeating yourself. Instead, refer the user to your previous answer, or choose a new approach altogether. If a conversation is looping, point that out to the user instead of maintaining the loop.
3. DO NOT HALLUCINATE: Do not state factual information that you do not know or are not sure about. If the user requests information you're unsure about, state that this information is not available to you. 
"""
        )
        if any([event.kind == "message" for event in staged_events]):
            builder.add_section(
                """
The interaction with the user has just began, and no messages were sent by either party.
If told so by a guideline or some other contextual condition, send the first message. Otherwise, do not produce a reply.
If you decide not to emit a message, output the following:
{{
    “last_message_of_user”: None,
    "produced_reply": false,
    "rationale": "<a few words to justify why a reply was NOT produced here>",
    "revisions": []
}}
Otherwise, follow the rest of this prompt to choose the content of your response. 
        """
            )

        else:
            builder.add_section("""
Since the interaction with the user is already ongoing, always produce a reply to the user's last message. The only exception where you may not produce a reply is if the user explicitly asked you not to respond to their message.
In all other cases, even if the user is indicating that the conversation is over, you are expected to produce a reply.
                """)

        builder.add_section(
            f"""
Propose incremental revisions to your reply, ensuring that your proposals adhere
to each and every one of the provided guidelines based on the most recent state of interaction.

Mind the priority scores assigned to each guideline, acknowledging that in some cases,
adherence to a higher-priority guideline may necessitate deviation from another.
If a given guideline contradicts a previous request made by the user, or if it's absolutely inappropriate given the state of the conversation, ignore the guideline while specifying why you broke it. 
Use your best judgement in applying prioritization.
Note too that it is permissible for the final revision to break rules IF AND ONLY IF
all of the broken rules were broken due to conscious prioritization of guidelines,
due to either (1) conflicting with another guideline, (2) contradicting a user's request or (3) lack of necessary context / data.
If you do not fulfill a guideline, you must clearly justify your reasoning for doing so in your reply.

Continuously critique each revision to refine the reply.
Ensure each critique is unique to prevent redundancy in the revision process.

Your final output should be a JSON object documenting the entire message development process.
This document should detail how each guideline was adhered to,
instances where one guideline was prioritized over another,
situations where guidelines could not be followed due to lack of context or data,
and the rationale for each decision made during the revision process. The exact format of this output will be provided to you at the end of this prompt.
IMPORTANT: Unless there is some conflict between the guidelines provided,
prefer to adhere to the provided guidelines over your own judgement.

DO NOT PRODUCE MORE THAN 5 REVISIONS. IF YOU REACH the 5th REVISION, STOP THERE.

Examine the following examples to understand your expected behavior:

###

Example 1: A reply that took critique in a few revisions to get right: ###
{{
    “last_message_of_user”: “<the user’s last message in the interaction>”,
    "rationale": "<a few words to justify why you decided to respond to the user at all>",
    "evaluations_for_each_of_the_provided_guidelines": [
        {{
            "number": 1,
            "instruction": "Do this [...]",
            "evaluation": "in this situation, I am instructed to do [...]",
            "adds_value": "I didn't do it yet, so I should do it now",
            "data_available": "no particular data is needed for this"
        }},
        {{
            "number": 2,
            "instruction": "Say this [...]",
            "evaluation": "in this situation, I am instructed to say [...]",
            "adds_value": "I didn't say it yet, so I should say it now",
            "data_available": "no particular data is needed for this"
        }},
        {{
            "number": 3,
            "instruction": "Say that [...]",
            "evaluation": "in this situation, I am instructed to say [...]",
            "adds_value": "I didn't say it yet, so I should say it now",
            "data_available": "no particular data is needed for this"
        }},
        {{
            "number": 4,
            "instruction": "Do that [...]",
            "evaluation": "in this situation, I am instructed to do [...]",
            "adds_value": "I didn't do it yet, so I should do it now",
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
                "#2; didn't say...",
                "#4; didn't do..."
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

Example 2: A reply where one guideline was prioritized over another: ###
{{
    “last_message_of_user”: “<the user’s last message in the interaction>”,
    "rationale": "<a few words to justify why you decided to respond to the user at all>",
    "evaluations_for_each_of_the_provided_guidelines": [
        {{
            "number": 1,
            "instruction": "When the user chooses and orders a burger, then provide it",
            "evaluation": "The user asked for a burger with cheese, so I need to provide it to him.",
            "adds_value": "I didn't provide the burger yet, so I should do so now.",
            "data_available": "The burger choice is available in the interaction"
        }},
        {{
            "number": 2,
            "instruction": "When the user chooses specific ingredients on the burger, only provide those ingredients if we have them fresh in stock; otherwise, reject the order."
            "evaluation": "The user chose cheese on the burger, but all of the cheese we currently have is expired",
            "adds_value": "I must reject the order, otherwise the user might eat bad cheese",
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


Example 3: Non-Adherence Due to Missing Data: ###
{{
    “last_message_of_user”: “<the user’s last message in the interaction>”,
    "rationale": "<a few words to justify why you decided to respond to the user at all>",
    "evaluations_for_each_of_the_provided_guidelines": [
        {{
            "number": 1,
            "instruction": "When the user asks for a drink, check the menu and offer what's on it"
            "evaluation": "The user did ask for a drink, so I should check the menu to see what's available.",
            "adds_value": "The user doesn't know what drinks we have yet, so I should tell him.",
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
            "guidelines_broken_due_to_missing_data": true,
            "guidelines_broken_only_due_to_prioritization": false
        }}
    ]
}}
###
"""  # noqa
        )
        builder.add_interaction_history(interaction_history)
        builder.add_context_variables(context_variables)
        builder.add_glossary(terms)
        builder.add_guideline_propositions(
            ordinary_guideline_propositions,
            tool_enabled_guideline_propositions,
        )
        #        builder.add_section(
        #            """
        # If a given guideline contradicts a previous request made by the user, or if it's absolutely inappropriate given the state of the conversation, ignore the guideline while specifying why you broke it in your response.
        #        """
        #        )  TODO delete if not needed
        builder.add_staged_events(staged_events)
        builder.add_section(
            f"""
Produce a valid JSON object in the following format: ###

{self._get_output_format(interaction_history, list(chain(ordinary_guideline_propositions, tool_enabled_guideline_propositions)))}"""
        )

        prompt = builder.build()
        with open("message event prompt.txt", "w") as f:  # TODO delete
            f.write(prompt)

        return prompt

    def _get_output_format(
        self, interaction_history: Sequence[Event], guidelines: Sequence[GuidelineProposition]
    ) -> str:
        last_user_message = next(
            (
                event.data["message"]
                for event in reversed(interaction_history)
                if (
                    event.kind == "message"
                    and event.source == "end_user"
                    and isinstance(event.data, dict)
                )
            ),
            "",
        )
        guidelines_output_format = "\n".join(
            [
                f"""
                {{
                    "number": {i},
                    "instruction": "{g.guideline.content.action}"
                    "evaluation": "<your evaluation of the guideline to the present state of the interaction>",
                    "adds_value": "<your assessment if and to what extent following this guideline now would add value>",
                    "data_available": "<explanation whether you are provided with the required data to follow this guideline now>"
                }},"""
                for i, g in enumerate(guidelines, start=1)
            ]
        )

        return f"""
        {{
            “last_message_of_user”: “{last_user_message}”,
            "rationale": "<a few words to explain why you should or shouldn't produce a reply to the user in this case>",
            "produced_reply": <BOOL>,
            "evaluations_for_each_of_the_provided_guidelines": [
{guidelines_output_format}
            ],
            "revisions": [
            {{
                "revision_number": 1,
                "content": <response chosen after revision 1>,
                "guidelines_followed": <list of guidelines that were followed>,
                "guidelines_broken": <list of guidelines that were broken>,
                "followed_all_guidelines": <BOOL>
            }},
            ...
            ]
        }}
        ###"""

    async def _generate_response_message(
        self,
        prompt: str,
        temperature: float,
    ) -> Optional[str]:
        message_event_response = await self._schematic_generator.generate(
            prompt=prompt,
            hints={"temperature": temperature},
        )

        if not message_event_response.content.produced_reply:
            self._logger.debug(f"MessageEventProducer produced no reply: {message_event_response}")
            return None

        if message_event_response.content.evaluations_for_each_of_the_provided_guidelines:
            self._logger.debug(
                "MessageEventProducer guideline evaluations: "
                f"{json.dumps([e.model_dump(mode="json") for e in message_event_response.content.evaluations_for_each_of_the_provided_guidelines], indent=2)}"
            )

        self._logger.debug(
            "MessageEventProducer revisions: "
            f"{json.dumps([r.model_dump(mode="json") for r in message_event_response.content.revisions], indent=2)}"
        )

        if first_correct_revision := next(
            (
                r
                for r in message_event_response.content.revisions
                if r.guidelines_broken_only_due_to_prioritization
                or r.guidelines_broken_due_to_missing_data
            ),
            "",
        ):
            # Sometimes the LLM continues generating revisions even after
            # it generated a correct one. Those next revisions tend to be
            # faulty, as they do not handle prioritization well.
            # This is a workaround.
            final_revision = first_correct_revision
        else:
            final_revision = message_event_response.content.revisions[-1]

        if (
            not final_revision.followed_all_guidelines
            and not final_revision.guidelines_broken_only_due_to_prioritization
        ):
            self._logger.warning(f"PROBLEMATIC RESPONSE: {final_revision.content}")

        with open("message event response.txt", "w") as f:  # TODO delete
            f.write(str(final_revision.content))
        return str(final_revision.content)
