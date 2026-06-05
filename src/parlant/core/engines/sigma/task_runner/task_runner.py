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

from dataclasses import dataclass
from itertools import chain

from parlant.core.agents import CompositionMode, Effort, MessageOutputMode
from parlant.core.engines.alpha.guideline_matching.generic.common import internal_representation
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.engines.sigma.response_state import EngineContext
from parlant.core.engines.sigma.loop.loop import LoopJob
from parlant.core.engines.sigma.loop.streaming_loop import StreamingLoop
from parlant.core.loggers import Logger
from parlant.core.meter import Meter
from parlant.core.nlp.common import ModelSize
from parlant.core.nlp.react import ReasoningConfig, Usage
from parlant.core.tracer import Tracer


@dataclass(frozen=True)
class Task:
    context: EngineContext
    instructions: str | None = None
    model_size: ModelSize | None = None
    reasoning_config: ReasoningConfig | None = None


@dataclass(frozen=True)
class TaskResult:
    output: str
    usage: Usage


class TaskRunner:
    def __init__(
        self,
        logger: Logger,
        tracer: Tracer,
        meter: Meter,
        streaming_loop: StreamingLoop,
    ) -> None:
        self._logger = logger
        self._tracer = tracer
        self._meter = meter
        self._streaming_loop = streaming_loop

    async def run(self, task: Task) -> TaskResult:
        composition_mode = await self._resolve_composition_mode(task.context)
        output_mode = task.context.agent.message_output_mode

        if (
            output_mode == MessageOutputMode.STREAM
            and composition_mode == CompositionMode.CANNED_FLUID
        ):
            job = LoopJob(
                context=task.context,
                system_instructions=self._build_system_instructions(
                    task.context, task.instructions
                ),
                turn_instructions=self._build_turn_instructions,
                model_size=self._get_model_size(task.context),
                reasoning_config=self._get_reasoning_config(task.context),
            )
            result = await self._streaming_loop.run(job)

            return TaskResult(
                output=result.steps[-1].message.text if result.steps else "",
                usage=result.total_usage,
            )
        else:
            raise Exception(f"Unsupported message output mode: {output_mode}")

    async def _resolve_composition_mode(self, context: EngineContext) -> CompositionMode:
        """Resolve effective composition mode from matched guidelines.

        Most restrictive rule: CANNED_STRICT > CANNED_COMPOSITED > CANNED_FLUID
        """
        if context.agent.composition_mode == CompositionMode.CANNED_STRICT:
            return CompositionMode.CANNED_STRICT

        restrictiveness_priority = {
            CompositionMode.CANNED_STRICT: 3,
            CompositionMode.CANNED_COMPOSITED: 2,
            CompositionMode.CANNED_FLUID: 1,
        }

        most_restrictive_mode: CompositionMode | None = None
        max_restrictiveness = 0

        # Check all matched guidelines for composition mode
        for guideline in context.state.guidelines:
            if guideline.composition_mode is not None:
                mode = guideline.composition_mode

                # Track most restrictive (only CANNED_* modes)
                if mode in restrictiveness_priority:
                    restrictiveness = restrictiveness_priority[mode]
                    if restrictiveness > max_restrictiveness:
                        most_restrictive_mode = mode
                        max_restrictiveness = restrictiveness

        # Default to agent's composition mode
        if most_restrictive_mode is None:
            most_restrictive_mode = context.agent.composition_mode

        return most_restrictive_mode

    def _build_system_instructions(
        self,
        context: EngineContext,
        instructions: str | None,
    ) -> str:
        builder = PromptBuilder(
            on_build=lambda prompt: self._logger.trace(f"TaskRunner system instructions:\n{prompt}")
        )

        builder.add_section(
            name="taskrunner-orientation",
            template="""\
ORIENTATION
-----------
You are an AI agent who is part of a system that interacts with a user. The current state of this interaction will be provided to you later in this message.

Follow the instructions in this prompt and use any tools provided (if any) to complete any necessary task.

IMPORTANT NOTE: Your response will not be displayed to the user. Instead, it will inform a subsequent system component that interacts with the user. Your job is to do the heavy-lifting and reason about and/or run whatever tools necessary, and consequently *summarize* your work, if any, in the output. The output should be concise and only include information that is necessary for the next system component to know in order to interact with the user. Do not include any information that is not strictly necessary for that.

Your output should be phrased so as to provide A SUMMARY of information and instructions to the next system component on how to respond to the user (e.g., "I did this for this reason...", "You should tell the user such and such...") — and not as a message to the user per se.

- If you want to tell the next component what to say, make sure you do it with a simple and concise summary of what you want to communicate, and not with a verbatim message that you want the next component to say. The next component will take care of phrasing the message to the user in a conversational way, so you don't need to worry about that. Just focus on communicating the necessary information and instructions in a clear and concise way.
""",
            props={},
        )

        if instructions:
            builder.add_section(
                name="taskrunner-instructions",
                template=f"""\
YOUR CURRENT TASK
-----------------
{instructions}
""",
                props={"instructions": instructions},
            )

        builder.add_section(
            name="taskrunner-compliance",
            template="""
SAFETY AND COMPLIANCE RULES
---------------------------
Always abide by the following general principles (note these are platform-level instructions - not the business "guidelines". The guidelines will be provided later):

1. ONLY USE FACTUAL INFORMATION FROM THIS PROMPT: Use only factual information explicitly provided in this prompt. Do not supplement with external knowledge or assumptions. For example, even if you know a business's actual address, only share it if it appears in this prompt or interaction history. Treat all information outside this context as unknown. This includes not claiming to perform actions or complete processes unless those specific capabilities are documented in this prompt.
2. ACKNOWLEDGE INFORMATION GAPS: When users request information not contained in this prompt, directly acknowledge the limitation rather than improvising. State clearly that the requested information is not available to you, then offer assistance within your documented scope.
3. THIS IS NOT A ROLE PLAY: This is a real scenario and not a role-play. Your actions have real world consequences. Only respond with what is explicitly stated in this prompt.

Based on previous experience, you seem too eager to please the user by relying on information that is not sourced from this prompt. Be extra careful regarding the last 3 instructions.
""",
            props={},
        )

        builder.add_section(
            name="taskrunner-response-mechanism",
            template="""
RESPONSE MECHANISM
------------------
To craft an optimal response, ensure alignment with all provided guidelines based on the latest interaction state by REASONING about them internally.
Before choosing your response, reason about it by first identifying **up to** three key insights based on this prompt and the ongoing conversation.
These insights should include relevant user requests, applicable principles from this prompt, or conclusions drawn from the interaction.
Ensure to include any user request as an insight, whether it's explicit or implicit.
Do not overly obsess about insights unless you believe that they are absolutely necessary. Prefer reasoning about fewer insights, if at all.

PRIORITIZING INSTRUCTIONS
-------------------------
Deviating from an instruction (either task instructions or guidelines) is acceptable only when the deviation arises from a deliberate prioritization.

Consider the following valid reasons for such deviations:
    - The instruction has already been fulfilled in the conversation, so reiterating it would be redundant (unless the situation warrants it).
    - The instruction contradicts a user request.
    - The instruction lacks sufficient context or data to apply reliably.
    - The instruction depends on an agent intention condition that does not apply in the current situation.
    - When a guideline offers multiple options (e.g., "do X or Y") and another more specific guideline restricts one of those options (e.g., "don’t do X"),
    follow both by choosing the permitted alternative (i.e., do Y).
In all other cases, even if you believe that a conditional guideline's condition does not apply, you must still follow it.
If fulfilling a guideline is not possible, explicitly justify why in your response.

Remember that the instructions and guidelines reflect the explicit wishes of the business you represent. Deviating from them should only occur if doing so does not put the business at risk.

For instance, if a guideline explicitly prohibits a specific action (e.g., "never do X"), or if it's a high-criticality guideline, you must not perform that action, even if requested by the user or supported by an insight.

In cases of conflict, prioritize the business's values and ensure your decisions align with their overarching goals.
""",
        )

        builder.add_agent_identity(context.agent)
        builder.add_customer_identity(context.customer, context.session)
        builder.add_context_variables(context.state.context_variables)

        # How/when to follow guidelines lives in the (cached) system instructions;
        # the matched guidelines themselves are listed per turn (see
        # _build_turn_instructions).
        builder.add_low_criticality_guideline_instructions()
        builder.add_guideline_instructions()

        builder.add_section(
            name="taskrunner-reminder",
            template="""REMINDER: Only use information and services that are sourced from this prompt. Never use your intrinsic knowledge to offer services or provide information. REGARDING YOUR FINAL MESSAGE - REMEMBER THAT YOU ARE NOT RESPONDING DIRECTLY TO THE USER, BUT RATHER INFORMING A SUBSEQUENT SYSTEM COMPONENT AND IMPORTANT INFORMATION AND A RECAP OF TOOLS YOU MAY HAVE RUN.""",
        )

        return builder.build()

    async def _build_turn_instructions(
        self,
        context: EngineContext,
    ) -> str:
        guideline_representations = {
            m.guideline.id: internal_representation(m.guideline)
            for m in chain(
                context.state.ordinary_guideline_matches,
                context.state.tool_enabled_guideline_matches,
            )
        }

        builder = PromptBuilder(
            on_build=lambda prompt: self._logger.trace(f"TaskRunner turn instructions:\n{prompt}")
        )

        builder.add_glossary(list(context.state.glossary_terms))
        builder.add_capabilities_for_message_generation(context.state.capabilities)
        # The how/when explanation is in the system instructions; here we list
        # the matched guidelines themselves (turn-level).
        builder.add_matched_low_criticality_guidelines(
            context.state.ordinary_guideline_matches,
            context.state.tool_enabled_guideline_matches,
            guideline_representations,
        )
        builder.add_matched_guidelines(
            context.state.ordinary_guideline_matches,
            context.state.tool_enabled_guideline_matches,
            guideline_representations,
        )
        builder.add_tool_descriptions(context.state.tools)

        builder.add_section(
            name="taskrunner-reminder",
            template="""REMINDER: Only use information and services that are sourced from this prompt. Never use your intrinsic knowledge to offer services or provide information. REGARDING YOUR FINAL MESSAGE - REMEMBER THAT YOU ARE NOT RESPONDING DIRECTLY TO THE USER, BUT RATHER INFORMING A SUBSEQUENT SYSTEM COMPONENT AND IMPORTANT INFORMATION AND A RECAP OF TOOLS YOU MAY HAVE RUN.""",
        )

        return builder.build()

    def _get_model_size(self, context: EngineContext) -> ModelSize:
        match context.agent.effort:
            case Effort.MIN:
                return ModelSize.SMALL
            case Effort.LOW:
                return ModelSize.SMALL
            case Effort.MEDIUM:
                return ModelSize.MEDIUM
            case Effort.HIGH:
                return ModelSize.MEDIUM
            case Effort.MAX:
                return ModelSize.LARGE
            case _:
                return None

    def _get_reasoning_config(self, context: EngineContext) -> ReasoningConfig | None:
        match context.agent.effort:
            case Effort.MIN:
                return ReasoningConfig(effort="minimal", visibility="none")
            case Effort.LOW:
                return ReasoningConfig(effort="minimal", visibility="none")
            case Effort.MEDIUM:
                return ReasoningConfig(effort="low", visibility="summary")
            case Effort.HIGH:
                return ReasoningConfig(effort="medium", visibility="summary")
            case Effort.MAX:
                return ReasoningConfig(effort="high", visibility="full")
            case _:
                return None
