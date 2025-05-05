from __future__ import annotations

from typing import Optional, Sequence

from parlant.core.guidelines import GuidelineContent
from parlant.core.loggers import Logger
from parlant.core.nlp.generation import SchematicGenerator
from parlant.core.engines.alpha.prompt_builder import PromptBuilder
from parlant.core.common import DefaultBaseModel
from parlant.core.services.indexing.common import ProgressReport
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.tools import Tool, ToolId


class GuidelineActionProposition(DefaultBaseModel):
    content: GuidelineContent
    rationale: str


class GuidelineActionPropositionSchema(DefaultBaseModel):
    action: str
    rationale: str


class GuidelineActionProposer:
    def __init__(
        self,
        logger: Logger,
        schematic_generator: SchematicGenerator[GuidelineActionPropositionSchema],
        service_registry: ServiceRegistry,
    ) -> None:
        self._logger = logger
        self._schematic_generator = schematic_generator
        self._service_registry = service_registry

    async def propose_action(
        self,
        guideline: GuidelineContent,
        tool_ids: Sequence[ToolId],
        progress_report: Optional[ProgressReport] = None,
    ) -> GuidelineActionProposition:
        if not tool_ids or guideline.action:
            return GuidelineActionProposition(
                content=guideline,
                rationale="No action proposed",
            )

        if progress_report:
            await progress_report.stretch(1)

        with self._logger.scope("GuidelineActionProposer"):
            tools: list[Tool] = []
            for tid in tool_ids:
                service = await self._service_registry.read_tool_service(tid.service_name)
                tool = await service.read_tool(tid.tool_name)
                tools.append(tool)

            proposition = await self._generate_action(guideline, tools, tool_ids)

            if progress_report:
                await progress_report.increment()

        return GuidelineActionProposition(
            content=GuidelineContent(
                condition=guideline.condition,
                action=proposition.action,
            ),
            rationale=proposition.rationale,
        )

    async def _build_prompt(
        self,
        guideline: GuidelineContent,
        tools: Sequence[Tool],
        tool_ids: Sequence[ToolId],
    ) -> PromptBuilder:
        builder = PromptBuilder()

        builder.add_section(
            name="guideline-action-proposer-general-instructions",
            template="""
Your task is to craft a *then* clause for a conversational AI guideline that should be executed when the guideline's *when* condition is true.

You will receive the *when* condition and a list of tools (including their descriptions and parameters). Your output should **only** be a JSON object following the required schema and contain a single field `action` with the textual instruction for the agent.

Guidelines for writing the action:
1. Be concise but explicit about invoking the tool(s).
2. Mention the tool name(s).
3. Do not include angle brackets, markdown, or additional commentary outside the JSON.
""",
        )

        builder.add_section(
            name="guideline-action-proposer-guideline",
            template="""
Guideline *when* condition:
+--------------------------
+{condition}
+""",
            props={"condition": guideline.condition},
        )

        tools_text = "\n".join(
            f"- {tid.to_string()}: {tool.description or 'No description'}"
            for tid, tool in zip(tool_ids, tools)
        )
        builder.add_section(
            name="guideline-action-proposer-tools",
            template="""
Relevant tools:
+--------------
+{tools_text}
+""",
            props={"tools_text": tools_text},
        )

        builder.add_section(
            name="guideline-action-proposer-output-format",
            template="""
Expected output (JSON):
```json
{{
  "action": "<SINGLE-LINE-INSTRUCTION>",
  "rationale": "<RATIONALE>"
}}
```
""",
        )

        return builder

    async def _generate_action(
        self,
        guideline: GuidelineContent,
        tools: Sequence[Tool],
        tool_ids: Sequence[ToolId],
    ) -> GuidelineActionPropositionSchema:
        prompt = await self._build_prompt(guideline, tools, tool_ids)

        response = await self._schematic_generator.generate(
            prompt=prompt,
            hints={"temperature": 0.0},
        )

        return response.content
