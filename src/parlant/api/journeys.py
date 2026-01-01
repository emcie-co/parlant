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

from collections import defaultdict
from fastapi import APIRouter, Path, Query, Request, status
from fastapi.responses import PlainTextResponse
from html import escape
from pydantic import Field
from typing import Annotated, Sequence, TypeAlias, cast

from parlant.api.authorization import Operation, AuthorizationPolicy
from parlant.api.common import (
    CompositionModeDTO,
    ExampleJson,
    apigen_config,
    composition_mode_dto_to_composition_mode,
    composition_mode_to_composition_mode_dto,
    example_json_content,
)
from parlant.core.app_modules.journeys import (
    JourneyConditionUpdateParams,
    JourneyGraph,
    JourneyTagUpdateParams,
)
from parlant.core.application import Application
from parlant.core.common import DefaultBaseModel, JSONSerializable
from parlant.core.journeys import (
    JourneyEdge,
    JourneyId,
    JourneyNode,
    JourneyNodeId,
    JourneyStore,
)
from parlant.core.guidelines import GuidelineId
from parlant.core.tags import TagId
import re

API_GROUP = "journeys"

JourneyIdPath: TypeAlias = Annotated[
    JourneyId,
    Path(
        description="Unique identifier for the journey",
        examples=["IUCGT-lvpS"],
        min_length=1,
    ),
]

JourneyTitleField: TypeAlias = Annotated[
    str,
    Field(
        description="The title of the journey",
        examples=["Customer Onboarding", "Product Support"],
        min_length=1,
        max_length=100,
    ),
]

JourneyDescriptionField: TypeAlias = Annotated[
    str,
    Field(
        description="Detailed description of the journey's purpose and flow",
        examples=[
            """1. Customer wants to lock their card
2. Customer reports that their card doesn't work
3. Customer suspects their card has been stolen"""
        ],
    ),
]

JourneyConditionField: TypeAlias = Annotated[
    str,
    Field(
        description="The condition that triggers this journey",
        examples=["Customer asks for help with onboarding"],
        min_length=1,
    ),
]

JourneyTagsField: TypeAlias = Annotated[
    list[TagId],
    Field(
        description="List of tag IDs associated with the journey",
        examples=[["tag1", "tag2"]],
    ),
]

journey_example: ExampleJson = {
    "id": "IUCGT-lvpS",
    "title": "Customer Onboarding",
    "description": """1. Customer wants to lock their card
2. Customer reports that their card doesn't work
3. Customer suspects their card has been stolen""",
    "conditions": [
        "customer needs unlocking their card",
        "customer needs help with card",
    ],
    "tags": ["tag1", "tag2"],
}

JourneyMermaidChartDTO: TypeAlias = Annotated[
    str,
    Field(
        description=(
            "Mermaid stateDiagram V2 definition (stateDiagram). Render with a Mermaid renderer."
        ),
        examples=[
            """
stateDiagram
    [*] --> A
    A --> B
    N1 --> END((End))
"""
        ],
    ),
]


class JourneyDTO(
    DefaultBaseModel,
    json_schema_extra={"example": journey_example},
):
    """
    A journey represents a guided interaction path for specific user scenarios.

    Each journey is triggered by a condition and contains steps to guide the interaction.
    """

    id: JourneyIdPath
    title: JourneyTitleField
    description: str
    conditions: Sequence[GuidelineId]
    tags: JourneyTagsField = []
    composition_mode: CompositionModeDTO | None = None


class JourneyCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": journey_example},
):
    """
    Parameters for creating a new journey.
    """

    title: JourneyTitleField
    description: str
    conditions: Sequence[JourneyConditionField]
    id: JourneyIdPath | None = None
    tags: JourneyTagsField | None = None
    composition_mode: CompositionModeDTO | None = None


JourneyConditionUpdateAddField: TypeAlias = Annotated[
    list[GuidelineId],
    Field(
        description="List of guideline IDs to add to the journey",
        examples=[["guid_123xz", "guid_456abc"]],
    ),
]

JourneyConditionUpdateRemoveField: TypeAlias = Annotated[
    list[GuidelineId],
    Field(
        description="List of guideline IDs to remove from the journey",
        examples=[["guid_123xz", "guid_456abc"]],
    ),
]

journey_condition_update_params_example: ExampleJson = {
    "add": [
        "guid_123xz",
        "guid_456abc",
    ],
    "remove": [
        "guid_789def",
        "guid_012ghi",
    ],
}


class JourneyConditionUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": journey_condition_update_params_example},
):
    """
    Parameters for updating an existing journey's conditions.
    """

    add: JourneyConditionUpdateAddField | None = None
    remove: JourneyConditionUpdateRemoveField | None = None


JourneyTagUpdateAddField: TypeAlias = Annotated[
    list[TagId],
    Field(
        description="List of tag IDs to add to the journey",
        examples=[["tag1", "tag2"]],
    ),
]

JourneyTagUpdateRemoveField: TypeAlias = Annotated[
    list[TagId],
    Field(
        description="List of tag IDs to remove from the journey",
        examples=[["tag1", "tag2"]],
    ),
]

journey_tag_update_params_example: ExampleJson = {
    "add": [
        "t9a8g703f4",
        "tag_456abc",
    ],
    "remove": [
        "tag_789def",
        "tag_012ghi",
    ],
}


class JourneyTagUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": journey_tag_update_params_example},
):
    """
    Parameters for updating an existing journey's tags.
    """

    add: JourneyTagUpdateAddField | None = None
    remove: JourneyTagUpdateRemoveField | None = None


class JourneyUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": journey_example},
):
    """
    Parameters for updating an existing journey.
    All fields are optional. Only provided fields will be updated.
    """

    title: JourneyTitleField | None = None
    description: str | None = None
    conditions: JourneyConditionUpdateParamsDTO | None = None
    tags: JourneyTagUpdateParamsDTO | None = None
    composition_mode: CompositionModeDTO | None = None


TagIdQuery: TypeAlias = Annotated[
    TagId | None,
    Query(
        description="The tag ID to filter journeys by",
        examples=["tag:123"],
    ),
]


async def _build_mermaid_chart(
    model: JourneyGraph,
) -> JourneyMermaidChartDTO:
    NORMAL_STYLE = "fill:#006e53,stroke:#ffffff,stroke-width:2px,color:#ffffff"
    TOOL_STYLE = "fill:#ffeeaa,stroke:#ffeeaa,stroke-width:2px,color:#dd6600"

    def _is_tool_node(node: JourneyNode) -> bool:
        return (
            cast(dict[str, JSONSerializable], node.metadata.get("journey_node", {})).get("kind")
            == "tool"
        )

    root_id: JourneyNodeId = model.journey.root_id
    nodes = model.nodes
    edges = model.edges

    node_by_id = {n.id: n for n in nodes if n.id != JourneyStore.END_NODE_ID}

    outgoing: dict[JourneyNodeId, list[JourneyEdge]] = defaultdict(list)
    for e in edges:
        outgoing[e.source].append(e)

    alias: dict[JourneyNodeId, str] = {}

    def mermaid_id(nid: JourneyNodeId) -> str:
        if nid == JourneyStore.END_NODE_ID:
            return "[*]"
        if nid not in alias:
            alias[nid] = f"N{len(alias)}"
        return alias[nid]

    def node_label(nid: JourneyNodeId) -> str:
        if nid == JourneyStore.END_NODE_ID:
            return "End"
        n = node_by_id.get(nid)
        if not n:
            return ""
        return n.action or ""

    lines: list[str] = []
    lines.append("stateDiagram-v2")

    visited: set[JourneyNodeId] = set()
    declared: set[JourneyNodeId] = set()

    state_decls: list[str] = []
    transitions: list[str] = []
    style_lines: list[str] = []

    def escape_mermaid(s: str) -> str:
        def convert_match(match: re.Match[str]) -> str:
            number = match.group(1)
            if number.startswith("x"):
                dec_num = int(number[1:], 16)  # convert hex to decimal
                return f"#{dec_num};"
            else:
                return f"#{number};"  # keep decimal as is

        html_escaped = escape(s, quote=True)

        # apply regex replacement to fix numeric character references for mermaid syntax
        return re.sub(r"&#(x[0-9a-fA-F]+|[0-9]+);", convert_match, html_escaped)

    def declare(nid: JourneyNodeId) -> None:
        if nid == JourneyStore.END_NODE_ID or nid in declared:
            return
        lbl = node_label(nid)
        if not lbl:
            return
        declared.add(nid)
        m = mermaid_id(nid)
        state_decls.append(f'    state "{escape_mermaid(lbl)}" as {m}')
        node = node_by_id.get(nid)
        if node and _is_tool_node(node):
            style_lines.append(f"style {m} {TOOL_STYLE}")
        else:
            style_lines.append(f"style {m} {NORMAL_STYLE}")

    declare(root_id)

    for e in outgoing.get(root_id, []):
        tid = e.target
        declare(tid)
        if e.condition:
            transitions.append(f"    [*] --> {mermaid_id(tid)}: {e.condition}")
        else:
            transitions.append(f"    [*] --> {mermaid_id(tid)}")

    stack: list[JourneyNodeId] = [root_id]
    while stack:
        nid = stack.pop()
        if nid in visited:
            continue
        visited.add(nid)

        for e in outgoing.get(nid, []):
            tid = e.target
            declare(tid)

            # Skip standard transition if it would be from an unlabeled root;
            # we already emitted [*] --> target above for those.
            if not (nid == root_id and node_label(nid) == ""):
                src = mermaid_id(nid)
                dst = mermaid_id(tid)
                if e.condition:
                    transitions.append(f"    {src} --> {dst}: {e.condition}")
                else:
                    transitions.append(f"    {src} --> {dst}")

            if tid != JourneyStore.END_NODE_ID and tid not in visited:
                stack.append(tid)

    orphans = [n.id for n in nodes if n.id not in visited and n.id != JourneyStore.END_NODE_ID]
    if orphans:
        lines.append("    %% Unreachable states:")
        for oid in orphans:
            declare(oid)
            lbl = node_label(oid)
            if lbl:
                lines.append(f"    %%   {mermaid_id(oid)}: {lbl}")
            else:
                lines.append(f"    %%   {mermaid_id(oid)}")

    lines.extend(state_decls)
    lines.extend(transitions)
    lines.extend(style_lines)

    return "\n".join(lines)


def create_router(
    authorization_policy: AuthorizationPolicy,
    app: Application,
) -> APIRouter:
    router = APIRouter()

    @router.post(
        "",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_journey",
        response_model=JourneyDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Journey successfully created. Returns the complete journey object including generated ID.",
                "content": example_json_content(journey_example),
            },
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create"),
    )
    async def create_journey(
        request: Request,
        params: JourneyCreationParamsDTO,
    ) -> JourneyDTO:
        """
        Creates a new journey in the system.

        The journey will be initialized with the provided title, description, and conditions.
        A unique identifier will be automatically generated unless a custom ID is provided.
        """
        await authorization_policy.authorize(request=request, operation=Operation.CREATE_JOURNEY)

        journey, guidelines = await app.journeys.create(
            title=params.title,
            description=params.description,
            conditions=params.conditions,
            tags=params.tags,
            id=params.id,
            composition_mode=composition_mode_dto_to_composition_mode(params.composition_mode)
            if params.composition_mode
            else None,
        )

        return JourneyDTO(
            id=journey.id,
            title=journey.title,
            description=journey.description,
            conditions=[g.id for g in guidelines],
            tags=journey.tags,
            composition_mode=composition_mode_to_composition_mode_dto(journey.composition_mode)
            if journey.composition_mode
            else None,
        )

    @router.get(
        "",
        operation_id="list_journeys",
        response_model=Sequence[JourneyDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "List of all journeys in the system",
                "content": example_json_content([journey_example]),
            }
        },
        **apigen_config(group_name=API_GROUP, method_name="list"),
    )
    async def list_journeys(
        request: Request,
        tag_id: TagIdQuery = None,
    ) -> Sequence[JourneyDTO]:
        """
        Retrieves a list of all journeys in the system.
        """
        await authorization_policy.authorize(request=request, operation=Operation.LIST_JOURNEYS)

        journeys = await app.journeys.find(tag_id)

        result = []
        for journey in journeys:
            result.append(
                JourneyDTO(
                    id=journey.id,
                    title=journey.title,
                    description=journey.description,
                    conditions=journey.conditions,
                    tags=journey.tags,
                    composition_mode=composition_mode_to_composition_mode_dto(
                        journey.composition_mode
                    )
                    if journey.composition_mode
                    else None,
                )
            )

        return result

    @router.get(
        "/{journey_id}",
        operation_id="read_journey",
        response_model=JourneyDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Journey details successfully retrieved. Returns the complete journey object.",
                "content": example_json_content(journey_example),
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Journey not found. the specified `journey_id` does not exist"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve"),
    )
    async def read_journey(
        request: Request,
        journey_id: JourneyIdPath,
    ) -> JourneyDTO:
        """
        Retrieves details of a specific journey by ID.
        """
        await authorization_policy.authorize(request=request, operation=Operation.READ_JOURNEY)

        model = await app.journeys.read(journey_id=journey_id)

        return JourneyDTO(
            id=model.journey.id,
            title=model.journey.title,
            description=model.journey.description,
            conditions=model.journey.conditions,
            tags=model.journey.tags,
            composition_mode=composition_mode_to_composition_mode_dto(
                model.journey.composition_mode
            )
            if model.journey.composition_mode
            else None,
        )

    @router.get(
        "/{journey_id}/mermaid",
        operation_id="journey_mermaid",
        response_class=PlainTextResponse,
        responses={
            status.HTTP_200_OK: {
                "description": "Mermaid stateDiagram V2 (text/plain). Copy/paste directly into a Mermaid renderer.",
                "content": {"text/plain": {"example": "stateDiagram\n  [*] --> A\n  A --> B\n"}},
            },
            status.HTTP_404_NOT_FOUND: {"description": "Journey not found"},
        },
        **apigen_config(group_name=API_GROUP, method_name="mermaid"),
    )
    async def journey_mermaid(
        request: Request,
        journey_id: JourneyIdPath,
    ) -> str:
        """
        Returns the journey as a Mermaid 'stateDiagramv-v2' string.
        Content-Type: text/plain
        """
        await authorization_policy.authorize(request=request, operation=Operation.READ_JOURNEY)

        model = await app.journeys.read(journey_id=journey_id)
        chart = await _build_mermaid_chart(model)

        return chart

    @router.patch(
        "/{journey_id}",
        operation_id="update_journey",
        response_model=JourneyDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Journey successfully updated. Returns the updated journey.",
                "content": example_json_content(journey_example),
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Journey not found. the specified `journey_id` does not exist"
            },
            status.HTTP_422_UNPROCESSABLE_CONTENT: {
                "description": "Validation error in update parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="update"),
    )
    async def update_journey(
        request: Request,
        journey_id: JourneyIdPath,
        params: JourneyUpdateParamsDTO,
    ) -> JourneyDTO:
        """
        Updates an existing journey's attributes.

        Only the provided attributes will be updated; others will remain unchanged.
        """
        await authorization_policy.authorize(request=request, operation=Operation.UPDATE_JOURNEY)

        journey = await app.journeys.update(
            journey_id=journey_id,
            title=params.title,
            description=params.description,
            conditions=JourneyConditionUpdateParams(
                add=params.conditions.add, remove=params.conditions.remove
            )
            if params.conditions
            else None,
            tags=JourneyTagUpdateParams(add=params.tags.add, remove=params.tags.remove)
            if params.tags
            else None,
            composition_mode=composition_mode_dto_to_composition_mode(params.composition_mode)
            if params.composition_mode
            else None,
        )

        return JourneyDTO(
            id=journey.id,
            title=journey.title,
            description=journey.description,
            conditions=journey.conditions,
            tags=journey.tags,
            composition_mode=composition_mode_to_composition_mode_dto(journey.composition_mode)
            if journey.composition_mode
            else None,
        )

    @router.delete(
        "/{journey_id}",
        operation_id="delete_journey",
        status_code=status.HTTP_204_NO_CONTENT,
        responses={
            status.HTTP_204_NO_CONTENT: {
                "description": "Journey successfully deleted. No content returned."
            },
            status.HTTP_404_NOT_FOUND: {
                "description": "Journey not found. The specified `journey_id` does not exist"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="delete"),
    )
    async def delete_journey(
        request: Request,
        journey_id: JourneyIdPath,
    ) -> None:
        """
        Deletes a journey from the system.

        Also deletes the associated guideline.
        Deleting a non-existent journey will return 404.
        No content will be returned from a successful deletion.
        """
        await authorization_policy.authorize(request=request, operation=Operation.DELETE_JOURNEY)

        await app.journeys.delete(journey_id)

    return router
