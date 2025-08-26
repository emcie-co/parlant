# Copyright 2025 Emcie Co Ltd.
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
from pydantic import Field
from typing import Annotated, Optional, Sequence, TypeAlias, cast

from parlant.api.authorization import Operation, AuthorizationPolicy
from parlant.core.common import DefaultBaseModel, JSONSerializable
from parlant.api.common import ExampleJson, apigen_config, example_json_content
from parlant.core.journeys import (
    Journey,
    JourneyEdge,
    JourneyId,
    JourneyNode,
    JourneyNodeId,
    JourneyStore,
    JourneyUpdateParams,
)
from parlant.core.guidelines import GuidelineId, GuidelineStore
from parlant.core.tags import TagId, Tag

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
        default=None,
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
            "Mermaid stateDiagram V2 definition (stateDiagram). " "Render with a Mermaid renderer."
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
    tags: JourneyTagsField


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
    tags: Optional[JourneyTagsField] = None


JourneyConditionUpdateAddField: TypeAlias = Annotated[
    list[GuidelineId],
    Field(
        default=None,
        description="List of guideline IDs to add to the journey",
        examples=[["guid_123xz", "guid_456abc"]],
    ),
]

JourneyConditionUpdateRemoveField: TypeAlias = Annotated[
    list[GuidelineId],
    Field(
        default=None,
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

    add: Optional[JourneyConditionUpdateAddField] = None
    remove: Optional[JourneyConditionUpdateRemoveField] = None


JourneyTagUpdateAddField: TypeAlias = Annotated[
    list[TagId],
    Field(
        default=None,
        description="List of tag IDs to add to the journey",
        examples=[["tag1", "tag2"]],
    ),
]

JourneyTagUpdateRemoveField: TypeAlias = Annotated[
    list[TagId],
    Field(
        default=None,
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

    add: Optional[JourneyTagUpdateAddField] = None
    remove: Optional[JourneyTagUpdateRemoveField] = None


class JourneyUpdateParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": journey_example},
):
    """
    Parameters for updating an existing journey.
    All fields are optional. Only provided fields will be updated.
    """

    title: Optional[JourneyTitleField] = None
    description: Optional[str] = None
    conditions: Optional[JourneyConditionUpdateParamsDTO] = None
    tags: Optional[JourneyTagUpdateParamsDTO] = None


TagIdQuery: TypeAlias = Annotated[
    Optional[TagId],
    Query(
        description="The tag ID to filter journeys by",
        examples=["tag:123"],
    ),
]


async def _build_mermaid_chart(
    journey_store: JourneyStore,
    journey: Journey,
) -> JourneyMermaidChartDTO:
    NORMAL_STYLE = "fill:#006e53,stroke:#ffffff,stroke-width:2px,color:#ffffff"
    TOOL_STYLE = "fill:#ffeeaa,stroke:#ffeeaa,stroke-width:2px,color:#dd6600"

    def _is_tool_node(node: JourneyNode) -> bool:
        return (
            cast(dict[str, JSONSerializable], node.metadata.get("journey_node", {})).get("kind")
            == "tool"
        )

    root_id: JourneyNodeId = journey.root_id
    nodes = await journey_store.list_nodes(journey.id)
    edges = await journey_store.list_edges(journey.id)

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

    def declare(nid: JourneyNodeId) -> None:
        if nid == JourneyStore.END_NODE_ID or nid in declared:
            return
        lbl = node_label(nid)
        if not lbl:
            return
        declared.add(nid)
        m = mermaid_id(nid)
        state_decls.append(f"    {m}: {lbl}")
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
    journey_store: JourneyStore,
    guideline_store: GuidelineStore,
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
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
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
        A unique identifier will be automatically generated.
        """
        await authorization_policy.authorize(request=request, operation=Operation.CREATE_JOURNEY)

        guidelines = [
            await guideline_store.create_guideline(
                condition=condition,
                action=None,
                tags=[],
            )
            for condition in params.conditions
        ]

        journey = await journey_store.create_journey(
            title=params.title,
            description=params.description,
            conditions=[g.id for g in guidelines],
            tags=params.tags,
        )

        for guideline in guidelines:
            await guideline_store.upsert_tag(
                guideline_id=guideline.id,
                tag_id=Tag.for_journey_id(journey.id),
            )

        return JourneyDTO(
            id=journey.id,
            title=journey.title,
            description=journey.description,
            conditions=[g.id for g in guidelines],
            tags=journey.tags,
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

        if tag_id:
            journeys = await journey_store.list_journeys(
                tags=[tag_id],
            )
        else:
            journeys = await journey_store.list_journeys()

        result = []
        for journey in journeys:
            result.append(
                JourneyDTO(
                    id=journey.id,
                    title=journey.title,
                    description=journey.description,
                    conditions=journey.conditions,
                    tags=journey.tags,
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

        journey = await journey_store.read_journey(journey_id=journey_id)

        return JourneyDTO(
            id=journey.id,
            title=journey.title,
            description=journey.description,
            conditions=journey.conditions,
            tags=journey.tags,
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

        journey = await journey_store.read_journey(journey_id=journey_id)
        chart = await _build_mermaid_chart(journey_store, journey)

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
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
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

        journey = await journey_store.read_journey(journey_id=journey_id)

        if params.conditions:
            if params.conditions.add:
                for condition in params.conditions.add:
                    await journey_store.add_condition(
                        journey_id=journey_id,
                        condition=condition,
                    )

                    guideline = await guideline_store.read_guideline(guideline_id=condition)

                    await guideline_store.upsert_tag(
                        guideline_id=condition,
                        tag_id=Tag.for_journey_id(journey_id),
                    )

            if params.conditions.remove:
                for condition in params.conditions.remove:
                    await journey_store.remove_condition(
                        journey_id=journey_id,
                        condition=condition,
                    )

                    guideline = await guideline_store.read_guideline(guideline_id=condition)

                    if guideline.tags == [Tag.for_journey_id(journey_id)]:
                        await guideline_store.delete_guideline(guideline_id=condition)
                    else:
                        await guideline_store.remove_tag(
                            guideline_id=condition,
                            tag_id=Tag.for_journey_id(journey_id),
                        )

        update_params: JourneyUpdateParams = {}
        if params.title:
            update_params["title"] = params.title
        if params.description:
            update_params["description"] = params.description

        if update_params:
            journey = await journey_store.update_journey(
                journey_id=journey_id,
                params=update_params,
            )

        if params.tags:
            if params.tags.add:
                for tag in params.tags.add:
                    await journey_store.upsert_tag(journey_id=journey_id, tag_id=tag)

            if params.tags.remove:
                for tag in params.tags.remove:
                    await journey_store.remove_tag(journey_id=journey_id, tag_id=tag)

        journey = await journey_store.read_journey(journey_id=journey_id)

        return JourneyDTO(
            id=journey.id,
            title=journey.title,
            description=journey.description,
            conditions=journey.conditions,
            tags=journey.tags,
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

        journey = await journey_store.read_journey(journey_id=journey_id)

        await journey_store.delete_journey(journey_id=journey_id)

        for condition in journey.conditions:
            if not await journey_store.list_journeys(condition=condition):
                await guideline_store.delete_guideline(guideline_id=condition)
            else:
                guideline = await guideline_store.read_guideline(guideline_id=condition)

                if guideline.tags == [Tag.for_journey_id(journey_id)]:
                    await guideline_store.delete_guideline(guideline_id=condition)
                else:
                    await guideline_store.remove_tag(
                        guideline_id=condition,
                        tag_id=Tag.for_journey_id(journey_id),
                    )

    return router
