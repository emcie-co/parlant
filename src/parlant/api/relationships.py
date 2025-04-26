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

from itertools import chain
from typing import Optional, Sequence, cast, Annotated, TypeAlias
from fastapi import APIRouter, HTTPException, Path, Query, status

from parlant.api import common
from parlant.api.common import (
    ExampleJson,
    GuidelineIdField,
    RelationshipDTO,
    GuidelineRelationshipKindDTO,
    TagDTO,
    TagIdField,
    apigen_config,
    guideline_relationship_kind_dto_to_kind,
    guideline_relationship_kind_to_dto,
    guideline_to_dto,
    relationship_example,
)
from parlant.core.common import DefaultBaseModel
from parlant.core.relationships import (
    EntityType,
    Relationship,
    RelationshipId,
    RelationshipStore,
)
from parlant.core.guidelines import Guideline, GuidelineId, GuidelineStore
from parlant.core.tags import Tag, TagId, TagStore

API_GROUP = "relationships"


relationship_creation_params_example: ExampleJson = {
    "source_guideline": "gid_123",
    "target_tag": "tid_456",
    "kind": "entailment",
}


class RelationshipCreationParamsDTO(
    DefaultBaseModel,
    json_schema_extra={"example": relationship_creation_params_example},
):
    source_guideline: Optional[GuidelineIdField] = None
    source_tag: Optional[TagIdField] = None
    target_guideline: Optional[GuidelineIdField] = None
    target_tag: Optional[TagIdField] = None
    kind: GuidelineRelationshipKindDTO


GuidelineIdQuery: TypeAlias = Annotated[
    GuidelineId,
    Query(description="The ID of the guideline to list relationships for"),
]


TagIdQuery: TypeAlias = Annotated[
    TagId,
    Query(description="The ID of the tag to list relationships for"),
]


IndirectQuery: TypeAlias = Annotated[
    bool,
    Query(description="Whether to include indirect relationships"),
]


RelationshipKindQuery: TypeAlias = Annotated[
    GuidelineRelationshipKindDTO,
    Query(description="The kind of relationship to list"),
]


RelationshipIdPath: TypeAlias = Annotated[
    RelationshipId,
    Path(
        description="identifier of relationship",
        examples=[RelationshipId("gr_123")],
    ),
]


def create_router(
    guideline_store: GuidelineStore,
    tag_store: TagStore,
    relationship_store: RelationshipStore,
) -> APIRouter:
    async def relationship_to_dto(
        relationship: Relationship,
    ) -> RelationshipDTO:
        source_guideline = (
            await guideline_store.read_guideline(
                guideline_id=cast(GuidelineId, relationship.source)
            )
            if relationship.source_type == EntityType.GUIDELINE
            else None
        )

        source_tag = (
            await tag_store.read_tag(tag_id=cast(TagId, relationship.source))
            if relationship.source_type == EntityType.TAG
            else None
        )

        target_guideline = (
            await guideline_store.read_guideline(
                guideline_id=cast(GuidelineId, relationship.target)
            )
            if relationship.target_type == EntityType.GUIDELINE
            else None
        )

        target_tag = (
            await tag_store.read_tag(tag_id=cast(TagId, relationship.target))
            if relationship.target_type == EntityType.TAG
            else None
        )

        return RelationshipDTO(
            id=relationship.id,
            source_guideline=guideline_to_dto(cast(Guideline, source_guideline))
            if relationship.source_type == EntityType.GUIDELINE
            else None,
            source_tag=TagDTO(
                id=cast(Tag, source_tag).id,
                name=cast(Tag, source_tag).name,
            )
            if relationship.source_type == EntityType.TAG
            else None,
            target_guideline=guideline_to_dto(cast(Guideline, target_guideline))
            if relationship.target_type == EntityType.GUIDELINE
            else None,
            target_tag=TagDTO(
                id=cast(Tag, target_tag).id,
                name=cast(Tag, target_tag).name,
            )
            if relationship.target_type == EntityType.TAG
            else None,
            indirect=True,
            kind=guideline_relationship_kind_to_dto(relationship.kind),
        )

    router = APIRouter()

    @router.post(
        "",
        status_code=status.HTTP_201_CREATED,
        operation_id="create_relationship",
        response_model=RelationshipDTO,
        responses={
            status.HTTP_201_CREATED: {
                "description": "Relationship successfully created. Returns the created relationship.",
                "content": common.example_json_content(relationship_example),
            },
            status.HTTP_422_UNPROCESSABLE_ENTITY: {
                "description": "Validation error in request parameters"
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="create"),
    )
    async def create_relationship(
        params: RelationshipCreationParamsDTO,
    ) -> RelationshipDTO:
        """
        Create a relationship.

        A relationship is a relationship between a guideline and a tag.
        It can be created between a guideline and a tag, or between two guidelines, or between two tags.
        """
        if params.source_guideline and params.source_tag:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="A relationship cannot have both a source guideline and a source tag",
            )
        elif params.target_guideline and params.target_tag:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="A relationship cannot have both a target guideline and a target tag",
            )
        elif (
            params.source_guideline
            and params.target_guideline
            and params.source_guideline == params.target_guideline
        ) or (params.source_tag and params.target_tag and params.source_tag == params.target_tag):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="source and target cannot be the same entity",
            )

        if params.source_guideline:
            await guideline_store.read_guideline(guideline_id=params.source_guideline)
        else:
            await tag_store.read_tag(tag_id=cast(TagId, params.source_tag))

        if params.target_guideline:
            await guideline_store.read_guideline(guideline_id=params.target_guideline)
        else:
            await tag_store.read_tag(tag_id=cast(TagId, params.target_tag))

        relationship = await relationship_store.create_relationship(
            source=cast(GuidelineId | TagId, params.source_guideline)
            if params.source_guideline
            else cast(GuidelineId | TagId, params.source_tag),
            source_type=EntityType.GUIDELINE if params.source_guideline else EntityType.TAG,
            target=cast(GuidelineId | TagId, params.target_guideline)
            if params.target_guideline
            else cast(GuidelineId | TagId, params.target_tag),
            target_type=EntityType.GUIDELINE if params.target_guideline else EntityType.TAG,
            kind=guideline_relationship_kind_dto_to_kind(params.kind),
        )

        return await relationship_to_dto(relationship=relationship)

    @router.get(
        "",
        operation_id="list_relationships",
        response_model=Sequence[RelationshipDTO],
        responses={
            status.HTTP_200_OK: {
                "description": "Relationships successfully retrieved. Returns a list of all relationships.",
                "content": common.example_json_content([relationship_example]),
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="list"),
    )
    async def list_relationships(
        kind: RelationshipKindQuery,
        indirect: IndirectQuery = True,
        guideline_id: Optional[GuidelineIdQuery] = None,
        tag_id: Optional[TagIdQuery] = None,
    ) -> Sequence[RelationshipDTO]:
        """
        List relationships.

        Either `guideline_id` or `tag_id` must be provided.
        """
        if guideline_id is None and tag_id is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Either guideline_id or tag_id must be provided",
            )

        if guideline_id:
            await guideline_store.read_guideline(guideline_id=guideline_id)
        elif tag_id:
            await tag_store.read_tag(tag_id=tag_id)

        entity_id = (
            cast(GuidelineId | TagId, guideline_id)
            if guideline_id
            else cast(GuidelineId | TagId, tag_id)
        )

        source_relationships = await relationship_store.list_relationships(
            kind=guideline_relationship_kind_dto_to_kind(kind),
            source=entity_id,
            indirect=indirect,
        )

        target_relationships = await relationship_store.list_relationships(
            kind=guideline_relationship_kind_dto_to_kind(kind),
            target=entity_id,
            indirect=indirect,
        )
        relationships = chain(source_relationships, target_relationships)

        return [
            await relationship_to_dto(relationship=relationship) for relationship in relationships
        ]

    @router.get(
        "/{relationship_id}",
        operation_id="read_relationship",
        status_code=status.HTTP_200_OK,
        response_model=RelationshipDTO,
        responses={
            status.HTTP_200_OK: {
                "description": "Relationship successfully retrieved. Returns the requested relationship.",
                "content": common.example_json_content(relationship_example),
            },
        },
        **apigen_config(group_name=API_GROUP, method_name="retrieve"),
    )
    async def read_relationship(
        relationship_id: RelationshipIdPath,
    ) -> RelationshipDTO:
        """
        Read a relationship by ID.
        """
        relationship = await relationship_store.read_relationship(id=relationship_id)

        return await relationship_to_dto(relationship=relationship)

    @router.delete(
        "/{relationship_id}",
        operation_id="delete_relationship",
        status_code=status.HTTP_204_NO_CONTENT,
        responses={
            status.HTTP_204_NO_CONTENT: {"description": "Relationship successfully deleted."},
            status.HTTP_404_NOT_FOUND: {"description": "Relationship not found."},
        },
        **apigen_config(group_name=API_GROUP, method_name="delete"),
    )
    async def delete_relationship(
        relationship_id: RelationshipIdPath,
    ) -> None:
        """
        Delete a relationship by ID.
        """
        await relationship_store.delete_relationship(id=relationship_id)

    return router
