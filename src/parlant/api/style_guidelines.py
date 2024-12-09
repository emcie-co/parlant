# # Copyright 2024 Emcie Co Ltd.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# from fastapi import APIRouter, HTTPException, Path, status
# from pydantic import BaseModel
# from typing import Optional, Sequence, TypeAlias, Annotated

# from datetime import datetime
# from parlant.api.common import ExampleJson
# from parlant.core.common import ItemNotFoundError
# from parlant.core.style_guidelines import StyleGuidelineDocumentStore, StyleGuidelineId, StyleGuidelineContent, StyleGuideline
# from parlant.core.persistence.document_database import DocumentDatabase

# API_GROUP = "style_guidelines"

# style_guideline_dto_example: ExampleJson = {
#     "id": "style_12345",
#     "before": "Hello, how can I help you?",
#     "after": "Hello, Terry, how can I help you?",
#     "violation": "Did not greet the user with their first name.",
#     "style_guide": "When greeting the user, include their first name in the greeting.",
# }

# StyleGuidelineIdPath: TypeAlias = Annotated[
#     StyleGuidelineId,
#     Path(description="Unique identifier for the style guideline", examples=["style_12345"]),
# ]


# class StyleGuidelineDTO(BaseModel):
#     id: StyleGuidelineIdPath
#     before: str
#     after: str
#     violation: str
#     style_guide: str


# class StyleGuidelineCreationParams(BaseModel):
#     before: str
#     after: str
#     violation: str
#     style_guide: str


# class StyleGuidelineUpdateParams(BaseModel):
#     before: Optional[str]
#     after: Optional[str]
#     violation: Optional[str]
#     style_guide: Optional[str]


# def create_style_guidelines_router(database: DocumentDatabase) -> APIRouter:
#     router = APIRouter()
#     store = StyleGuidelineDocumentStore(database)

#     @router.post(
#         "/style-guidelines",
#         response_model=StyleGuidelineDTO,
#         status_code=status.HTTP_201_CREATED,
#         operation_id="create_style_guideline",
#         response_model=StyleGuidelineCreationResult,
#         responses={
#             status.HTTP_201_CREATED: {
#                 "description": "Style guidelines successfully created. Returns the created style guidelines.",
#                 "content": common.example_json_content(guideline_creation_result_example),
#             },
#             status.HTTP_404_NOT_FOUND: {"description": "Agent not found"},
#             status.HTTP_422_UNPROCESSABLE_ENTITY: {
#                 "description": "Validation error in request parameters"
#             },
#         },
#         **apigen_config(group_name=API_GROUP, method_name="create"),
#     )
#     async def create_style_guideline(params: StyleGuidelineCreationParams) -> StyleGuidelineDTO:
#         guideline = await store.create_style_guideline(
#             style_guideline_set="default",
#             before=params.before,
#             after=params.after,
#             violation=params.violation,
#             style_guide=params.style_guide,
#         )
#         return StyleGuidelineDTO(
#             id=guideline.id,
#             before=guideline.content.before,
#             after=guideline.content.after,
#             violation=guideline.content.violation,
#             style_guide=guideline.content.style_guide,
#         )

#     @router.get(
#         "/style-guidelines/{style_guideline_id}",
#         response_model=StyleGuidelineDTO,
#         operation_id="get_style_guideline",
#         summary="Get a specific style guideline",
#     )
#     async def get_style_guideline(style_guideline_id: StyleGuidelineIdPath) -> StyleGuidelineDTO:
#         try:
#             guideline = await store.read_style_guideline(
#                 style_guideline_set="default",
#                 style_guideline_id=style_guideline_id,
#             )
#             return StyleGuidelineDTO(
#                 id=guideline.id,
#                 before=guideline.content.before,
#                 after=guideline.content.after,
#                 violation=guideline.content.violation,
#                 style_guide=guideline.content.style_guide,
#             )
#         except ItemNotFoundError:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"Style guideline with id '{style_guideline_id}' not found",
#             )

#     @router.get(
#         "/style-guidelines",
#         response_model=Sequence[StyleGuidelineDTO],
#         operation_id="list_style_guidelines",
#         summary="List all style guidelines",
#     )
#     async def list_style_guidelines() -> Sequence[StyleGuidelineDTO]:
#         guidelines = await store.list_style_guidelines(style_guideline_set="default")
#         return [
#             StyleGuidelineDTO(
#                 id=guideline.id,
#                 before=guideline.content.before,
#                 after=guideline.content.after,
#                 violation=guideline.content.violation,
#                 style_guide=guideline.content.style_guide,
#             )
#             for guideline in guidelines
#         ]

#     @router.patch(
#         "/style-guidelines/{style_guideline_id}",
#         response_model=StyleGuidelineDTO,
#         operation_id="update_style_guideline",
#         summary="Update a specific style guideline",
#     )
#     async def update_style_guideline(
#         style_guideline_id: StyleGuidelineIdPath, params: StyleGuidelineUpdateParams
#     ) -> StyleGuidelineDTO:
#         try:
#             updated_guideline = await store.update_style_guideline(
#                 style_guideline_id=style_guideline_id,
#                 params=params.dict(exclude_unset=True),
#             )
#             return StyleGuidelineDTO(
#                 id=updated_guideline.id,
#                 before=updated_guideline.content.before,
#                 after=updated_guideline.content.after,
#                 violation=updated_guideline.content.violation,
#                 style_guide=updated_guideline.content.style_guide,
#             )
#         except ItemNotFoundError:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"Style guideline with id '{style_guideline_id}' not found",
#             )

#     @router.delete(
#         "/style-guidelines/{style_guideline_id}",
#         status_code=status.HTTP_204_NO_CONTENT,
#         operation_id="delete_style_guideline",
#         summary="Delete a specific style guideline",
#     )
#     async def delete_style_guideline(style_guideline_id: StyleGuidelineIdPath):
#         try:
#             await store.delete_style_guideline(
#                 style_guideline_set="default", style_guideline_id=style_guideline_id
#             )
#         except ItemNotFoundError:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"Style guideline with id '{style_guideline_id}' not found",
#             )

#     return router
