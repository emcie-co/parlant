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

# from typing import NewType, Optional, Sequence
# from typing_extensions import override, TypedDict
# from abc import ABC, abstractmethod
# from dataclasses import dataclass
# from datetime import datetime, timezone

# from parlant.core.common import ItemNotFoundError, UniqueId, Version, generate_id
# from parlant.core.persistence.document_database import DocumentDatabase, ObjectId

# StyleGuidelineId = NewType("StyleGuidelineId", str)


# @dataclass(frozen=True)
# class StyleGuidelineContent:
#     before: str
#     after: str
#     violation: str
#     style_guide: str


# @dataclass(frozen=True)
# class StyleGuideline:
#     id: StyleGuidelineId
#     creation_utc: datetime
#     content: StyleGuidelineContent


# class StyleGuidelineUpdateParams(TypedDict, total=False):
#     style_guideline_set: str
#     before: str
#     after: str
#     violation: str
#     style_guide: str


# class _StyleGuidelineDocument(TypedDict, total=False):
#     id: ObjectId
#     version: Version.String
#     creation_utc: str
#     style_guideline_set: str
#     before: str
#     after: str
#     violation: str
#     style_guide: str


# class GuidelineStore(ABC):
#     @abstractmethod
#     async def create_style_guideline(
#         self,
#         style_guideline_set: str,
#         before: str,
#         after: str,
#         violation: str,
#         style_guide: str,
#         creation_utc: Optional[datetime] = None,
#     ) -> StyleGuideline: ...

#     @abstractmethod
#     async def list_style_guidelines(
#         self,
#         style_guideline_set: str,
#     ) -> Sequence[StyleGuideline]: ...

#     @abstractmethod
#     async def read_style_guideline(
#         self,
#         style_guideline_set: str,
#         style_guideline_id: StyleGuidelineId,
#     ) -> StyleGuideline: ...

#     @abstractmethod
#     async def delete_style_guideline(
#         self,
#         style_guideline_set: str,
#         style_guideline_id: StyleGuidelineId,
#     ) -> None: ...

#     @abstractmethod
#     async def update_style_guideline(
#         self,
#         style_guideline_id: StyleGuidelineId,
#         params: StyleGuidelineUpdateParams,
#     ) -> StyleGuideline: ...


# class StyleGuidelineDocumentStore(GuidelineStore):
#     VERSION = Version.from_string("0.1.0")

#     def __init__(self, database: DocumentDatabase):
#         self._collection = database.get_or_create_collection(
#             name="style_guidelines",
#             schema=_StyleGuidelineDocument,
#         )

#     def _serialize(
#         self,
#         style_guideline: StyleGuideline,
#         style_guideline_set: str,
#     ) -> _StyleGuidelineDocument:
#         return _StyleGuidelineDocument(
#             id=ObjectId(style_guideline.id),
#             version=self.VERSION.to_string(),
#             creation_utc=style_guideline.creation_utc.isoformat(),
#             style_guideline_set=style_guideline_set,
#             before=style_guideline.content.before,
#             after=style_guideline.content.after,
#             violation=style_guideline.content.violation,
#             style_guide=style_guideline.content.style_guide,
#         )

#     def _deserialize(
#         self,
#         style_guideline_document: _StyleGuidelineDocument,
#     ) -> StyleGuideline:
#         return StyleGuideline(
#             id=StyleGuidelineId(style_guideline_document["id"]),
#             creation_utc=datetime.fromisoformat(style_guideline_document["creation_utc"]),
#             content=StyleGuidelineContent(
#                 before=style_guideline_document["before"],
#                 after=style_guideline_document["after"],
#                 violation=style_guideline_document["violation"],
#                 style_guide=style_guideline_document["style_guide"],
#             ),
#         )

#     @override
#     async def create_style_guideline(
#         self,
#         style_guideline_set: str,
#         before: str,
#         after: str,
#         violation: str,
#         style_guide: str,
#         creation_utc: Optional[datetime] = None,
#     ) -> StyleGuideline:
#         creation_utc = creation_utc or datetime.now(timezone.utc)

#         style_guideline = StyleGuideline(
#             id=StyleGuidelineId(generate_id()),
#             creation_utc=creation_utc,
#             content=StyleGuidelineContent(
#                 before=before,
#                 after=after,
#                 violation=violation,
#                 style_guide=style_guide,
#             ),
#         )

#         await self._collection.insert_one(
#             document=self._serialize(
#                 style_guideline=style_guideline,
#                 style_guideline_set=style_guideline_set,
#             )
#         )

#         return style_guideline

#     @override
#     async def list_style_guidelines(
#         self,
#         style_guideline_set: str,
#     ) -> Sequence[StyleGuideline]:
#         return [
#             self._deserialize(d)
#             for d in await self._collection.find(
#                 filters={"style_guideline_set": {"$eq": style_guideline_set}}
#             )
#         ]

#     @override
#     async def read_style_guideline(
#         self,
#         style_guideline_set: str,
#         style_guideline_id: StyleGuidelineId,
#     ) -> StyleGuideline:
#         guideline_document = await self._collection.find_one(
#             filters={
#                 "style_guideline_set": {"$eq": style_guideline_set},
#                 "id": {"$eq": style_guideline_id},
#             }
#         )

#         if not guideline_document:
#             raise ItemNotFoundError(
#                 item_id=UniqueId(style_guideline_id),
#                 message=f"with guideline_set '{style_guideline_set}'",
#             )

#         return self._deserialize(guideline_document)

#     @override
#     async def delete_style_guideline(
#         self,
#         style_guideline_set: str,
#         style_guideline_id: StyleGuidelineId,
#     ) -> None:
#         result = await self._collection.delete_one(
#             filters={
#                 "style_guideline_set": {"$eq": style_guideline_set},
#                 "id": {"$eq": style_guideline_id},
#             }
#         )

#         if not result.deleted_document:
#             raise ItemNotFoundError(
#                 item_id=UniqueId(style_guideline_id),
#                 message=f"with style guideline_set '{style_guideline_set}'",
#             )

#     @override
#     async def update_style_guideline(
#         self,
#         style_guideline_id: StyleGuidelineId,
#         params: StyleGuidelineUpdateParams,
#     ) -> StyleGuideline:
#         style_guideline_document = _StyleGuidelineDocument(
#             {
#                 **(
#                     {"style_guideline_set": params["style_guideline_set"]}
#                     if "style_guideline_set" in params
#                     else {}
#                 ),
#                 **({"before": params["before"]} if "before" in params else {}),
#                 **({"after": params["after"]} if "after" in params else {}),
#                 **({"violation": params["violation"]} if "violation" in params else {}),
#                 **({"style_guide": params["style_guide"]} if "style_guide" in params else {}),
#             }
#         )

#         result = await self._collection.update_one(
#             filters={"id": {"$eq": style_guideline_id}},
#             params=style_guideline_document,
#         )

#         assert result.updated_document

#         return self._deserialize(result.updated_document)
