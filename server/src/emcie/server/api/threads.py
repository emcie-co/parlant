# Copyright (c) 2024 Emcie
# All rights reserved.
#
# This file and its contents are the property of Emcie and are strictly confidential.
# No part of this file may be reproduced, distributed, or transmitted in any form or by any means,
# including photocopying, recording, or other electronic or mechanical methods,
# without the prior written permission of Emcie.
#
# Website: https://emcie.co
from datetime import datetime
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import List

from emcie.server.threads import (
    MessageId,
    MessageRole,
    ThreadId,
    ThreadStore,
)


class MessageDTO(BaseModel):
    id: MessageId
    role: MessageRole
    content: str
    completed: bool
    creation_utc: datetime
    revision: int


class CreateThreadResponse(BaseModel):
    thread_id: ThreadId


class CreateMessageRequest(BaseModel):
    role: MessageRole
    content: str
    completed: bool = False


class CreateMessageResponse(BaseModel):
    message: MessageDTO


class PatchMessageRequest(BaseModel):
    target_revision: int
    content_delta: str
    completed: bool


class PatchMessageResponse(BaseModel):
    patched_message: MessageDTO


class ListMessagesResponse(BaseModel):
    messages: List[MessageDTO]


class ReadMessageResponse(BaseModel):
    message: MessageDTO


def create_router(thread_store: ThreadStore) -> APIRouter:
    router = APIRouter()

    @router.post("/")
    async def create_thread() -> CreateThreadResponse:
        thread = await thread_store.create_thread()

        return CreateThreadResponse(
            thread_id=thread.id,
        )

    @router.post("/{thread_id}/messages")
    async def create_message(
        thread_id: ThreadId,
        request: CreateMessageRequest,
    ) -> CreateMessageResponse:
        message = await thread_store.create_message(
            thread_id=thread_id,
            role=request.role,
            content=request.content,
            completed=request.completed,
        )

        return CreateMessageResponse(
            message=MessageDTO(
                id=message.id,
                role=message.role,
                content=message.content,
                completed=message.completed,
                creation_utc=message.creation_utc,
                revision=message.revision,
            )
        )

    @router.patch("/{thread_id}/messages/{message_id}")
    async def patch_message(
        thread_id: ThreadId,
        message_id: MessageId,
        request: PatchMessageRequest,
    ) -> PatchMessageResponse:
        try:
            patched_message = await thread_store.patch_message(
                thread_id=thread_id,
                message_id=message_id,
                target_revision=request.target_revision,
                content_delta=request.content_delta,
                completed=request.completed,
            )

            return PatchMessageResponse(
                patched_message=MessageDTO(
                    id=patched_message.id,
                    role=patched_message.role,
                    content=patched_message.content,
                    completed=patched_message.completed,
                    creation_utc=patched_message.creation_utc,
                    revision=patched_message.revision,
                )
            )
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(e),
            )

    @router.get("/{thread_id}/messages")
    async def list_messages(
        thread_id: ThreadId,
    ) -> ListMessagesResponse:
        messages = await thread_store.list_messages(thread_id)

        return ListMessagesResponse(
            messages=[
                MessageDTO(
                    id=m.id,
                    role=m.role,
                    content=m.content,
                    completed=m.completed,
                    creation_utc=m.creation_utc,
                    revision=m.revision,
                )
                for m in messages
            ]
        )

    @router.get("/{thread_id}/messages/{message_id}")
    async def read_message(
        thread_id: ThreadId,
        message_id: MessageId,
    ) -> ReadMessageResponse:
        message = await thread_store.read_message(
            thread_id=thread_id,
            message_id=message_id,
        )

        return ReadMessageResponse(
            message=MessageDTO(
                id=message.id,
                role=message.role,
                content=message.content,
                completed=message.completed,
                creation_utc=message.creation_utc,
                revision=message.revision,
            )
        )

    return router
