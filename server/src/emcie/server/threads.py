from dataclasses import dataclass
from typing import Dict, Iterable, Literal, NewType, Optional
from datetime import datetime
from tinydb import TinyDB, Query
from loguru import logger

from emcie.server import common


MessageId = NewType("MessageId", str)
ThreadId = NewType("ThreadId", str)


MessageRole = Literal["user", "assistant"]


@dataclass(frozen=True)
class Message:
    id: MessageId
    thread_id: ThreadId
    role: MessageRole
    content: str
    completed: bool
    creation_utc: datetime
    revision: int = 0


@dataclass(frozen=True)
class Thread:
    id: ThreadId


class ThreadStore:
    def __init__(
        self,
    ) -> None:
        self._threads: Dict[ThreadId, Thread] = {}
        self._messages: Dict[ThreadId, Dict[MessageId, Message]] = {}

    async def create_thread(self) -> Thread:
        thread_id = ThreadId(common.generate_id())
        self._threads[thread_id] = Thread(thread_id)
        self._messages[thread_id] = {}
        return self._threads[thread_id]

    async def create_message(
        self,
        thread_id: ThreadId,
        role: MessageRole,
        content: str,
        completed: bool,
        creation_utc: Optional[datetime] = None,
    ) -> Message:
        message = Message(
            id=MessageId(common.generate_id()),
            thread_id=thread_id,
            role=role,
            content=content,
            completed=completed,
            creation_utc=creation_utc or datetime.utcnow(),
        )

        self._messages[thread_id][message.id] = message

        return message

    async def patch_message(
        self,
        thread_id: ThreadId,
        message_id: MessageId,
        target_revision: int,
        content_delta: str,
        completed: bool,
    ) -> Message:
        message = self._messages[thread_id][message_id]

        if message.revision != target_revision:
            raise ValueError("Target revision is not the current one")

        message = Message(
            id=message.id,
            thread_id=thread_id,
            role=message.role,
            content=(message.content + content_delta),
            completed=completed,
            creation_utc=message.creation_utc,
            revision=(message.revision + 1),
        )

        self._messages[thread_id][message_id] = message

        return message

    async def list_messages(self, thread_id: ThreadId) -> Iterable[Message]:
        return self._messages[thread_id].values()

    async def read_message(
        self,
        thread_id: ThreadId,
        message_id: MessageId,
    ) -> Message:
        return self._messages[thread_id][message_id]


class JsonThreadStore(ThreadStore):
    def __init__(self, filename: str) -> None:
        super().__init__()
        self.db = TinyDB(filename, indent=2)

        self._threads = {
            t["id"]: Thread(id=t["id"]) for t in self.db.search(Query().type == "thread")
        }

        self._messages = {}

        for t in self._threads.values():
            self._messages[t.id] = {}

        messages = self.db.search(Query().type == "message")

        for m in messages:
            self._messages[m["thread_id"]][m["id"]] = Message(
                id=m["id"],
                thread_id=m["thread_id"],
                role=m["role"],
                content=m["content"],
                completed=m["completed"],
                creation_utc=m["creation_utc"],
                revision=m["revision"],
            )

    async def create_thread(self) -> Thread:
        thread = await super().create_thread()

        self.db.insert(
            {
                "type": "thread",
                "id": thread.id,
            }
        )

        return thread

    async def create_message(
        self,
        thread_id: ThreadId,
        role: Literal["user", "assistant"],
        content: str,
        completed: bool,
        creation_utc: datetime | None = None,
    ) -> Message:
        message = await super().create_message(
            thread_id,
            role,
            content,
            completed,
            creation_utc,
        )

        self.db.insert(
            {
                "type": "message",
                "id": message.id,
                "thread_id": message.thread_id,
                "role": message.role,
                "revision": message.revision,
                "completed": message.completed,
                "content": message.content,
                "creation_utc": str(message.creation_utc),
            }
        )

        return message

    async def patch_message(
        self,
        thread_id: ThreadId,
        message_id: MessageId,
        target_revision: int,
        content_delta: str,
        completed: bool,
    ) -> Message:
        message = await super().patch_message(
            thread_id,
            message_id,
            target_revision,
            content_delta,
            completed,
        )

        self.db.update(
            {
                "revision": message.revision,
                "content": message.content,
                "completed": message.completed,
            },
            Query().id == str(message_id),
        )

        return message

    async def list_messages(self, thread_id: ThreadId) -> Iterable[Message]:
        Q = Query()  # noqa
        messages = self.db.search((Q.type == "message") & (Q.thread_id == thread_id))
        return [
            Message(
                id=m["id"],
                thread_id=m["thread_id"],
                role=m["role"],
                content=m["content"],
                completed=m["completed"],
                creation_utc=m["creation_utc"],
                revision=m["revision"],
            )
            for m in messages
        ]

    async def read_message(self, thread_id: ThreadId, message_id: MessageId) -> Message:
        m = self.db.get(Query().id == message_id)
        return Message(
            id=m["id"],
            thread_id=m["thread_id"],
            role=m["role"],
            content=m["content"],
            completed=m["completed"],
            creation_utc=m["creation_utc"],
            revision=m["revision"],
        )
