import asyncio
from contextlib import contextmanager
import logging
from typing import Any, Awaitable, Callable, Generator, Iterator, Optional, TypeVar, cast

from lagom import Container

from parlant.adapters.nlp.openai import GPT_4o
from parlant.core.agents import Agent, AgentId, AgentStore
from parlant.core.application import Application
from parlant.core.async_utils import Timeout
from parlant.core.context_variables import (
    ContextVariable,
    ContextVariableId,
    ContextVariableStore,
    ContextVariableValue,
)
from parlant.core.end_users import EndUser, EndUserId, EndUserStore
from parlant.core.glossary import GlossaryStore, Term
from parlant.core.guideline_tool_associations import GuidelineToolAssociationStore
from parlant.core.guidelines import Guideline, GuidelineStore
from parlant.core.logging import Logger
from parlant.core.common import DefaultBaseModel, JSONSerializable
from parlant.core.services.tools.service_registry import ServiceRegistry
from parlant.core.sessions import Event, MessageEventData, Session, SessionStore, SessionId
from parlant.core.tools import LocalToolService, ToolId, ToolResult

T = TypeVar("T")


class NLPTestSchema(DefaultBaseModel):
    answer: bool


class SyncAwaiter:
    def __init__(self, event_loop: asyncio.AbstractEventLoop) -> None:
        self.event_loop = event_loop

    def __call__(self, awaitable: Generator[Any, None, T] | Awaitable[T]) -> T:
        return self.event_loop.run_until_complete(awaitable)  # type: ignore


class TestLogger(Logger):
    def __init__(self) -> None:
        self.logger = logging.getLogger("TestLogger")

    def debug(self, message: str) -> None:
        self.logger.debug(message)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message)

    def critical(self, message: str) -> None:
        self.logger.critical(message)

    @contextmanager
    def operation(self, name: str, props: dict[str, Any] = {}) -> Iterator[None]:
        yield


async def nlp_test(context: str, predicate: str) -> bool:
    schematic_generator = GPT_4o[NLPTestSchema](logger=TestLogger())

    inference = await schematic_generator.generate(
        prompt=f"""\
Given a context and a predicate, determine whether the
predicate applies with respect to the given context.
If the predicate applies, the answer is true;
otherwise, the answer is false.

Context: ###
{context}
###

Predicate: ###
{predicate}
###

Output JSON structure: ###
{{
    answer: <BOOL>
}}
###

Example #1: ###
{{
    answer: true
}}
###

Example #2: ###
{{
    answer: false
}}
###
""",
        hints={"temperature": 0.0, "strict": True},
    )
    return inference.content.answer


async def create_agent(container: Container, name: str) -> Agent:
    return await container[AgentStore].create_agent(name="test-agent")


async def create_end_user(container: Container, name: str) -> EndUser:
    return await container[EndUserStore].create_end_user(
        name=name,
        email="test@user.com",
    )


async def create_session(
    container: Container,
    agent_id: AgentId,
    end_user_id: Optional[EndUserId] = None,
    title: Optional[str] = None,
) -> Session:
    return await container[SessionStore].create_session(
        end_user_id or (await create_end_user(container, "Auto-Created User")).id,
        agent_id=agent_id,
        title=title,
    )


async def create_term(
    container: Container,
    agent_id: AgentId,
    name: str,
    description: str,
    synonyms: list[str],
) -> Term:
    return await container[GlossaryStore].create_term(
        term_set=agent_id,
        name=name,
        description=description,
        synonyms=synonyms,
    )


async def create_context_variable(
    container: Container,
    agent_id: AgentId,
    name: str,
) -> ContextVariable:
    return await container[ContextVariableStore].create_variable(
        variable_set=agent_id,
        name=name,
        description="",
        tool_id=None,
        freshness_rules=None,
    )


async def set_context_variable_value(
    container: Container,
    agent_id: AgentId,
    variable_id: ContextVariableId,
    key: str,
    data: JSONSerializable,
) -> ContextVariableValue:
    return await container[ContextVariableStore].update_value(
        variable_set=agent_id,
        key=key,
        variable_id=variable_id,
        data=data,
    )


async def create_guideline(
    container: Container,
    agent_id: AgentId,
    predicate: str,
    action: str,
    tool_function: Optional[Callable[[], ToolResult]] = None,
) -> Guideline:
    local_tool_service = container[LocalToolService]

    guideline = await container[GuidelineStore].create_guideline(
        guideline_set=agent_id,
        predicate=predicate,
        action=action,
    )

    if tool_function:
        local_tool_service = cast(
            LocalToolService, await container[ServiceRegistry].read_tool_service("local")
        )

        existing_tools = await local_tool_service.list_tools()

        tool = next((t for t in existing_tools if t.name == tool_function.__name__), None)

        if not tool:
            tool = await local_tool_service.create_tool(
                name=tool_function.__name__,
                module_path=tool_function.__module__,
                description="",
                parameters={},
                required=[],
            )

        await container[GuidelineToolAssociationStore].create_association(
            guideline_id=guideline.id,
            tool_id=ToolId("local", tool_function.__name__),
        )

    return guideline


async def read_reply(
    container: Container,
    session_id: SessionId,
    user_event_offset: int,
) -> Event:
    return next(
        iter(
            await container[SessionStore].list_events(
                session_id=session_id,
                source="ai_agent",
                min_offset=user_event_offset,
                kinds=["message"],
            )
        )
    )


async def post_message(
    container: Container,
    session_id: SessionId,
    message: str,
    response_timeout: Timeout = Timeout.none(),
) -> Event:
    end_user_id = (await container[SessionStore].read_session(session_id)).end_user_id
    end_user = await container[EndUserStore].read_end_user(end_user_id)

    data: MessageEventData = {
        "message": message,
        "participant": {
            "id": end_user_id,
            "display_name": end_user.name,
        },
    }

    event = await container[Application].post_event(
        session_id=session_id,
        kind="message",
        data=data,
    )

    if response_timeout:
        await container[Application].wait_for_update(
            session_id=session_id,
            min_offset=event.offset + 1,
            kinds=["message"],
            timeout=response_timeout,
        )

    return event
