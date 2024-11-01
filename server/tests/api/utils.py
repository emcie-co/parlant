from typing import Callable, Optional, cast

from lagom import Container

from emcie.server.core.tools import ToolResult
from emcie.server.core.common import JSONSerializable
from emcie.server.core.agents import Agent, AgentId, AgentStore
from emcie.server.core.async_utils import Timeout
from emcie.server.core.context_variables import (
    ContextVariable,
    ContextVariableId,
    ContextVariableStore,
    ContextVariableValue,
)
from emcie.server.core.end_users import EndUser, EndUserId, EndUserStore
from emcie.server.core.glossary import GlossaryStore, Term
from emcie.server.core.guideline_tool_associations import GuidelineToolAssociationStore
from emcie.server.core.guidelines import Guideline, GuidelineStore
from emcie.server.core.mc import MC
from emcie.server.core.services.tools.service_registry import ServiceRegistry
from emcie.server.core.sessions import Event, MessageEventData, Session, SessionId, SessionStore
from emcie.server.core.tools import LocalToolService, ToolId


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

    event = await container[MC].post_event(
        session_id=session_id,
        kind="message",
        data=data,
    )

    if response_timeout:
        await container[MC].wait_for_update(
            session_id=session_id,
            min_offset=event.offset + 1,
            kinds=["message"],
            timeout=response_timeout,
        )

    return event
