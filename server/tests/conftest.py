import asyncio
from typing import Any, AsyncIterator
from fastapi.testclient import TestClient
from lagom import Container, Singleton
from pytest import fixture, Config

from emcie.server.api.app import create_app
from emcie.server.core.context_variables import ContextVariableDocumentStore, ContextVariableStore
from emcie.server.core.end_users import EndUserDocumentStore, EndUserStore
from emcie.server.core.guideline_connections import (
    GuidelineConnectionDocumentStore,
    GuidelineConnectionStore,
)
from emcie.server.core.guidelines import GuidelineDocumentStore, GuidelineStore
from emcie.server.core.persistence.transient_database import TransientDocumentDatabase
from emcie.server.core.sessions import (
    PollingSessionListener,
    SessionDocumentStore,
    SessionListener,
    SessionStore,
)
from emcie.server.core.tools import ToolDocumentStore, ToolStore
from emcie.server.engines.alpha.engine import AlphaEngine
from emcie.server.engines.common import Engine
from emcie.server.mc import MC
from emcie.server.core.agents import AgentDocumentStore, AgentStore
from emcie.server.core.persistence.document_database import (
    DocumentDatabase,
)
from emcie.server.core.guideline_tool_associations import (
    GuidelineToolAssociationDocumentStore,
    GuidelineToolAssociationStore,
)

from .test_utilities import SyncAwaiter


@fixture
async def sync_await() -> SyncAwaiter:
    return SyncAwaiter(asyncio.get_event_loop())


@fixture
def test_config(pytestconfig: Config) -> dict[str, Any]:
    return {"patience": 10}


@fixture
async def container() -> AsyncIterator[Container]:
    container = Container(log_undefined_deps=True)

    container[DocumentDatabase] = TransientDocumentDatabase
    container[AgentStore] = Singleton(AgentDocumentStore)
    container[GuidelineStore] = Singleton(GuidelineDocumentStore)
    container[GuidelineConnectionStore] = Singleton(GuidelineConnectionDocumentStore)
    container[ToolStore] = Singleton(ToolDocumentStore)
    container[SessionStore] = Singleton(SessionDocumentStore)
    container[ContextVariableStore] = Singleton(ContextVariableDocumentStore)
    container[EndUserStore] = Singleton(EndUserDocumentStore)
    container[GuidelineToolAssociationStore] = Singleton(GuidelineToolAssociationDocumentStore)
    container[SessionListener] = PollingSessionListener
    container[Engine] = AlphaEngine

    async with MC(container) as mc:
        container[MC] = mc
        yield container


@fixture
async def client(container: Container) -> AsyncIterator[TestClient]:
    app = await create_app(container)

    with TestClient(app) as client:
        yield client
