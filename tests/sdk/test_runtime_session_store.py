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

"""SDK session calls reach the store that holds live sessions.

Managed hosting keeps live sessions in a runtime store that the app and the
engine reach, separate from the store that holds what the SDK declares at
startup. These tests split the two the same way.
"""

from typing import Callable, cast

import parlant.sdk as p
from parlant.adapters.db.transient import TransientDocumentDatabase
from parlant.core.sessions import SessionDocumentStore, SessionId, SessionStore
from parlant.core.store_provider import StoreProvider, StoreProviderHints, StoreT
from tests.sdk.utils import Context, SDKTest


class _RuntimeSessionStoreProvider(StoreProvider):
    def __init__(self, base: StoreProvider, runtime_store: SessionStore) -> None:
        self.base = base
        self.runtime_store = runtime_store

    def get_store(
        self, store_type: Callable[..., StoreT], hints: StoreProviderHints = {}
    ) -> StoreT:
        if store_type is SessionStore and hints.get("call_site") in {"app", "engine"}:
            return cast(StoreT, self.runtime_store)
        return self.base.get_store(store_type, hints)


class _RuntimeSessionStoreTest(SDKTest):
    async def test_run(self) -> None:
        async with SessionDocumentStore(TransientDocumentDatabase()) as runtime_store:
            self.runtime_store = runtime_store
            await super().test_run()

    async def configure_container(self, container: p.Container) -> p.Container:
        container = await super().configure_container(container)
        container[StoreProvider] = _RuntimeSessionStoreProvider(
            container[StoreProvider], self.runtime_store
        )
        return container

    async def setup(self, server: p.Server) -> None:
        self.agent = await server.create_agent(
            id="runtime-store-agent",
            name="Runtime Store Agent",
            instructions="Agent for testing where SDK session calls go",
        )

    async def create_live_session(self, ctx: Context) -> SessionId:
        created = await ctx.client.sessions.create(agent_id=self.agent.id, allow_greeting=False)
        return SessionId(created.id)


class Test_that_session_metadata_reads_and_writes_reach_the_runtime_session_store(
    _RuntimeSessionStoreTest
):
    async def run(self, ctx: Context) -> None:
        session_id = await self.create_live_session(ctx)
        metadata = p.SessionMetadata(session_id, {}, self.server)

        await metadata.set("needs_human", True)

        assert (await self.runtime_store.read_session(session_id)).metadata["needs_human"] is True
        assert await metadata.get("needs_human") is True

        await metadata.delete("needs_human")

        assert "needs_human" not in (await self.runtime_store.read_session(session_id)).metadata


class Test_that_server_get_session_and_session_update_reach_the_runtime_session_store(
    _RuntimeSessionStoreTest
):
    async def run(self, ctx: Context) -> None:
        session_id = await self.create_live_session(ctx)

        session = await self.server.get_session(id=session_id)
        await session.update(mode="manual", title="Handed to customer service")

        saved = await self.runtime_store.read_session(session_id)
        assert saved.mode == "manual"
        assert saved.title == "Handed to customer service"
